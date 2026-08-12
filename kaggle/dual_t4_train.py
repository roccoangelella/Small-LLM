#!/usr/bin/env python3
"""Execute a pinned Kaggle trainer worktree as exact-batch two-T4 DDP.

This file is deliberately a Kaggle execution shim rather than a generic trainer
feature. Model, optimizer, scheduler, dataset, checkpoint, and W&B semantics
come from the experiment's pinned worktree. The shim adds only the qualified
execution topology:

* two replicated Tesla T4 ranks;
* one 16-sequence global optimizer block split 8/8;
* microbatch four, with ``no_sync`` on each non-final local microbatch;
* DDP-average compensation via ``world_size * local_loss / global_tokens``;
* synchronized non-finite/overflow decisions before either rank can step;
* rank-zero-only validation, checkpointing, remote publication, and telemetry;
* raw-model checkpoint serialization so no ``module.`` keys enter snapshots.

Modal does not import or invoke this module.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

WORLD_SIZE = 2
SEQUENCES_PER_BLOCK = 16
MICROBATCH_SIZE = 4
CONTEXT_LENGTH = 2048
AUTOTUNE_CONFIG_CAP = 6
EXPECTED_TORCH = "2.10.0"
EXPECTED_CUDA = "12.8"
EXPECTED_TRITON = "3.6.0"
EXPECTED_FLA = "0.5.2"


def _arguments(argv: Sequence[str] | None) -> tuple[Path, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--worktree", type=Path, required=True)
    args, trainer_argv = parser.parse_known_args(argv)
    worktree = args.worktree.resolve()
    if not (worktree / "trainer").is_dir() or not (worktree / "model").is_dir():
        raise SystemExit(f"pinned trainer worktree is invalid: {worktree}")
    return worktree, trainer_argv


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def _require_runtime(torch: Any) -> None:
    actual = {
        "torch": torch.__version__.split("+", 1)[0],
        "cuda": str(torch.version.cuda),
        "triton": _distribution_version("triton"),
        "fla-core": _distribution_version("fla-core"),
    }
    expected = {
        "torch": EXPECTED_TORCH,
        "cuda": EXPECTED_CUDA,
        "triton": EXPECTED_TRITON,
        "fla-core": EXPECTED_FLA,
    }
    mismatches = {
        key: {"expected": value, "actual": actual[key]}
        for key, value in expected.items()
        if actual[key] != value
    }
    if mismatches:
        raise RuntimeError(
            "Kaggle DDP runtime drifted from the qualified dual-T4 stack: "
            f"{mismatches}"
        )


def _representative_config_indices(count: int, cap: int) -> list[int]:
    if count <= 0:
        return []
    if cap >= count:
        return list(range(count))
    if cap <= 1:
        return [count // 2]
    return [round(index * (count - 1) / (cap - 1)) for index in range(cap)]


def _install_bounded_autotune() -> None:
    """Reuse the exact qualification-time six-candidate Triton policy."""

    import triton.runtime.autotuner as triton_autotuner

    current = triton_autotuner.Autotuner.prune_configs
    if getattr(current, "_small_llm_kaggle_ddp_bounded", False):
        return
    original = current

    def prune_configs(self: Any, kwargs: dict[str, Any]) -> list[Any]:
        configs = list(original(self, kwargs))
        if len(configs) <= AUTOTUNE_CONFIG_CAP:
            return configs
        indices = _representative_config_indices(len(configs), AUTOTUNE_CONFIG_CAP)
        return [configs[index] for index in indices]

    prune_configs._small_llm_kaggle_ddp_bounded = True  # type: ignore[attr-defined]
    triton_autotuner.Autotuner.prune_configs = prune_configs


def _with_raw_model(engine: Any, function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    wrapper = engine.model
    raw = getattr(engine, "_small_llm_raw_model", wrapper)
    if raw is wrapper:
        return function(engine, *args, **kwargs)
    engine.model = raw
    try:
        return function(engine, *args, **kwargs)
    finally:
        engine.model = wrapper


def _local_scaler_found_inf(engine: Any) -> bool:
    """Read the inf result recorded by the pinned PyTorch 2.10 GradScaler."""

    if not engine.scaler.is_enabled():
        return False
    states = getattr(engine.scaler, "_per_optimizer_states", None)
    if states is None:
        raise RuntimeError("qualified GradScaler no longer exposes optimizer inf state")
    state = states.get(id(engine.optimizer))
    if not isinstance(state, dict):
        raise RuntimeError("GradScaler did not record optimizer state after unscale")
    found = state.get("found_inf_per_device")
    if not isinstance(found, dict) or not found:
        raise RuntimeError("GradScaler did not record found_inf after unscale")
    return any(float(value.detach().item()) != 0.0 for value in found.values())


def _prewarm_raw_model(engine: Any, *, rank: int) -> None:
    """Populate Triton/FLA caches once without changing optimizer or token state."""

    if rank != 0:
        return
    import torch
    from trainer.precision import autocast_context

    raw_model = engine.model
    before_step = int(engine.global_step)
    before_tokens = int(engine.consumed_tokens)
    before_scale = float(engine.scaler.get_scale())
    engine.optimizer.zero_grad(set_to_none=True)
    inputs = torch.zeros(
        (MICROBATCH_SIZE, CONTEXT_LENGTH),
        dtype=torch.long,
        device=engine.device,
    )
    print(
        "[kaggle-ddp] cold FLA/Triton prewarm on rank 0 at local shape "
        f"{MICROBATCH_SIZE}x{CONTEXT_LENGTH}; no optimizer step is executed",
        flush=True,
    )
    started = time.perf_counter()
    with autocast_context(engine.config.precision, engine.device):
        logits = raw_model(inputs)
        # Backpropagating one vocabulary column reaches the entire decoder while
        # avoiding an extra full-vocabulary FP32 allocation on a 16-GiB T4.
        objective = logits[..., 0].float().mean()
    objective.backward()
    torch.cuda.synchronize(engine.device)
    engine.optimizer.zero_grad(set_to_none=True)
    del objective, logits, inputs
    torch.cuda.empty_cache()
    if (
        engine.global_step != before_step
        or engine.consumed_tokens != before_tokens
        or float(engine.scaler.get_scale()) != before_scale
    ):
        raise RuntimeError("DDP prewarm mutated scientific trainer state")
    print(
        f"[kaggle-ddp] prewarm complete in {time.perf_counter() - started:.2f}s; "
        "DDP ranks will reuse the selected Triton configs",
        flush=True,
    )


def _distributed_train_step(engine: Any, batch: Any) -> Any:
    import torch
    import torch.distributed as dist
    from torch.nn import functional as F
    from trainer.precision import autocast_context
    import trainer.step as pinned_step

    if batch.split != "train" or batch.sequence_count != SEQUENCES_PER_BLOCK:
        raise ValueError(
            f"Kaggle DDP requires a {SEQUENCES_PER_BLOCK}-sequence train block"
        )
    rank = int(engine._small_llm_rank)
    world_size = int(engine._small_llm_world_size)
    if world_size != WORLD_SIZE:
        raise RuntimeError(f"Kaggle DDP requires world_size={WORLD_SIZE}")
    if engine.config.microbatch_size != MICROBATCH_SIZE:
        raise RuntimeError(f"Kaggle DDP requires microbatch={MICROBATCH_SIZE}")

    rows_per_rank = batch.sequence_count // world_size
    if rows_per_rank * world_size != batch.sequence_count:
        raise RuntimeError("optimizer block cannot be split evenly across DDP ranks")
    rank_start = rank * rows_per_rank
    rank_stop = rank_start + rows_per_rank

    started = time.perf_counter()
    if engine.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(engine.device)
    retries, loss_value, grad_value, lr = 0, math.nan, math.nan, math.nan
    initial_scaler_scale = float(engine.scaler.get_scale())
    scaler_scale = initial_scaler_scale
    overflow_retry_limit = (
        pinned_step._fp16_overflow_retry_limit(
            engine.scaler, engine.config.max_overflow_retries
        )
        if engine.scaler.is_enabled()
        else engine.config.max_overflow_retries
    )
    role_gradient_norms: dict[str, float] = {}
    update_statistics: dict[str, object] = {}
    gradient_clipped = False

    while True:
        engine.optimizer.zero_grad(set_to_none=True)
        next_tokens = engine.consumed_tokens + batch.target_token_count
        lr = engine.scheduler.prepare_step(next_tokens)
        local_loss = torch.zeros((), dtype=torch.float32, device=engine.device)
        local_forward_finite = True
        starts = list(range(rank_start, rank_stop, engine.config.microbatch_size))
        for local_index, start in enumerate(starts):
            stop = min(rank_stop, start + engine.config.microbatch_size)
            micro_inputs = batch.input_ids[start:stop].to(
                device=engine.device, non_blocking=True
            )
            micro_labels = batch.labels[start:stop].to(
                device=engine.device, non_blocking=True
            )
            sync_context = contextlib.nullcontext()
            if local_index + 1 < len(starts):
                sync_context = engine.model.no_sync()
            with sync_context:
                with autocast_context(engine.config.precision, engine.device):
                    logits = engine.model(micro_inputs)
                    if logits.ndim != 3 or logits.shape[:2] != micro_labels.shape:
                        raise RuntimeError("model logits do not match training labels")
                    loss_sum = F.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]),
                        micro_labels.reshape(-1),
                        reduction="sum",
                    )
                local_forward_finite = local_forward_finite and bool(
                    torch.isfinite(loss_sum)
                )
                local_loss += loss_sum.detach().float()
                # DDP averages gradients across ranks. Multiplying each local
                # contribution by world_size recovers the exact serial global
                # block gradient after the all-reduce average.
                scaled = world_size * loss_sum / batch.target_token_count
                engine.scaler.scale(scaled).backward()
            del logits, loss_sum, scaled, micro_inputs, micro_labels

        engine.scaler.unscale_(engine.optimizer)
        scaler_found_inf = _local_scaler_found_inf(engine)
        role_gradient_norms = pinned_step._optimizer_gradient_norms(engine.optimizer)
        raw_model = engine._small_llm_raw_model
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            raw_model.parameters(), engine.config.max_grad_norm
        )
        finite_gradient = bool(torch.isfinite(gradient_norm))
        grad_value = float(gradient_norm.detach())
        gradient_clipped = finite_gradient and grad_value > float(engine.config.max_grad_norm)

        # All failure decisions are synchronized before either GradScaler is
        # allowed to invoke the underlying optimizer.
        flags = torch.tensor(
            [
                int(not local_forward_finite),
                int(not finite_gradient),
                int(scaler_found_inf),
            ],
            dtype=torch.int32,
            device=engine.device,
        )
        dist.all_reduce(flags, op=dist.ReduceOp.MAX)
        forward_bad, gradient_bad, any_scaler_inf = (
            bool(int(item)) for item in flags.tolist()
        )
        if forward_bad:
            engine.optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError(
                "non-finite FP16 training loss on at least one DDP rank; "
                "loss-scale reduction cannot repair a forward loss "
                f"(block={batch.block_id})"
            )

        if any_scaler_inf:
            # With DDP-reduced gradients every rank must observe the same inf.
            # Prove that before calling scaler.step(): if any rank disagrees, fail
            # closed rather than risk one optimizer mutating while another skips.
            unanimous = torch.tensor(
                int(scaler_found_inf), dtype=torch.int32, device=engine.device
            )
            dist.all_reduce(unanimous, op=dist.ReduceOp.MIN)
            if not bool(int(unanimous.item())):
                engine.optimizer.zero_grad(set_to_none=True)
                raise RuntimeError(
                    "asymmetric GradScaler found_inf across DDP ranks before optimizer step"
                )
            retries += 1
            engine.overflow_events += 1
            scale_before = float(engine.scaler.get_scale())
            pinned_step._clear_optimizer_step_statistics(engine.optimizer)
            # found_inf is true on every rank, so GradScaler skips the underlying
            # optimizer on every rank, then backs the scale off identically.
            engine.scaler.step(engine.optimizer)
            engine.scaler.update()
            scaler_scale = float(engine.scaler.get_scale())
            if scaler_scale >= scale_before:
                raise RuntimeError("GradScaler did not reduce scale after synchronized overflow")
            if retries > overflow_retry_limit:
                engine.optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(
                    "FP16 DDP optimizer step remained non-finite after dynamic scale "
                    "calibration; block remains unacknowledged "
                    f"(block={batch.block_id}, attempts={retries}, "
                    f"initial_scale={initial_scaler_scale:g}, "
                    f"current_scale={scaler_scale:g}, retry_limit={overflow_retry_limit})"
                )
            continue

        if gradient_bad:
            engine.optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError(
                "non-finite DDP gradient norm without GradScaler found_inf; "
                "loss-scale replay cannot safely repair this condition"
            )

        # Confirm the global loss while all ranks are still at the same atomic
        # boundary, then take the optimizer step on both replicas.
        dist.all_reduce(local_loss, op=dist.ReduceOp.SUM)
        scale_before = float(engine.scaler.get_scale())
        pinned_step._clear_optimizer_step_statistics(engine.optimizer)
        engine.scaler.step(engine.optimizer)
        engine.scaler.update()
        scaler_scale = float(engine.scaler.get_scale())
        if engine.scaler.is_enabled() and scaler_scale < scale_before:
            # We already proved found_inf was false on both ranks.
            raise RuntimeError(
                "GradScaler reported an unexpected post-synchronization overflow"
            )

        update_statistics = pinned_step._optimizer_step_statistics(engine.optimizer)
        loss_value = float(local_loss / batch.target_token_count)
        engine.consumed_tokens = next_tokens
        engine.global_step += 1
        engine.scheduler.commit(engine.consumed_tokens)
        break

    elapsed = max(time.perf_counter() - started, 1e-12)
    if engine.device.type == "cuda":
        peak = int(torch.cuda.max_memory_allocated(engine.device))
        peak_reserved = int(torch.cuda.max_memory_reserved(engine.device))
    else:
        peak = 0
        peak_reserved = 0
    return pinned_step.StepMetrics(
        step=engine.global_step,
        block_id=batch.block_id,
        loss=loss_value,
        learning_rate=lr,
        gradient_norm=grad_value,
        sequences=batch.sequence_count,
        target_tokens=batch.target_token_count,
        consumed_tokens=engine.consumed_tokens,
        elapsed_seconds=elapsed,
        tokens_per_second=batch.target_token_count / elapsed,
        overflow_retries=retries,
        peak_memory_bytes=peak,
        grad_scaler_scale=scaler_scale,
        gradient_clipped=gradient_clipped,
        overflow_events_total=engine.overflow_events,
        peak_reserved_memory_bytes=peak_reserved,
        optimizer_gradient_norms=role_gradient_norms,
        optimizer_update_statistics=update_statistics,
    )


def _install_pinned_trainer_ddp(*, rank: int, local_rank: int, world_size: int) -> Any:
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    import trainer.cli as trainer_cli
    import trainer.engine as trainer_engine
    import trainer.evaluation as trainer_evaluation
    import trainer.session as trainer_session
    import trainer.step as trainer_step

    original_setup = trainer_cli.setup
    original_evaluate_batches = trainer_engine.evaluate_batches
    original_engine_state_dict = trainer_engine.engine_state_dict
    original_load_engine_state = trainer_engine.load_engine_state
    original_session_step = trainer_session.TrainingSession.step
    original_session_save = trainer_session.TrainingSession.save_checkpoint
    original_remote = trainer_cli.configure_remote_publication
    original_wandb = trainer_cli.configure_wandb

    def portable_state_dict(engine: Any) -> dict[str, object]:
        return _with_raw_model(engine, original_engine_state_dict)

    def portable_load_state(engine: Any, state: Mapping[str, object]) -> None:
        _with_raw_model(engine, original_load_engine_state, state)

    def distributed_evaluate(
        engine: Any,
        batches: Any,
        *,
        maximum_batches: int | None = None,
        microbatch_size: int = 1,
    ) -> dict[str, float | int]:
        if rank != 0:
            # Rank 0 alone owns validation. Rank 1 reaches the next session-step
            # barrier and waits there until rank 0 finishes all side effects.
            return {"loss": 0.0, "perplexity": 1.0, "target_tokens": 0, "blocks": 0}
        return _with_raw_model(
            engine,
            original_evaluate_batches,
            batches,
            maximum_batches=maximum_batches,
            microbatch_size=microbatch_size,
        )

    def synchronized_session_step(self: Any, timeout: float | None = None) -> Any:
        # This barrier keeps rank 1 from beginning the next optimizer block while
        # rank 0 is validating/checkpointing/publishing the previous boundary.
        dist.barrier()
        return original_session_step(self, timeout=timeout)

    def primary_only_save(self: Any, *args: Any, **kwargs: Any) -> Any:
        if rank != 0:
            return None
        return original_session_save(self, *args, **kwargs)

    def primary_remote(args: Any) -> Any:
        return original_remote(args) if rank == 0 else None

    def primary_wandb(*args: Any, **kwargs: Any) -> Any:
        return original_wandb(*args, **kwargs) if rank == 0 else None

    def setup(args: Any) -> Any:
        if args.sequences_per_block != SEQUENCES_PER_BLOCK:
            raise RuntimeError(
                f"Kaggle DDP requires sequences_per_block={SEQUENCES_PER_BLOCK}"
            )
        if args.microbatch_size != MICROBATCH_SIZE:
            raise RuntimeError(f"Kaggle DDP requires microbatch={MICROBATCH_SIZE}")
        if args.precision != "fp16" or args.architecture != "gdn2_hybrid":
            raise RuntimeError("Kaggle DDP is qualified only for FP16 gdn2_hybrid training")
        requested_device = args.device
        args.device = f"cuda:{local_rank}"
        try:
            result = original_setup(args)
        finally:
            args.device = requested_device
        model_config, trainer_config, engine, session, coordinator = result

        # Resume, when requested, has already loaded the topology-neutral raw
        # checkpoint at this point.
        _prewarm_raw_model(engine, rank=rank)
        dist.barrier()
        raw_model = engine.model
        engine._small_llm_raw_model = raw_model
        engine._small_llm_rank = rank
        engine._small_llm_local_rank = local_rank
        engine._small_llm_world_size = world_size
        engine.model = DistributedDataParallel(
            raw_model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
        )
        return model_config, trainer_config, engine, session, coordinator

    trainer_engine.train_step = _distributed_train_step
    trainer_step.train_step = _distributed_train_step
    trainer_engine.engine_state_dict = portable_state_dict
    trainer_engine.load_engine_state = portable_load_state
    trainer_engine.evaluate_batches = distributed_evaluate
    trainer_evaluation.evaluate_batches = distributed_evaluate
    trainer_session.TrainingSession.step = synchronized_session_step
    trainer_session.TrainingSession.save_checkpoint = primary_only_save
    trainer_cli.setup = setup
    trainer_cli.configure_remote_publication = primary_remote
    trainer_cli.configure_wandb = primary_wandb
    if rank != 0:
        trainer_cli.print = lambda *args, **kwargs: None
    return trainer_cli


def main(argv: Sequence[str] | None = None) -> int:
    worktree, trainer_argv = _arguments(argv)
    # The pinned worktree must win module resolution even though this shim lives
    # in the controlling checkout.
    sys.path.insert(0, str(worktree))

    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ["TRITON_CACHE_AUTOTUNING"] = "1"
    os.environ["FLA_CACHE_RESULTS"] = "1"

    import torch
    import torch.distributed as dist

    _require_runtime(torch)
    try:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    except (KeyError, ValueError) as error:
        raise RuntimeError("Kaggle dual-T4 training must be launched through torchrun") from error
    if world_size != WORLD_SIZE:
        raise RuntimeError(f"Kaggle dual-T4 training requires world_size={WORLD_SIZE}")
    if torch.cuda.device_count() < WORLD_SIZE:
        raise RuntimeError("Kaggle dual-T4 training requires two visible CUDA devices")
    names = [torch.cuda.get_device_name(index) for index in range(WORLD_SIZE)]
    if any(name != "Tesla T4" for name in names):
        raise RuntimeError(f"Kaggle dual-T4 training requires two Tesla T4 GPUs; found {names}")

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    _install_bounded_autotune()
    if rank == 0:
        print(
            "[kaggle-ddp] standard execution: 2x Tesla T4, global block=16, "
            "8 sequences/rank, microbatch=4, exact-batch DDP",
            flush=True,
        )
    trainer_cli = _install_pinned_trainer_ddp(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
    )
    exit_code = 1
    try:
        exit_code = int(trainer_cli.main(list(trainer_argv)))
        dist.barrier()
        return exit_code
    finally:
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                if exit_code == 0:
                    raise


if __name__ == "__main__":
    raise SystemExit(main())
