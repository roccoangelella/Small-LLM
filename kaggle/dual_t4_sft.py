#!/usr/bin/env python3
"""Run SFT as an exact global-token objective across Kaggle's two Tesla T4 GPUs."""
from __future__ import annotations

import argparse
import contextlib
from datetime import timedelta
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

SUPPORTED_MICROBATCH_SIZES = (1, 2, 4)
CONTROL_GROUP_TIMEOUT_SECONDS = 60 * 60


def _new_control_group(distributed: Any) -> Any:
    """Create the CPU group used while rank zero owns long side effects.

    Rank one may reach a cadence boundary while rank zero is still validating,
    generating behavior probes, saving, or publishing. Waiting in NCCL there
    starts the default ten-minute CUDA collective watchdog even though no GPU
    collective is expected yet. A bounded Gloo group keeps that wait entirely
    on the control plane; both ranks return to NCCL together for the next DDP
    optimizer block.
    """

    return distributed.new_group(
        backend="gloo",
        timeout=timedelta(seconds=CONTROL_GROUP_TIMEOUT_SECONDS),
    )


def _control_barrier(distributed: Any, group: Any) -> None:
    distributed.barrier(group=group)


def _arguments(argv: Sequence[str] | None) -> tuple[Path, int, int, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--sft-fraction-numerator", type=int, required=True)
    parser.add_argument("--sft-fraction-denominator", type=int, required=True)
    args, trainer_argv = parser.parse_known_args(argv)
    worktree = args.worktree.resolve()
    if not (worktree / "trainer").is_dir() or not (worktree / "post_training" / "sft").is_dir():
        raise SystemExit(f"pinned SFT worktree is invalid: {worktree}")
    numerator = int(args.sft_fraction_numerator)
    denominator = int(args.sft_fraction_denominator)
    if numerator <= 0 or denominator <= 0 or numerator >= denominator:
        raise SystemExit("SFT fraction must be in (0, 1)")
    return worktree, numerator, denominator, list(trainer_argv)


def _rank_row_indices(sequence_count: int, rank: int, world_size: int) -> tuple[int, ...]:
    if sequence_count <= 0:
        raise ValueError("sequence_count must be positive")
    if world_size <= 0 or rank < 0 or rank >= world_size:
        raise ValueError("invalid DDP rank geometry")
    return tuple(range(rank, sequence_count, world_size))


def _argument_value(argv: Sequence[str], flag: str) -> str:
    values = list(argv)
    try:
        index = values.index(flag)
    except ValueError as error:
        raise RuntimeError(f"required trainer argument is missing: {flag}") from error
    if index + 1 >= len(values):
        raise RuntimeError(f"trainer argument has no value: {flag}")
    return str(values[index + 1])


def _distributed_sft_train_step(engine: Any, batch: Any) -> Any:
    import torch
    import torch.distributed as dist
    from torch.nn import functional as F
    from trainer.precision import autocast_context
    import trainer.step as pinned_step
    from trainer.types import IGNORE_INDEX

    rank = int(engine._small_llm_rank)
    world_size = int(engine._small_llm_world_size)
    if batch.split != "train" or batch.sequence_count <= 0:
        raise ValueError("SFT DDP requires a non-empty train block")
    if world_size != 2:
        raise RuntimeError("SFT DDP requires world_size=2")
    if engine.config.microbatch_size not in SUPPORTED_MICROBATCH_SIZES:
        raise RuntimeError(
            f"SFT DDP microbatch_size must be one of {SUPPORTED_MICROBATCH_SIZES}"
        )

    ordered_inputs, ordered_labels = pinned_step._ordered_batch_tensors(batch)
    indices = _rank_row_indices(batch.sequence_count, rank, world_size)
    if indices:
        index_tensor = torch.tensor(indices, dtype=torch.long)
        local_inputs = ordered_inputs.index_select(0, index_tensor)
        local_labels = ordered_labels.index_select(0, index_tensor)
    else:
        local_inputs = ordered_inputs[:0]
        local_labels = ordered_labels[:0]

    rows_per_rank = math.ceil(batch.sequence_count / world_size)
    missing = rows_per_rank - int(local_inputs.shape[0])
    if missing:
        dummy_inputs = ordered_inputs[:1].repeat(missing, 1)
        dummy_labels = torch.full_like(dummy_inputs, IGNORE_INDEX)
        local_inputs = torch.cat((local_inputs, dummy_inputs), dim=0)
        local_labels = torch.cat((local_labels, dummy_labels), dim=0)

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
        starts = list(range(0, rows_per_rank, engine.config.microbatch_size))

        for local_index, start in enumerate(starts):
            stop = min(rows_per_rank, start + engine.config.microbatch_size)
            micro_inputs = local_inputs[start:stop]
            micro_labels = local_labels[start:stop]
            active_columns = micro_labels.ne(IGNORE_INDEX).any(dim=0)
            nonzero = active_columns.nonzero(as_tuple=False)
            if nonzero.numel():
                width = int(nonzero[-1].item()) + 1
                micro_inputs = micro_inputs[:, :width]
                micro_labels = micro_labels[:, :width]
            micro_inputs = micro_inputs.to(device=engine.device, non_blocking=True)
            micro_labels = micro_labels.to(device=engine.device, non_blocking=True)

            sync_context = (
                engine.model.no_sync()
                if local_index + 1 < len(starts)
                else contextlib.nullcontext()
            )
            with sync_context:
                with autocast_context(engine.config.precision, engine.device):
                    logits = engine.model(micro_inputs)
                    if logits.ndim != 3 or logits.shape[:2] != micro_labels.shape:
                        raise RuntimeError("model logits do not match SFT labels")
                    if bool(micro_labels.ne(IGNORE_INDEX).any()):
                        loss_sum = F.cross_entropy(
                            logits.reshape(-1, logits.shape[-1]),
                            micro_labels.reshape(-1),
                            reduction="sum",
                            ignore_index=IGNORE_INDEX,
                        )
                    else:
                        loss_sum = logits[..., 0].float().sum() * 0.0
                local_forward_finite = local_forward_finite and bool(torch.isfinite(loss_sum))
                local_loss += loss_sum.detach().float()
                scaled = world_size * loss_sum / batch.target_token_count
                engine.scaler.scale(scaled).backward()
            del logits, loss_sum, scaled, micro_inputs, micro_labels

        engine.scaler.unscale_(engine.optimizer)
        scaler_found_inf = engine._small_llm_local_scaler_found_inf(engine)
        role_gradient_norms = pinned_step._optimizer_gradient_norms(engine.optimizer)
        raw_model = engine._small_llm_raw_model
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            raw_model.parameters(), engine.config.max_grad_norm
        )
        finite_gradient = bool(torch.isfinite(gradient_norm))
        grad_value = float(gradient_norm.detach())
        gradient_clipped = finite_gradient and grad_value > float(engine.config.max_grad_norm)

        flags = torch.tensor(
            [int(not local_forward_finite), int(not finite_gradient), int(scaler_found_inf)],
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
                f"non-finite FP16 SFT loss on a DDP rank (block={batch.block_id})"
            )

        if any_scaler_inf:
            unanimous = torch.tensor(
                int(scaler_found_inf), dtype=torch.int32, device=engine.device
            )
            dist.all_reduce(unanimous, op=dist.ReduceOp.MIN)
            if not bool(int(unanimous.item())):
                engine.optimizer.zero_grad(set_to_none=True)
                raise RuntimeError("asymmetric GradScaler found_inf across SFT DDP ranks")
            retries += 1
            engine.overflow_events += 1
            scale_before = float(engine.scaler.get_scale())
            pinned_step._clear_optimizer_step_statistics(engine.optimizer)
            engine.scaler.step(engine.optimizer)
            engine.scaler.update()
            scaler_scale = float(engine.scaler.get_scale())
            if scaler_scale >= scale_before:
                raise RuntimeError("GradScaler did not reduce scale after synchronized overflow")
            if retries > overflow_retry_limit:
                engine.optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(
                    "FP16 SFT optimizer step remained non-finite after dynamic scale calibration; "
                    f"block={batch.block_id}, attempts={retries}, initial_scale={initial_scaler_scale:g}"
                )
            continue

        if gradient_bad:
            engine.optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("non-finite SFT DDP gradient norm")

        dist.all_reduce(local_loss, op=dist.ReduceOp.SUM)
        scale_before = float(engine.scaler.get_scale())
        pinned_step._clear_optimizer_step_statistics(engine.optimizer)
        engine.scaler.step(engine.optimizer)
        engine.scaler.update()
        scaler_scale = float(engine.scaler.get_scale())
        if engine.scaler.is_enabled() and scaler_scale < scale_before:
            raise RuntimeError("unexpected post-synchronization SFT GradScaler overflow")
        update_statistics = pinned_step._optimizer_step_statistics(engine.optimizer)
        loss_value = float(local_loss / batch.target_token_count)
        engine.consumed_tokens = next_tokens
        engine.global_step += 1
        engine.scheduler.commit(engine.consumed_tokens)
        break

    elapsed = max(time.perf_counter() - started, 1e-12)
    peak = int(torch.cuda.max_memory_allocated(engine.device)) if engine.device.type == "cuda" else 0
    peak_reserved = int(torch.cuda.max_memory_reserved(engine.device)) if engine.device.type == "cuda" else 0
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


def _dummy_validation() -> dict[str, object]:
    return {
        "loss": 0.0,
        "perplexity": 1.0,
        "target_tokens": 0,
        "blocks": 0,
        "elapsed_seconds": 0.0,
    }


def _dummy_behavior() -> dict[str, object]:
    return {
        "summary": {
            "pass_rate": 0.0,
            "eos_termination_rate": 0.0,
            "runaway_rate": 0.0,
            "empty_rate": 0.0,
            "role_leak_rate": 0.0,
            "mean_response_tokens": 0.0,
            "mean_trigram_repetition": 0.0,
        },
        "cases": [],
    }


def _disable_secondary_remote_side_effects(sft_train: Any) -> None:
    """Make every remote cadence action a no-op on non-primary DDP ranks."""

    sft_train.CheckpointCoordinator.publish = lambda *args, **kwargs: None
    sft_train.cleanup_remote_publication = lambda *args, **kwargs: None


def _rewrite_summary_fraction(checkpoint_dir: Path, fraction: float) -> None:
    path = checkpoint_dir / "sft-summary.json"
    if not path.is_file():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("budget"), dict):
        payload["budget"]["fraction"] = fraction
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    worktree, numerator, denominator, trainer_argv = _arguments(argv)
    sys.path.insert(0, str(worktree))
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ["TRITON_CACHE_AUTOTUNING"] = "1"
    os.environ["FLA_CACHE_RESULTS"] = "1"

    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    import dual_t4_train as qualified
    import post_training.sft.train_cli as sft_train
    import trainer.engine as trainer_engine
    import trainer.session as trainer_session
    from trainer.model_artifact import download_verified_model_artifact

    qualified._require_runtime(torch)
    try:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    except (KeyError, ValueError) as error:
        raise RuntimeError("SFT dual-T4 training must be launched through torchrun") from error
    if world_size != qualified.WORLD_SIZE:
        raise RuntimeError(f"SFT dual-T4 training requires world_size={qualified.WORLD_SIZE}")
    if torch.cuda.device_count() < world_size:
        raise RuntimeError("SFT dual-T4 training requires two visible CUDA devices")
    names = [torch.cuda.get_device_name(index) for index in range(world_size)]
    if any(name != "Tesla T4" for name in names):
        raise RuntimeError(f"SFT dual-T4 training requires two Tesla T4 GPUs; found {names}")

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    control_group = _new_control_group(dist)
    qualified._install_bounded_autotune()

    original_engine = sft_train.TrainerEngine
    original_state_dict = trainer_engine.engine_state_dict
    original_load_state = trainer_engine.load_engine_state
    original_evaluate = trainer_engine.evaluate_batches
    original_session_step = trainer_session.TrainingSession.step
    original_behavior = sft_train.evaluate_behavior
    original_budget = sft_train.sft_budget_from_parent
    original_select_resume = sft_train._select_resume
    original_path_write_text = Path.write_text
    checkpoint_dir = Path(_argument_value(trainer_argv, "--checkpoint-dir")).resolve()
    microbatch_size = int(_argument_value(trainer_argv, "--microbatch-size"))
    if microbatch_size not in SUPPORTED_MICROBATCH_SIZES:
        raise RuntimeError(
            f"SFT DDP microbatch_size must be one of {SUPPORTED_MICROBATCH_SIZES}"
        )
    fraction = numerator / denominator

    class DistributedSFTTrainerEngine(original_engine):
        def __init__(self, model: Any, config: Any, *, device: Any = None, optimizer: Any = None) -> None:
            super().__init__(model, config, device=f"cuda:{local_rank}", optimizer=optimizer)
            if config.microbatch_size != microbatch_size:
                raise RuntimeError("SFT DDP profile and trainer microbatch sizes disagree")
            qualified._prewarm_raw_model(
                self,
                rank=rank,
                microbatch_size=microbatch_size,
            )
            _control_barrier(dist, control_group)
            raw_model = self.model
            self._small_llm_raw_model = raw_model
            self._small_llm_rank = rank
            self._small_llm_local_rank = local_rank
            self._small_llm_world_size = world_size
            self._small_llm_local_scaler_found_inf = qualified._local_scaler_found_inf
            self.model = DistributedDataParallel(
                raw_model,
                device_ids=[local_rank],
                output_device=local_rank,
                broadcast_buffers=False,
                gradient_as_bucket_view=True,
            )

        def train_batch(self, batch: Any) -> Any:
            return _distributed_sft_train_step(self, batch)

        def evaluate(self, batches: Any, *, maximum_batches: int | None = None) -> dict[str, float | int]:
            if rank != 0:
                return {"loss": 0.0, "perplexity": 1.0, "target_tokens": 0, "blocks": 0}
            return qualified._with_raw_model(
                self,
                original_evaluate,
                batches,
                maximum_batches=maximum_batches,
            )

        def state_dict(self) -> dict[str, object]:
            return qualified._with_raw_model(self, original_state_dict)

        def load_state_dict(self, state: Mapping[str, object]) -> None:
            qualified._with_raw_model(self, original_load_state, state)

    def synchronized_session_step(self: Any, timeout: float | None = None) -> Any:
        _control_barrier(dist, control_group)
        return original_session_step(self, timeout=timeout)

    def profile_budget(parent_consumed_tokens: int, **_: Any) -> int:
        return original_budget(
            parent_consumed_tokens,
            numerator=numerator,
            denominator=denominator,
        )

    def synchronized_resume(
        args: argparse.Namespace,
        *,
        token: str | None,
        expected_hashes: tuple[str, str, str],
        expected_identity: Mapping[str, object],
    ) -> tuple[Path, dict[str, object]] | None:
        if rank == 0:
            resumed = original_select_resume(
                args,
                token=token,
                expected_hashes=expected_hashes,
                expected_identity=expected_identity,
            )
            resume_step = -1
            if resumed is not None:
                resume_root, resume_info = resumed
                checkpoint_id = str(resume_info["checkpoint_id"])
                resume_step = int(resume_info["step"])
                if resume_info.get("transport") == "remote":
                    target = args.checkpoint_dir / checkpoint_id
                    if not target.exists():
                        shutil.copytree(resume_root, target)
            signal = torch.tensor([resume_step], dtype=torch.int64, device=f"cuda:{local_rank}")
            dist.broadcast(signal, src=0)
            return resumed

        signal = torch.tensor([-1], dtype=torch.int64, device=f"cuda:{local_rank}")
        dist.broadcast(signal, src=0)
        expected_step = int(signal.item())
        if expected_step < 0:
            return None
        local = sft_train._local_resume(
            args,
            expected_hashes=expected_hashes,
            expected_identity=expected_identity,
        )
        if local is None or int(local[1]["step"]) != expected_step:
            raise RuntimeError(
                "rank 1 could not load the exact SFT checkpoint selected by rank 0"
            )
        return local

    def stable_parent_checkpoint(
        *,
        repo_id: str,
        run_id: str,
        pointer: str = "best",
        token: str | None = None,
        revision: str | None = None,
        destination: Path | str | None = None,
    ) -> tuple[Path, dict[str, object]]:
        if destination is None:
            raise RuntimeError("SFT stable parent download requires an explicit destination")
        root, metadata = download_verified_model_artifact(
            repo_id=repo_id,
            run_id=run_id,
            token=token,
            revision=revision,
            destination=Path(destination),
        )
        return root, dict(metadata)

    sft_train.TrainerEngine = DistributedSFTTrainerEngine
    trainer_session.TrainingSession.step = synchronized_session_step
    sft_train.sft_budget_from_parent = profile_budget
    sft_train._select_resume = synchronized_resume
    sft_train.download_parent_checkpoint = stable_parent_checkpoint

    if rank == 0:
        def primary_behavior(model: Any, *args: Any, **kwargs: Any) -> dict[str, object]:
            return original_behavior(getattr(model, "module", model), *args, **kwargs)

        real_print = print

        def primary_print(*args: Any, **kwargs: Any) -> None:
            if len(args) == 1 and isinstance(args[0], str):
                try:
                    payload = json.loads(args[0])
                except (json.JSONDecodeError, TypeError):
                    payload = None
                if isinstance(payload, dict) and isinstance(payload.get("sft_summary"), dict):
                    summary = payload["sft_summary"]
                    if isinstance(summary.get("budget"), dict):
                        summary["budget"]["fraction"] = fraction
                    args = (json.dumps(payload, sort_keys=True),)
            real_print(*args, **kwargs)

        sft_train.evaluate_behavior = primary_behavior
        sft_train.print = primary_print
        print(
            "[kaggle-sft-ddp] execution: 2x Tesla T4, variable SFT blocks, "
            f"microbatch={microbatch_size}, exact global-token objective, "
            f"sft_fraction={fraction:.2%}",
            flush=True,
        )
    else:
        summary_path = (checkpoint_dir / "sft-summary.json").resolve()

        def rank1_write_text(path: Path, data: str, *args: Any, **kwargs: Any) -> int:
            if path.resolve() == summary_path:
                return len(data)
            return original_path_write_text(path, data, *args, **kwargs)

        Path.write_text = rank1_write_text
        sft_train._wandb_run = lambda *args, **kwargs: None
        sft_train._validation = lambda *args, **kwargs: _dummy_validation()
        sft_train.evaluate_behavior = lambda *args, **kwargs: _dummy_behavior()
        trainer_session.TrainingSession.save_checkpoint = lambda *args, **kwargs: None
        _disable_secondary_remote_side_effects(sft_train)
        sft_train.print = lambda *args, **kwargs: None

    exit_code = 1
    try:
        exit_code = int(sft_train.main(trainer_argv))
        if rank == 0 and exit_code == 0:
            _rewrite_summary_fraction(checkpoint_dir, fraction)
        _control_barrier(dist, control_group)
        return exit_code
    finally:
        if rank != 0:
            Path.write_text = original_path_write_text
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                if exit_code == 0:
                    raise


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "_control_barrier",
    "_disable_secondary_remote_side_effects",
    "_rank_row_indices",
    "main",
]
