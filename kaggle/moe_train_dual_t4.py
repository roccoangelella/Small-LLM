#!/usr/bin/env python3
"""Two-rank, exact-global-block Kaggle adapter for accepted production MoE.

The scientific model, optimizer, schedule, checkpoint, and data cursor remain in
MOE_model/trainer. Only the execution topology changes: split each 64-sequence
block 32/32, reduce the gradients, and merge the Quantile Balancing frontiers
*before* committing one identical router state on both ranks.
"""
from __future__ import annotations

from contextlib import nullcontext
from datetime import timedelta
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

WORLD_SIZE = 2
GLOBAL_SEQUENCES = 64


class FollowerCache:
    """Rank 1 reads rank 0's verified shard window; only rank 0 downloads/evicts."""

    def __init__(self, root: Path, *, timeout: float = 3 * 60 * 60) -> None:
        from dataset.incremental_frontier import _frontier_shards

        self.root = root
        contract = json.loads((root / "run_contract.json").read_text())
        frontier = json.loads((root / "shard_frontier.json").read_text())
        if (frontier.get("contract_sha256") != contract.get("contract_sha256")
                or frontier.get("run_id") != contract.get("run_id")
                or frontier.get("producer_complete") is not True):
            raise RuntimeError("rank-1 corpus frontier is not a complete, matching run")
        self.planned_block_count = int(contract["planned_train_blocks"])
        self.shards = _frontier_shards(frontier, "ready_train_shards")
        if not self.shards or self.shards[-1].last_block_id < self.planned_block_count - 1:
            raise RuntimeError("rank-1 corpus frontier does not cover the run")
        self.timeout = timeout
        self._verified: set[str] = set()

    def shard_for_block(self, block_id: int) -> Any:
        from dataset.incremental_frontier import _train_index_for_block

        if block_id < 0 or block_id >= self.planned_block_count:
            raise RuntimeError("rank-1 block is outside the frozen corpus")
        index = _train_index_for_block(self.shards, block_id)
        if index is None:
            raise RuntimeError("rank-1 block is not in the verified frontier")
        return self.shards[index]

    def ensure_block(self, block_id: int) -> None:
        from dataset.incremental_frontier import _file_matches

        shard = self.shard_for_block(block_id)
        if shard.filename in self._verified:
            return
        deadline = time.monotonic() + self.timeout
        while not _file_matches(self.root, shard):
            if time.monotonic() >= deadline:
                raise TimeoutError(f"rank 0 did not provide verified shard {shard.filename}")
            time.sleep(1)
        self._verified.add(shard.filename)

    def acknowledge(self, block_id: int) -> None:
        self.shard_for_block(block_id)  # cursor validation; only rank 0 evicts

    def restore_after_acknowledged(self, block_id: int) -> None:
        if block_id + 1 < self.planned_block_count:
            self.shard_for_block(block_id + 1)


def _loss_module(raw: Any) -> Any:
    from torch import nn
    from model.fused_loss import chunked_cross_entropy_sum
    from trainer.types import IGNORE_INDEX

    class Loss(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.raw = raw

        def forward(self, inputs: Any, labels: Any) -> tuple[Any, Any, Any]:
            hidden, aux = self.raw.hidden_states_with_aux(inputs)
            if hidden.shape[:2] != labels.shape:
                raise RuntimeError("DDP MoE hidden states and labels disagree")
            ce_sum = chunked_cross_entropy_sum(
                hidden, self.raw.token_embedding.weight, labels,
                semantic_vocab_size=self.raw.config.semantic_vocab_size,
                ignore_index=IGNORE_INDEX,
            )
            return ce_sum, aux.z_loss, aux.layers

    return Loss()


def merge_router_frontiers(routers: list[Any], telemetry: list[dict[str, object]]) -> None:
    """All ranks commit identical global counts and exact global score quantiles."""
    import torch
    import torch.distributed as dist

    for router, item in zip(routers, telemetry, strict=True):
        counts = item["counts"]
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        for key in ("selected_probability_sum", "entropy_sum"):
            dist.all_reduce(item[key], op=dist.ReduceOp.SUM)
        item["token_count"] = int(item["token_count"]) * WORLD_SIZE
        if router.balancing == "quantile":
            frontier = router._score_frontier
            target = router._quantile_target
            if frontier is None or target is None or frontier.shape[1] < target:
                raise RuntimeError("rank has no complete Quantile Balancing frontier")
            gathered = [torch.empty_like(frontier) for _ in range(WORLD_SIZE)]
            dist.all_gather(gathered, frontier.contiguous())
            router._score_frontier = torch.cat(gathered, dim=1).topk(target, dim=1).values
        router.commit_load(counts)


def any_rank_drain(local_drain: bool, device: Any) -> bool:
    """Stop both ranks if either wall clock expires after a global update."""
    import torch
    import torch.distributed as dist

    flag = torch.tensor(int(local_drain), dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


def distributed_step(engine: Any, batch: Any) -> Any:
    import torch
    import torch.distributed as dist
    from trainer.precision import autocast_context
    from trainer.step import (_clear_optimizer_step_statistics, _optimizer_gradient_norms,
                              _optimizer_step_statistics, _ordered_batch_tensors)
    from MOE_model.metrics import MoEStepMetrics
    from MOE_model.step import _accumulate_telemetry, _empty_telemetry, _finalize_telemetry

    raw = engine._small_llm_raw_model
    if batch.split != "train" or batch.sequence_count != GLOBAL_SEQUENCES:
        raise RuntimeError("MoE DDP requires the frozen 64-sequence optimizer block")
    if engine.config.precision != "bf16" or engine.scaler.is_enabled():
        raise RuntimeError("MoE dual T4 continuation requires unscaled BF16")
    rank = engine._small_llm_rank
    # Detect cursor/geometry divergence before any DDP collectives in backward.
    ids = torch.tensor([batch.block_id, batch.sequence_count, batch.target_token_count],
                       dtype=torch.int64, device=engine.device)
    low, high = ids.clone(), ids.clone()
    dist.all_reduce(low, op=dist.ReduceOp.MIN)
    dist.all_reduce(high, op=dist.ReduceOp.MAX)
    if not torch.equal(low, high):
        raise RuntimeError("MoE DDP ranks read different optimizer blocks")
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(engine.device)
    size = engine.config.microbatch_size
    if size not in {1, 2} or (GLOBAL_SEQUENCES // WORLD_SIZE) % size:
        raise RuntimeError("MoE T4 microbatch must divide 32 and be qualified as 1 or 2")
    inputs, labels = _ordered_batch_tensors(batch)
    total_positions = int(inputs.numel())
    routers = [block.ffn.router for block in raw.blocks]
    for router in routers:
        router.begin_step(total_positions)
    engine.optimizer.zero_grad(set_to_none=True)
    next_tokens = engine.consumed_tokens + batch.target_token_count
    lr = engine.scheduler.prepare_step(next_tokens)
    local_ce = torch.zeros((), dtype=torch.float32, device=engine.device)
    local_z = torch.zeros((), dtype=torch.float32, device=engine.device)
    # Telemetry helpers inspect engine.model.config; DDP has no config attribute.
    wrapped = engine.model
    engine.model = raw
    try:
        telemetry = _empty_telemetry(engine)
    finally:
        engine.model = wrapped
    first = rank * (GLOBAL_SEQUENCES // WORLD_SIZE)
    last = first + GLOBAL_SEQUENCES // WORLD_SIZE
    for start in range(first, last, size):
        stop = start + size
        micro_inputs = inputs[start:stop].to(device=engine.device, non_blocking=True)
        micro_labels = labels[start:stop].to(device=engine.device, non_blocking=True)
        sync = wrapped.no_sync() if stop < last else nullcontext()
        with sync, autocast_context(engine.config.precision, engine.device):
            ce_sum, z_loss, layers = wrapped(micro_inputs, micro_labels)
            # DDP averages. Compensate to recover the serial 64-sequence gradient.
            fraction = float(micro_inputs.numel()) / total_positions
            objective = WORLD_SIZE * (
                ce_sum / batch.target_token_count
                + float(raw.config.router_z_loss_coefficient) * z_loss * fraction
            )
            objective.backward()
        local_ce += ce_sum.detach().float()
        local_z += z_loss.detach().float() * fraction
        _accumulate_telemetry(telemetry, layers)
    raw_grad_norm = torch.nn.utils.clip_grad_norm_(raw.parameters(), engine.config.max_grad_norm)
    finite = torch.tensor([int(torch.isfinite(local_ce)), int(torch.isfinite(local_z)),
                           int(torch.isfinite(raw_grad_norm))], device=engine.device)
    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    if not bool(torch.all(finite)):
        engine.optimizer.zero_grad(set_to_none=True)
        engine.scheduler.cancel_step()
        raise FloatingPointError("non-finite MoE DDP loss or gradients; no rank updated")
    grad_value = float(raw_grad_norm.detach())
    role_gradient_norms = _optimizer_gradient_norms(engine.optimizer)
    _clear_optimizer_step_statistics(engine.optimizer)
    engine.optimizer.step()
    update_statistics = _optimizer_step_statistics(engine.optimizer)
    engine.consumed_tokens = next_tokens
    engine.global_step += 1
    engine.scheduler.commit(engine.consumed_tokens)
    merge_router_frontiers(routers, telemetry)
    dist.all_reduce(local_ce, op=dist.ReduceOp.SUM)
    dist.all_reduce(local_z, op=dist.ReduceOp.SUM)
    wrapped = engine.model
    engine.model = raw
    try:
        moe = _finalize_telemetry(engine, telemetry) if rank == 0 else {}
    finally:
        engine.model = wrapped
    ce = float(local_ce / batch.target_token_count)
    z = float(local_z)
    elapsed = max(time.perf_counter() - started, 1e-12)
    return MoEStepMetrics(
        step=engine.global_step, block_id=batch.block_id, loss=ce,
        objective_loss=ce + raw.config.router_z_loss_coefficient * z,
        router_z_loss=z, router_z_loss_coefficient=raw.config.router_z_loss_coefficient,
        learning_rate=lr, gradient_norm=grad_value, sequences=batch.sequence_count,
        target_tokens=batch.target_token_count, consumed_tokens=engine.consumed_tokens,
        elapsed_seconds=elapsed, tokens_per_second=batch.target_token_count / elapsed,
        overflow_retries=0, peak_memory_bytes=torch.cuda.max_memory_allocated(engine.device),
        grad_scaler_scale=float(engine.scaler.get_scale()),
        gradient_clipped=grad_value > engine.config.max_grad_norm,
        overflow_events_total=engine.overflow_events,
        peak_reserved_memory_bytes=torch.cuda.max_memory_reserved(engine.device),
        optimizer_gradient_norms=role_gradient_norms,
        optimizer_update_statistics=update_statistics, moe=moe,
    )


def install_ddp(rank: int, local_rank: int) -> None:
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    import MOE_model.__main__ as entry
    import MOE_model.setup as setup_module
    from MOE_model.engine import MoETrainerEngine
    from trainer.engine import TrainerEngine
    from trainer.session import TrainingSession
    import trainer.state as state_module

    original_setup = setup_module.setup
    original_step = TrainingSession.step
    original_save = TrainingSession.save_checkpoint
    original_evaluate = TrainerEngine.evaluate
    original_state_dict = state_module.engine_state_dict

    def setup(args: Any) -> Any:
        if args.model_size != "accepted" or args.precision != "bf16" or args.compile_mode != "off":
            raise RuntimeError("MoE DDP only supports accepted BF16 model with compile=off")
        args.device = f"cuda:{local_rank}"
        if rank == 1:
            setup_module._rolling_cache = lambda a: FollowerCache(Path(a.dataset_dir))
        result = original_setup(args)  # restores raw model/optimizer/cursor first
        model_config, config, engine, session, coordinator = result
        if session.source.sequences_per_block != GLOBAL_SEQUENCES:
            raise RuntimeError("MoE DDP requires a 64-sequence dataset block")
        if rank == 1:
            args.experiment_dir = None
        raw = engine.model
        if raw.config.dense.dropout != 0:
            raise RuntimeError("MoE DDP RNG duplication requires a dropout-free model")
        # A one-GPU checkpoint may have one CUDA RNG entry. Duplicate rank 0's
        # saved CUDA state across GPUs; training has no stochastic forward ops.
        rng = torch.cuda.get_rng_state(0).to(device=engine.device)
        dist.broadcast(rng, src=0)
        torch.cuda.set_rng_state(rng.cpu(), device=engine.device)
        engine._small_llm_raw_model = raw
        engine._small_llm_distributed_drain = lambda local: any_rank_drain(local, engine.device)
        engine._small_llm_rank = rank
        engine.model = DistributedDataParallel(
            _loss_module(raw), device_ids=[local_rank], output_device=local_rank,
            broadcast_buffers=False, gradient_as_bucket_view=True,
        )
        return result

    def step(self: Any, timeout: float | None = None) -> Any:
        dist.barrier()  # rank 1 waits while rank 0 validates/publishes
        return original_step(self, timeout=timeout)

    def save(self: Any, *args: Any, **kwargs: Any) -> Any:
        return original_save(self, *args, **kwargs) if rank == 0 else None

    def evaluate(self: Any, *args: Any, **kwargs: Any) -> Any:
        if rank != 0:
            return {"loss": 0.0, "perplexity": 1.0, "target_tokens": 0, "blocks": 0}
        wrapped = self.model
        self.model = self._small_llm_raw_model
        try:
            return original_evaluate(self, *args, **kwargs)
        finally:
            self.model = wrapped

    def checkpoint_state(engine: Any, *, cpu: bool = True) -> dict[str, object]:
        result = original_state_dict(engine, cpu=cpu)
        if hasattr(engine, "_small_llm_raw_model"):
            local_rng = torch.cuda.get_rng_state(engine.device).clone()
            result["cuda_rng_states"] = [local_rng.clone(), local_rng.clone()]
        return result

    entry.setup = setup
    MoETrainerEngine.train_batch = distributed_step
    TrainerEngine.evaluate = evaluate
    TrainingSession.step = step
    TrainingSession.save_checkpoint = save
    state_module.engine_state_dict = checkpoint_state
    if rank != 0:
        import trainer.cli as cli
        cli.print = lambda *args, **kwargs: None


def main(argv: list[str] | None = None) -> int:
    os.environ["TRITON_CACHE_AUTOTUNING"] = "1"
    os.environ["FLA_CACHE_RESULTS"] = "1"
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    import torch
    import torch.distributed as dist
    source = ROOT / "kaggle" / "src" / "dual_t4_train.py"
    spec = importlib.util.spec_from_file_location("qualified_kaggle_t4_autotune", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("missing qualified Kaggle T4 runtime")
    shim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shim)
    shim._require_runtime(torch)
    try:
        rank, local_rank, world = (int(os.environ[key]) for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
    except (KeyError, ValueError) as error:
        raise RuntimeError("launch MoE DDP through torchrun --nproc-per-node=2") from error
    if world != WORLD_SIZE or torch.cuda.device_count() != WORLD_SIZE:
        raise RuntimeError("MoE DDP requires exactly two visible T4 GPUs")
    if any(torch.cuda.get_device_name(i) != "Tesla T4" for i in range(WORLD_SIZE)):
        raise RuntimeError("MoE DDP requires two Tesla T4 GPUs")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=2))
    shim._install_bounded_autotune()
    install_ddp(rank, local_rank)
    try:
        import MOE_model.__main__ as entry
        result = int(entry.main(argv))
        dist.barrier()
        return result
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
