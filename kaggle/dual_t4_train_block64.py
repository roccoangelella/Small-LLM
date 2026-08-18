#!/usr/bin/env python3
"""Run the 100M/10B block64 path on Kaggle exact-batch two-T4 DDP.

The shared DDP implementation remains the authority for optimizer-step
synchronization, overflow handling, raw-model checkpointing, and rank-zero side
effects. This entrypoint changes only execution concerns required by the 100M
T4 path:

- preserve the 64-sequence global optimizer block as 32 ordered rows per rank;
- use execution microbatch two, the largest 100M/T4 shape already shown to
  complete real optimizer updates with material memory headroom;
- route long cadence/final rendezvous through a one-hour CPU/Gloo control group
  so rank zero validation/checkpoint/publication cannot trip the default
  ten-minute NCCL watchdog observed in the 100M SFT path;
- use a separate five-minute monitored Gloo startup rendezvous after prewarm so
  a stalled rank cannot silently consume an hour of Kaggle GPU time;
- preseed a manifest-verified portable Triton cache before importing Torch when
  a compatible cache dataset is attached, otherwise use normal JIT fallback;
- emit first-update entry/completion milestones so a qualification stall is not
  mistaken for another startup or cache problem.
"""
from __future__ import annotations

from datetime import timedelta
import os
import sys
from typing import Any, Sequence

import dual_t4_train as base
import triton_cache

SEQUENCES_PER_BLOCK = 64
MICROBATCH_SIZE = 2
CONTROL_GROUP_TIMEOUT_SECONDS = 60 * 60
STARTUP_BARRIER_TIMEOUT_SECONDS = 5 * 60


def _install_geometry_overrides() -> None:
    """Adapt the shared exact-batch engine to block64 and microbatch two."""

    original_prewarm = base._prewarm_raw_model
    original_train_step = base._distributed_train_step

    def prewarm(engine: Any, *, rank: int, microbatch_size: int = MICROBATCH_SIZE) -> None:
        original_prewarm(engine, rank=rank, microbatch_size=microbatch_size)

    def diagnostic_train_step(engine: Any, batch: Any) -> Any:
        rank = int(getattr(engine, "_small_llm_rank", -1))
        first_step = int(getattr(engine, "global_step", -1)) == 18_000
        if first_step:
            print(
                f"[kaggle-ddp][rank {rank}] first distributed optimizer step entered "
                f"at block={getattr(batch, 'block_id', 'unknown')}; "
                "32 sequences/rank, 16 local microbatches",
                flush=True,
            )
        result = original_train_step(engine, batch)
        if first_step:
            print(
                f"[kaggle-ddp][rank {rank}] first distributed optimizer step completed "
                f"in {float(getattr(result, 'elapsed_seconds', 0.0)):.2f}s",
                flush=True,
            )
        return result

    base.SEQUENCES_PER_BLOCK = SEQUENCES_PER_BLOCK
    base.MICROBATCH_SIZE = MICROBATCH_SIZE
    base._prewarm_raw_model = prewarm
    base._distributed_train_step = diagnostic_train_step


def _new_control_group(distributed: Any) -> Any:
    return distributed.new_group(
        backend="gloo",
        timeout=timedelta(seconds=CONTROL_GROUP_TIMEOUT_SECONDS),
    )


def _install_control_barrier(distributed: Any, group: Any) -> None:
    """Use monitored Gloo for startup and long-lived Gloo for later barriers."""

    original_barrier = distributed.barrier
    startup_pending = True

    def control_barrier(*args: Any, **kwargs: Any) -> Any:
        nonlocal startup_pending
        # Preserve an explicitly requested group. The shared pretraining shim
        # does not pass one today, but fail gracefully if that changes later.
        if args or kwargs.get("group") is not None:
            return original_barrier(*args, **kwargs)
        if startup_pending:
            startup_pending = False
            rank = os.environ.get("RANK", "?")
            print(
                f"[kaggle-ddp][rank {rank}] reached post-prewarm startup barrier; "
                f"timeout={STARTUP_BARRIER_TIMEOUT_SECONDS}s",
                flush=True,
            )
            monitored = getattr(distributed, "monitored_barrier", None)
            if not callable(monitored):
                raise RuntimeError("qualified PyTorch lacks dist.monitored_barrier")
            result = monitored(
                group=group,
                timeout=timedelta(seconds=STARTUP_BARRIER_TIMEOUT_SECONDS),
                wait_all_ranks=True,
            )
            print(
                f"[kaggle-ddp][rank {rank}] post-prewarm startup barrier passed",
                flush=True,
            )
            return result
        return original_barrier(group=group)

    distributed.barrier = control_barrier


def main(argv: Sequence[str] | None = None) -> int:
    if SEQUENCES_PER_BLOCK % base.WORLD_SIZE != 0:
        raise RuntimeError("block64 cannot be split evenly across the two T4 ranks")
    if (SEQUENCES_PER_BLOCK // base.WORLD_SIZE) % MICROBATCH_SIZE != 0:
        raise RuntimeError("per-rank block cannot be divided into exact microbatches")
    _install_geometry_overrides()

    worktree, trainer_argv = base._arguments(argv)
    sys.path.insert(0, str(worktree))

    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ["TRITON_CACHE_AUTOTUNING"] = "1"
    os.environ["FLA_CACHE_RESULTS"] = "1"

    # torchrun starts two Python processes. The cache helper serializes seed
    # installation with a filesystem lock, so only one rank extracts the
    # dataset while the other observes the atomically installed cache.
    triton_cache.prepare_environment()

    import torch
    import torch.distributed as dist

    base._require_runtime(torch)
    try:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    except (KeyError, ValueError) as error:
        raise RuntimeError("Kaggle dual-T4 training must be launched through torchrun") from error
    if world_size != base.WORLD_SIZE:
        raise RuntimeError(f"Kaggle dual-T4 training requires world_size={base.WORLD_SIZE}")
    if torch.cuda.device_count() < base.WORLD_SIZE:
        raise RuntimeError("Kaggle dual-T4 training requires two visible CUDA devices")
    names = [torch.cuda.get_device_name(index) for index in range(base.WORLD_SIZE)]
    if any(name != "Tesla T4" for name in names):
        raise RuntimeError(f"Kaggle dual-T4 training requires two Tesla T4 GPUs; found {names}")

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    control_group = _new_control_group(dist)
    _install_control_barrier(dist, control_group)
    base._install_bounded_autotune()
    if rank == 0:
        print(
            "[kaggle-ddp] 100M/10B execution: 2x Tesla T4, global block=64, "
            "32 sequences/rank, microbatch=2, control_barrier=gloo-1h, "
            "startup_barrier=gloo-monitored-5m, exact-batch DDP",
            flush=True,
        )
    trainer_cli = base._install_pinned_trainer_ddp(
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
