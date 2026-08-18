#!/usr/bin/env python3
"""Run the 100M/10B block64 path on Kaggle exact-batch two-T4 DDP.

The shared DDP implementation remains the authority for optimizer-step
synchronization, overflow handling, raw-model checkpointing, and rank-zero side
effects. This entrypoint changes only execution concerns required by the 100M
T4 path:

- preserve the 64-sequence global optimizer block as 32 ordered rows per rank;
- use execution microbatch two, the largest 100M/T4 shape already shown to
  complete real optimizer updates with material memory headroom;
- route long cadence/prewarm/final rendezvous through a one-hour CPU/Gloo
  control group so rank zero validation/checkpoint/publication cannot trip the
  default ten-minute NCCL watchdog observed in the 100M SFT path.
"""
from __future__ import annotations

from datetime import timedelta
import os
import sys
from typing import Any, Sequence

import dual_t4_train as base

SEQUENCES_PER_BLOCK = 64
MICROBATCH_SIZE = 2
CONTROL_GROUP_TIMEOUT_SECONDS = 60 * 60


def _install_geometry_overrides() -> None:
    """Adapt the shared exact-batch engine to block64 and microbatch two."""

    original_prewarm = base._prewarm_raw_model

    def prewarm(engine: Any, *, rank: int, microbatch_size: int = MICROBATCH_SIZE) -> None:
        original_prewarm(engine, rank=rank, microbatch_size=microbatch_size)

    base.SEQUENCES_PER_BLOCK = SEQUENCES_PER_BLOCK
    base.MICROBATCH_SIZE = MICROBATCH_SIZE
    base._prewarm_raw_model = prewarm


def _new_control_group(distributed: Any) -> Any:
    return distributed.new_group(
        backend="gloo",
        timeout=timedelta(seconds=CONTROL_GROUP_TIMEOUT_SECONDS),
    )


def _install_control_barrier(distributed: Any, group: Any) -> None:
    """Make only synchronization barriers use Gloo; DDP all-reduces stay NCCL."""

    original_barrier = distributed.barrier

    def control_barrier(*args: Any, **kwargs: Any) -> Any:
        # Preserve an explicitly requested group. The shared pretraining shim
        # does not pass one today, but fail gracefully if that changes later.
        if args or kwargs.get("group") is not None:
            return original_barrier(*args, **kwargs)
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
            "32 sequences/rank, microbatch=2, control_barrier=gloo-1h, exact-batch DDP",
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
