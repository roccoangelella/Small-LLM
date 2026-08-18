#!/usr/bin/env python3
"""Run the Kaggle two-T4 DDP shim for the 100M/10B block64 path.

The shared DDP implementation remains the authority for synchronization,
overflow handling, raw-model checkpointing, and rank-zero side effects. This
entrypoint changes execution slicing only: a 64-sequence global optimizer block
is split 32/32 across ranks and each rank uses microbatch two, the largest
100M/T4 shape already shown to complete real optimizer updates with headroom.
"""
from __future__ import annotations

import builtins
from typing import Any, Sequence

import dual_t4_train as base

SEQUENCES_PER_BLOCK = 64
MICROBATCH_SIZE = 2


def _install_geometry_overrides() -> None:
    """Adapt only block geometry, prewarm shape, and the historical banner."""

    original_prewarm = base._prewarm_raw_model

    def prewarm(engine: Any, *, rank: int, microbatch_size: int = MICROBATCH_SIZE) -> None:
        original_prewarm(engine, rank=rank, microbatch_size=microbatch_size)

    def geometry_print(*args: Any, **kwargs: Any) -> None:
        if args and args[0] == (
            "[kaggle-ddp] standard execution: 2x Tesla T4, global block=16, "
            "8 sequences/rank, microbatch=4, exact-batch DDP"
        ):
            args = (
                "[kaggle-ddp] standard execution: 2x Tesla T4, global block=64, "
                "32 sequences/rank, microbatch=2, exact-batch DDP",
                *args[1:],
            )
        builtins.print(*args, **kwargs)

    base.SEQUENCES_PER_BLOCK = SEQUENCES_PER_BLOCK
    base.MICROBATCH_SIZE = MICROBATCH_SIZE
    base._prewarm_raw_model = prewarm
    base.print = geometry_print


def main(argv: Sequence[str] | None = None) -> int:
    if SEQUENCES_PER_BLOCK % base.WORLD_SIZE != 0:
        raise RuntimeError("block64 cannot be split evenly across the two T4 ranks")
    if (SEQUENCES_PER_BLOCK // base.WORLD_SIZE) % MICROBATCH_SIZE != 0:
        raise RuntimeError("per-rank block cannot be divided into exact microbatches")
    _install_geometry_overrides()
    return base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
