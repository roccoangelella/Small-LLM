#!/usr/bin/env python3
"""Run the qualified Kaggle two-T4 DDP shim with a 64-sequence global block.

The shared DDP implementation remains the authority for synchronization,
overflow handling, raw-model checkpointing, and rank-zero side effects. This
entrypoint changes only the frozen global optimizer-block geometry required by
the 100M/10B trajectory.
"""
from __future__ import annotations

import builtins
from typing import Any, Sequence

import dual_t4_train as base

SEQUENCES_PER_BLOCK = 64


def _install_geometry_banner() -> None:
    """Keep the shared shim's historical banner accurate for block64 execution."""

    def geometry_print(*args: Any, **kwargs: Any) -> None:
        if args and args[0] == (
            "[kaggle-ddp] standard execution: 2x Tesla T4, global block=16, "
            "8 sequences/rank, microbatch=4, exact-batch DDP"
        ):
            args = (
                "[kaggle-ddp] standard execution: 2x Tesla T4, global block=64, "
                "32 sequences/rank, microbatch=4, exact-batch DDP",
                *args[1:],
            )
        builtins.print(*args, **kwargs)

    base.print = geometry_print


def main(argv: Sequence[str] | None = None) -> int:
    if SEQUENCES_PER_BLOCK % base.WORLD_SIZE != 0:
        raise RuntimeError("block64 cannot be split evenly across the two T4 ranks")
    base.SEQUENCES_PER_BLOCK = SEQUENCES_PER_BLOCK
    _install_geometry_banner()
    return base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
