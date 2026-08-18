#!/usr/bin/env python3
"""Run the qualified Kaggle two-T4 DDP shim with a 64-sequence global block.

The shared DDP implementation remains the authority for synchronization,
overflow handling, raw-model checkpointing, and rank-zero side effects.  This
entrypoint changes only the frozen global optimizer-block geometry required by
the 100M/10B trajectory.
"""
from __future__ import annotations

from typing import Sequence

import dual_t4_train as base

SEQUENCES_PER_BLOCK = 64


def main(argv: Sequence[str] | None = None) -> int:
    if SEQUENCES_PER_BLOCK % base.WORLD_SIZE != 0:
        raise RuntimeError("block64 cannot be split evenly across the two T4 ranks")
    base.SEQUENCES_PER_BLOCK = SEQUENCES_PER_BLOCK
    return base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
