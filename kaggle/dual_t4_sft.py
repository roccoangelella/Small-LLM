#!/usr/bin/env python3
"""Kaggle two-T4 DDP execution shim for supervised fine-tuning."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


def _arguments(argv: Sequence[str] | None) -> tuple[Path, int, int, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--sft-fraction-numerator", type=int, required=True)
    parser.add_argument("--sft-fraction-denominator", type=int, required=True)
    args, trainer_argv = parser.parse_known_args(argv)
    worktree = args.worktree.resolve()
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


if __name__ == "__main__":
    raise SystemExit("dual_t4_sft.py must be launched through the SFT runtime")


__all__ = ["_rank_row_indices"]
