#!/usr/bin/env python3
"""Run the accepted 100M/2B 10% SFT with fresh aggressive LR decay.

This is deliberately a narrow wrapper around the already-qualified dual-T4 SFT
execution path. It changes only the S0 trainer schedule constructor for the 10%
run; DDP slicing, optimizer, checkpointing, evaluation, and resume behavior stay
owned by ``dual_t4_sft.py``.
"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Sequence


def _argument_value(argv: Sequence[str], flag: str) -> str:
    values = list(argv)
    try:
        index = values.index(flag)
    except ValueError as error:
        raise RuntimeError(f"required 10% SFT wrapper argument is missing: {flag}") from error
    if index + 1 >= len(values):
        raise RuntimeError(f"10% SFT wrapper argument has no value: {flag}")
    return str(values[index + 1])


def main(argv: Sequence[str] | None = None) -> int:
    supplied = list(sys.argv[1:] if argv is None else argv)
    numerator = int(_argument_value(supplied, "--sft-fraction-numerator"))
    denominator = int(_argument_value(supplied, "--sft-fraction-denominator"))
    if numerator * 10 != denominator:
        raise RuntimeError(
            "the aggressive SFT wrapper is frozen to the exact 10% S0 experiment"
        )

    worktree = Path(_argument_value(supplied, "--worktree")).resolve()
    if not (worktree / "post_training" / "sft").is_dir():
        raise RuntimeError(f"10% SFT worktree is invalid: {worktree}")
    sys.path.insert(0, str(worktree))

    import post_training.sft.train_cli as sft_train
    from post_training.sft.config import (
        S0_AGGRESSIVE_PEAK_LR,
        build_s0_aggressive_trainer_config,
    )
    import dual_t4_sft

    original_builder = sft_train.build_s0_trainer_config

    def aggressive_builder(schedule, **kwargs):
        supplied_lr = float(kwargs.get("learning_rate", S0_AGGRESSIVE_PEAK_LR))
        if supplied_lr != S0_AGGRESSIVE_PEAK_LR:
            raise RuntimeError(
                "100M/2B 10% SFT peak LR is frozen at 3e-5 under ADR-0126"
            )
        return build_s0_aggressive_trainer_config(schedule, **kwargs)

    sft_train.build_s0_trainer_config = aggressive_builder
    try:
        return int(dual_t4_sft.main(supplied))
    finally:
        sft_train.build_s0_trainer_config = original_builder


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
