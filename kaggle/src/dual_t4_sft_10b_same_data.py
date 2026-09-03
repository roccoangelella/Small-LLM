#!/usr/bin/env python3
"""Run the 100M/10B parent on the exact 100M/2B 10% S0 corpus.

This wrapper preserves the already accepted 10% S0 training schedule and dataset
identity, while replacing only the parent checkpoint transport. The experiment is
therefore an apples-to-apples SFT comparison: same 100M architecture, same SFT
loss-bearing tokens, same SFT data, same peak-through-3000 schedule, different
pretraining horizon.
"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Sequence

EXPECTED_PARENT_RUN_ID = "100m-10b-deep-decay-from-step15500"
EXPECTED_PARENT_POINTER = "latest"
EXPECTED_SFT_TARGETS = 200_100_044
EXPECTED_PARENT_TARGETS = 10_000_007_168


def _argument_value(argv: Sequence[str], flag: str) -> str:
    values = list(argv)
    try:
        index = values.index(flag)
    except ValueError as error:
        raise RuntimeError(f"required 10B same-data SFT argument is missing: {flag}") from error
    if index + 1 >= len(values):
        raise RuntimeError(f"10B same-data SFT argument has no value: {flag}")
    return str(values[index + 1])


def _expect_argument(argv: Sequence[str], flag: str, expected: str) -> None:
    actual = _argument_value(argv, flag)
    if actual != expected:
        raise RuntimeError(
            f"10B same-data SFT is frozen to {flag} {expected}; received {actual}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    supplied = list(sys.argv[1:] if argv is None else argv)
    _expect_argument(supplied, "--parent-run-id", EXPECTED_PARENT_RUN_ID)
    _expect_argument(supplied, "--parent-pointer", EXPECTED_PARENT_POINTER)
    _expect_argument(supplied, "--sft-fraction-numerator", str(EXPECTED_SFT_TARGETS))
    _expect_argument(supplied, "--sft-fraction-denominator", str(EXPECTED_PARENT_TARGETS))

    worktree = Path(_argument_value(supplied, "--worktree")).resolve()
    if not (worktree / "post_training" / "sft").is_dir():
        raise RuntimeError(f"10B same-data SFT worktree is invalid: {worktree}")
    sys.path.insert(0, str(worktree))

    import dual_t4_sft
    import post_training.sft.train_cli as sft_train
    from post_training.sft.config import S0_AGGRESSIVE_PEAK_LR
    from post_training.sft.s0_peak3000_schedule import (
        build_s0_peak3000_trainer_config,
    )
    import trainer.model_artifact as model_artifact
    from post_training.sft.checkpoints import download_parent_checkpoint

    original_builder = sft_train.build_s0_trainer_config
    original_model_artifact = model_artifact.download_verified_model_artifact

    def aggressive_builder(schedule, **kwargs):
        supplied_lr = float(kwargs.get("learning_rate", S0_AGGRESSIVE_PEAK_LR))
        if supplied_lr != S0_AGGRESSIVE_PEAK_LR:
            raise RuntimeError(
                "100M/10B same-data SFT peak LR is frozen at 3e-5 to match the 100M/2B 10% run"
            )
        return build_s0_peak3000_trainer_config(schedule, **kwargs)

    def bucket_latest_parent_artifact(*, repo_id: str, run_id: str, token: str | None = None,
                                      revision: str | None = None, destination: Path | str | None = None):
        if run_id != EXPECTED_PARENT_RUN_ID:
            raise RuntimeError(
                f"10B same-data SFT parent run mismatch: expected {EXPECTED_PARENT_RUN_ID}, got {run_id}"
            )
        if destination is None:
            raise RuntimeError("10B same-data SFT parent download requires an explicit destination")
        root, metadata = download_parent_checkpoint(
            repo_id=repo_id,
            run_id=run_id,
            pointer=EXPECTED_PARENT_POINTER,
            transport="hf_storage_bucket",
            token=token,
            revision=revision,
            destination=Path(destination),
        )
        return root, dict(metadata)

    sft_train.build_s0_trainer_config = aggressive_builder
    model_artifact.download_verified_model_artifact = bucket_latest_parent_artifact
    try:
        return int(dual_t4_sft.main(supplied))
    finally:
        sft_train.build_s0_trainer_config = original_builder
        model_artifact.download_verified_model_artifact = original_model_artifact


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXPECTED_PARENT_POINTER",
    "EXPECTED_PARENT_RUN_ID",
    "EXPECTED_PARENT_TARGETS",
    "EXPECTED_SFT_TARGETS",
    "main",
]
