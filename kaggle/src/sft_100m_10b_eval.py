"""Kaggle-side post-SFT qualification for the 100M/10B same-data experiment."""
from __future__ import annotations

import os
from pathlib import Path
import sys

import sft_runtime as base
import sft_scaled_runtime


def evaluate(
    profile: base.SFTProfileSpec,
    *,
    dataset_dir: str | None = None,
    eval_dir: str | None = None,
    parent_repo_id: str | None = None,
    checkpoint_repo_id: str | None = None,
    parent_checkpoint_dir: str | None = None,
    sft_checkpoint_dir: str | None = None,
    output: str | None = None,
    suite: str = "full",
    device: str = "auto",
    precision: str = "auto",
    batch_size: int = 1,
    validation_blocks: int = 32,
    test_blocks: int = 32,
) -> int:
    bundle = base._find_bundle(dataset_dir, profile)
    sft_scaled_runtime._verify_published_10pct_training_bundle(bundle)

    parent_repo = (
        parent_repo_id
        or os.environ.get("SMALL_LLM_PARENT_HF_REPO_ID")
        or os.environ.get("SMALL_LLM_HF_REPO_ID")
    )
    checkpoint_repo = checkpoint_repo_id or os.environ.get("SMALL_LLM_SFT_HF_REPO_ID", parent_repo)
    if not parent_checkpoint_dir and not parent_repo:
        raise base.RuntimeFailure("evaluation requires parent checkpoint repository ID or local directory")
    if not sft_checkpoint_dir and not checkpoint_repo:
        raise base.RuntimeFailure("evaluation requires SFT checkpoint repository ID or local directory")

    if eval_dir:
        selected_eval_dir = Path(eval_dir).expanduser().resolve()
    else:
        test_eval = base.REPO / "tests" / "test_datasets" / "eval_core_v1"
        selected_eval_dir = test_eval.resolve() if (test_eval / "manifest.json").is_file() else (base.WORK / "eval_core_v1")

    selected_output = (
        Path(output).expanduser().resolve()
        if output
        else profile.run_root / f"post-sft-{suite}-qualification.json"
    )
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "sft_100m_10b_eval_runner.py"),
        "--dataset-dir", str(bundle),
        "--eval-dir", str(selected_eval_dir),
        "--suite", suite,
        "--device", device,
        "--precision", precision,
        "--batch-size", str(batch_size),
        "--validation-blocks", str(validation_blocks),
        "--test-blocks", str(test_blocks),
        "--output", str(selected_output),
    ]
    if parent_checkpoint_dir:
        cmd += ["--parent-checkpoint-dir", str(Path(parent_checkpoint_dir).expanduser().resolve())]
    else:
        cmd += [
            "--parent-repo-id", str(parent_repo),
            "--parent-run-id", profile.parent_run_id,
            "--parent-pointer", str(getattr(profile, "parent_pointer", "latest")),
        ]
    if sft_checkpoint_dir:
        cmd += ["--sft-checkpoint-dir", str(Path(sft_checkpoint_dir).expanduser().resolve())]
    else:
        cmd += [
            "--sft-repo-id", str(checkpoint_repo),
            "--sft-run-id", profile.sft_run_id,
            "--sft-pointer", "latest",
        ]
    return base._run(base._uv_prefix() + cmd, cwd=base.REPO)


__all__ = ["evaluate"]
