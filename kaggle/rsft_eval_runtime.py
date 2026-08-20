#!/usr/bin/env python3
"""Kaggle runtime for the canonical 100M/2B R-SFT qualification."""
from __future__ import annotations

import json
import os
from pathlib import Path

import rsft_prepare
import rsft_runtime
import sft_runtime as base


def _resolve_s0_bundle(explicit: str | None) -> Path:
    return rsft_prepare.resolve_s0_bundle(explicit, worktree=base.REPO)


def _resolve_rsft_bundle(explicit: str | None, *, s0_bundle: Path) -> Path:
    if explicit:
        root = base._find_bundle(explicit)
        rsft_runtime._require_atomic_production_bundle(root)
        return root
    root = rsft_prepare.prepare_production_bundle(
        worktree=base.REPO,
        s0_bundle=str(s0_bundle),
    )
    rsft_runtime._require_atomic_production_bundle(root)
    return root


def _repo_ids(
    *,
    parent_repo_id: str | None,
    checkpoint_repo_id: str | None,
    parent_checkpoint_dir: str | None,
    rsft_checkpoint_dir: str | None,
) -> tuple[str | None, str | None]:
    parent_repo = (
        parent_repo_id
        or os.environ.get("SMALL_LLM_SFT_HF_REPO_ID")
        or os.environ.get("SMALL_LLM_HF_REPO_ID")
    )
    rsft_repo = (
        checkpoint_repo_id
        or os.environ.get("SMALL_LLM_RSFT_HF_REPO_ID")
        or os.environ.get("SMALL_LLM_SFT_HF_REPO_ID")
        or os.environ.get("SMALL_LLM_HF_REPO_ID")
    )
    if not parent_checkpoint_dir and not parent_repo:
        raise base.RuntimeFailure(
            "R-SFT evaluation requires the S0 checkpoint repository; pass --parent-repo-id or set "
            "SMALL_LLM_SFT_HF_REPO_ID/SMALL_LLM_HF_REPO_ID"
        )
    if not rsft_checkpoint_dir and not rsft_repo:
        raise base.RuntimeFailure(
            "R-SFT evaluation requires the R-SFT checkpoint repository; pass --checkpoint-repo-id or set "
            "SMALL_LLM_RSFT_HF_REPO_ID/SMALL_LLM_SFT_HF_REPO_ID/SMALL_LLM_HF_REPO_ID"
        )
    return parent_repo, rsft_repo


def evaluation_plan(
    profile: rsft_runtime.RSFTProfile,
    *,
    s0_bundle: str | None,
    dataset_dir: str | None,
    eval_dir: str | None,
    parent_repo_id: str | None,
    checkpoint_repo_id: str | None,
    parent_checkpoint_dir: str | None,
    rsft_checkpoint_dir: str | None,
    output: str | None,
    suite: str,
    device: str,
    precision: str,
    batch_size: int,
    validation_blocks: int,
    test_blocks: int,
    reasoning_samples: int,
    reasoning_max_new_tokens: int,
) -> dict[str, object]:
    parent_repo, rsft_repo = _repo_ids(
        parent_repo_id=parent_repo_id,
        checkpoint_repo_id=checkpoint_repo_id,
        parent_checkpoint_dir=parent_checkpoint_dir,
        rsft_checkpoint_dir=rsft_checkpoint_dir,
    )
    selected_output = (
        Path(output).expanduser().resolve()
        if output
        else profile.run_root / f"post-rsft-{suite}-qualification.json"
    )
    return {
        "schema": "small-llm-rsft-eval-plan-v1",
        "stage": "r_sft_r0",
        "contract": "atomic-production-v1",
        "comparison": f"{rsft_runtime.PARENT_RUN_ID}->{profile.sft_run_id}",
        "s0_bundle": str(Path(s0_bundle).expanduser().resolve()) if s0_bundle else "auto:attached-or-private-kaggle",
        "rsft_bundle": (
            str(Path(dataset_dir).expanduser().resolve())
            if dataset_dir
            else "auto:rebuild-verified-from-committed-12306-corpus"
        ),
        "eval_core_v1": eval_dir or "auto:attached-kaggle-eval_core_v1",
        "s0_repo_id": parent_repo,
        "rsft_repo_id": rsft_repo,
        "s0_checkpoint_dir": parent_checkpoint_dir,
        "rsft_checkpoint_dir": rsft_checkpoint_dir,
        "output": str(selected_output),
        "suite": suite,
        "device": device,
        "precision": precision,
        "batch_size": batch_size,
        "validation_blocks": validation_blocks,
        "test_blocks": test_blocks,
        "reasoning_samples": reasoning_samples,
        "reasoning_max_new_tokens": reasoning_max_new_tokens,
        "historical_greedy": {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "max_new_tokens": 32},
        "historical_wider": {"temperature": 1.0, "top_p": 0.9, "top_k": 20},
        "reasoning_sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 0},
    }


def evaluate(
    profile: rsft_runtime.RSFTProfile,
    *,
    s0_bundle: str | None = None,
    dataset_dir: str | None = None,
    eval_dir: str | None = None,
    parent_repo_id: str | None = None,
    checkpoint_repo_id: str | None = None,
    parent_checkpoint_dir: str | None = None,
    rsft_checkpoint_dir: str | None = None,
    output: str | None = None,
    suite: str = "full",
    device: str = "auto",
    precision: str = "auto",
    batch_size: int = 1,
    validation_blocks: int = 32,
    test_blocks: int = 32,
    reasoning_samples: int = 8,
    reasoning_max_new_tokens: int = 256,
) -> int:
    if not eval_dir:
        raise base.RuntimeFailure(
            "eval_core_v1 was not found; attach the same eval_core_v1 Kaggle dataset used for S0 or pass --eval-dir"
        )
    selected_eval = Path(eval_dir).expanduser().resolve()
    if not selected_eval.is_dir():
        raise base.RuntimeFailure(f"eval_core_v1 directory does not exist: {selected_eval}")

    parent_repo, rsft_repo = _repo_ids(
        parent_repo_id=parent_repo_id,
        checkpoint_repo_id=checkpoint_repo_id,
        parent_checkpoint_dir=parent_checkpoint_dir,
        rsft_checkpoint_dir=rsft_checkpoint_dir,
    )
    selected_s0_bundle = _resolve_s0_bundle(s0_bundle)
    selected_rsft_bundle = _resolve_rsft_bundle(dataset_dir, s0_bundle=selected_s0_bundle)
    selected_output = (
        Path(output).expanduser().resolve()
        if output
        else profile.run_root / f"post-rsft-{suite}-qualification.json"
    )

    cmd = [
        *base._uv_prefix(),
        "python",
        "-m",
        "post_training.rsft_eval_suite",
        "--s0-bundle",
        str(selected_s0_bundle),
        "--rsft-bundle",
        str(selected_rsft_bundle),
        "--eval-dir",
        str(selected_eval),
        "--suite",
        suite,
        "--device",
        device,
        "--precision",
        precision,
        "--batch-size",
        str(batch_size),
        "--validation-blocks",
        str(validation_blocks),
        "--test-blocks",
        str(test_blocks),
        "--bootstrap-samples",
        "200",
        "--reasoning-samples",
        str(reasoning_samples),
        "--reasoning-max-new-tokens",
        str(reasoning_max_new_tokens),
        "--output",
        str(selected_output),
        "--s0-run-id",
        rsft_runtime.PARENT_RUN_ID,
        "--s0-pointer",
        "latest",
        "--rsft-run-id",
        profile.sft_run_id,
        "--rsft-pointer",
        "latest",
    ]
    if parent_checkpoint_dir:
        cmd += ["--s0-checkpoint-dir", str(Path(parent_checkpoint_dir).expanduser().resolve())]
    else:
        assert parent_repo is not None
        cmd += ["--s0-repo-id", parent_repo]
    if rsft_checkpoint_dir:
        cmd += ["--rsft-checkpoint-dir", str(Path(rsft_checkpoint_dir).expanduser().resolve())]
    else:
        assert rsft_repo is not None
        cmd += ["--rsft-repo-id", rsft_repo]

    print(
        "[rsft-eval] "
        + json.dumps(
            {
                "comparison": f"{rsft_runtime.PARENT_RUN_ID}->{profile.sft_run_id}",
                "s0_bundle": str(selected_s0_bundle),
                "rsft_bundle": str(selected_rsft_bundle),
                "eval_core_v1": str(selected_eval),
                "output": str(selected_output),
                "reasoning_samples": reasoning_samples,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return base._run(cmd, cwd=base.REPO)


__all__ = ["evaluate", "evaluation_plan"]
