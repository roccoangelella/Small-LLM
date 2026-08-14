#!/usr/bin/env python3
"""Scaled SFT runtime extensions."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
from typing import Any

import dual_t4_runtime
import sft_runtime as base


# Keep inline qualification deliberately tiny while both DDP workers are alive.
# Full post-SFT qualification runs separately after training, when the duplicate
# dual-T4 host footprint is gone.
INLINE_VALIDATION_BLOCKS = 1
INLINE_BEHAVIOR_CASES = 2

# Kaggle's two Python/DDP workers share one host-memory budget.  Bound glibc
# arena growth and omit qualification-only optimizer tensor cloning in this
# execution path; neither setting changes optimizer state or model updates.
KAGGLE_SFT_PROCESS_ENV = (
    "MALLOC_ARENA_MAX=2",
    "MALLOC_TRIM_THRESHOLD_=131072",
    "SMALL_LLM_DISABLE_OPTIMIZER_TELEMETRY=1",
)


def _require_stable_parent_artifact(
    *,
    repo_id: str,
    run_id: str,
    token: str | None,
    api: Any | None = None,
) -> None:
    """Fail on a wrong parent repository before W&B or GPU setup begins."""
    if not token:
        raise base.RuntimeFailure("HF_TOKEN is required for the private SFT parent artifact")
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="model")
    except Exception as error:  # noqa: BLE001 - normalize the remote boundary for operators
        raise base.RuntimeFailure(
            f"cannot inspect Hugging Face parent repository {repo_id!r}"
        ) from error

    root = f"models/{run_id}/"
    pointer = root + "artifact.json"
    checkpoint = re.compile(rf"^{re.escape(root)}step-\d{{8}}/.+")
    if pointer not in files and not any(
        isinstance(path, str) and checkpoint.fullmatch(path) is not None for path in files
    ):
        raise base.RuntimeFailure(
            f"Hugging Face parent repository {repo_id!r} contains no stable artifact for "
            f"run {run_id!r}; expected {pointer} or {root}step-XXXXXXXX/. "
            "Set SMALL_LLM_HF_REPO_ID to the repository for this parent model, or pass "
            "--parent-repo-id explicitly."
        )
    print(
        f"[sft-parent-preflight] repo={repo_id} run={run_id} status=available",
        flush=True,
    )


def prepare(
    profile: base.SFTProfileSpec,
    *,
    replay_root: str,
    prepared_dir: str | None,
    output_dir: str | None,
    parent_consumed_tokens: int | None,
    revision: str | None,
) -> int:
    replay = base._resolve_replay_root(replay_root)
    worktree = base._prepare_worktree(profile)
    prepared = Path(prepared_dir).expanduser().resolve() if prepared_dir else profile.default_prepared
    output = Path(output_dir).expanduser().resolve() if output_dir else profile.default_bundle
    prepared_manifest_path = prepared / "prepared-manifest.json"
    revision_args = ["--revision", revision] if revision else []
    if prepared_manifest_path.is_file():
        if revision is not None:
            try:
                payload = json.loads(prepared_manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError) as error:
                raise base.RuntimeFailure("existing prepared SFT source manifest is invalid") from error
            if not isinstance(payload, dict) or payload.get("revision") != revision:
                raise base.RuntimeFailure("existing prepared SFT source uses a different pinned revision")
    else:
        base._run(
            base._uv_prefix(datasets=True)
            + ["python", "-m", "post_training.sft.bundle", "prepare", "--output-dir", str(prepared), *revision_args],
            cwd=worktree,
        )

    exact_parent_tokens = base._exact_parent_tokens(profile, parent_consumed_tokens)
    expected_targets = base._expected_sft_targets(profile, exact_parent_tokens)
    if not base._verify_existing_bundle_budget(output, expected_targets=expected_targets):
        if output.exists():
            raise base.RuntimeFailure(
                f"refusing to replace incomplete/non-bundle SFT output directory: {output}"
            )
        base._run(
            base._uv_prefix()
            + [
                "python", "-m", "post_training.sft.scaled_bundle",
                "--prepared-dir", str(prepared),
                "--replay-root", str(replay),
                "--output-dir", str(output),
                "--parent-consumed-tokens", str(exact_parent_tokens),
                "--fraction-numerator", str(profile.sft_fraction_numerator),
                "--fraction-denominator", str(profile.sft_fraction_denominator),
                "--optimizer-target-tokens", "32768",
                "--instruction-share", "0.85",
                "--replay-share", "0.15",
                "--seed", "17",
            ],
            cwd=worktree,
        )
    return base._run(
        base._uv_prefix() + ["python", "-m", "post_training.sft.bundle", "verify", "--dataset-dir", str(output)],
        cwd=worktree,
    )


def publish(profile: base.SFTProfileSpec, **kwargs) -> int:
    prepare(
        profile,
        replay_root=kwargs["replay_root"],
        prepared_dir=kwargs.get("prepared_dir"),
        output_dir=kwargs.get("output_dir"),
        parent_consumed_tokens=kwargs.get("parent_consumed_tokens"),
        revision=kwargs.get("revision"),
    )
    return base.publish(profile, **kwargs)


def train(
    profile: base.SFTProfileSpec,
    *,
    dataset_dir: str | None,
    parent_repo_id: str | None,
    checkpoint_repo_id: str | None,
    max_steps_this_session: int | None,
    wandb_entity: str | None,
) -> int:
    worktree = base._prepare_worktree(profile)
    bundle = base._find_bundle(dataset_dir)
    parent_repo = parent_repo_id or os.environ.get("SMALL_LLM_HF_REPO_ID")
    checkpoint_repo = checkpoint_repo_id or os.environ.get("SMALL_LLM_SFT_HF_REPO_ID", parent_repo)
    if not parent_repo:
        raise base.RuntimeFailure("pass --parent-repo-id or set SMALL_LLM_HF_REPO_ID")
    if not checkpoint_repo:
        raise base.RuntimeFailure("pass --checkpoint-repo-id or set SMALL_LLM_SFT_HF_REPO_ID")
    entity = wandb_entity or os.environ.get("WANDB_ENTITY")
    _require_stable_parent_artifact(
        repo_id=parent_repo,
        run_id=profile.parent_run_id,
        token=os.environ.get("HF_TOKEN"),
    )
    base._wandb_preflight(profile, worktree=worktree, entity=entity)

    trainer_args = [
        "--dataset-dir", str(bundle),
        "--checkpoint-dir", str(profile.checkpoint_dir),
        "--sft-run-id", profile.sft_run_id,
        "--parent-repo-id", parent_repo,
        "--parent-run-id", profile.parent_run_id,
        "--parent-pointer", "best",
        "--checkpoint-repo-id", checkpoint_repo,
        "--device", "cuda",
        "--precision", "fp16",
        "--microbatch-size", str(profile.microbatch_size),
        "--learning-rate", str(profile.learning_rate),
        "--checkpoint-every-steps", str(profile.cadence_steps),
        "--evaluation-every-steps", str(profile.cadence_steps),
        "--remote-publish-every-steps", str(profile.cadence_steps),
        "--validation-blocks", str(INLINE_VALIDATION_BLOCKS),
        "--behavior-cases", str(INLINE_BEHAVIOR_CASES),
        "--wandb-mode", "online",
        "--wandb-project", "Small-LLM",
        "--wandb-run-id", profile.wandb_run_id,
        "--wandb-run-name", profile.wandb_run_name,
    ]
    if entity:
        trainer_args += ["--wandb-entity", entity]
    if max_steps_this_session is not None:
        trainer_args += ["--max-steps-this-session", str(max_steps_this_session)]

    command = ["env", *KAGGLE_SFT_PROCESS_ENV] + base._uv_prefix(wandb=True) + dual_t4_runtime.qualified_runtime_uv_args() + [
        "python", "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=2",
        str(worktree / "kaggle" / "dual_t4_sft.py"),
        "--worktree", str(worktree),
        "--sft-fraction-numerator", str(profile.sft_fraction_numerator),
        "--sft-fraction-denominator", str(profile.sft_fraction_denominator),
        *trainer_args,
    ]
    return base._run(command, cwd=worktree)


def evaluate(profile: base.SFTProfileSpec, **kwargs) -> int:
    return base.evaluate(profile, **kwargs)


__all__ = [
    "INLINE_BEHAVIOR_CASES",
    "INLINE_VALIDATION_BLOCKS",
    "KAGGLE_SFT_PROCESS_ENV",
    "evaluate",
    "prepare",
    "publish",
    "train",
]
