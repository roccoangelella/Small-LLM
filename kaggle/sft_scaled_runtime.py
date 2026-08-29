#!/usr/bin/env python3
"""Scaled SFT runtime extensions."""
from __future__ import annotations

from dataclasses import replace
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

# ADR-0122 is deliberately narrow: only the completed 100M/2B parent at exactly
# 10% uses the capacity-aware one-pass instruction recipe and frozen held-outs.
TEN_PERCENT_PARENT_TARGETS = 2_001_000_448
TEN_PERCENT_TRAIN_TARGETS = 200_100_044
TEN_PERCENT_RECIPE = "s0-10pct-capacity-aware-v1"
# This commit contains the dedicated builder plus its routing regression tests.
# The historical 4% profile keeps its older launch pin; only 10% bundle creation
# temporarily materializes a worktree at this implementation commit.
TEN_PERCENT_BUILD_COMMIT = "fdfab079bacbb8a1098bdcee7451347cf28bc1f6"
# ADR-0130 supersedes the earlier 15%-hold proposal. The training worktree is
# pinned to the commit that contains the exact step-64 warmup, peak-through-3000
# schedule, its dedicated dual-T4 wrapper, and regression tests.
TEN_PERCENT_TRAIN_COMMIT = "caa7fa54fe16510d30ef92eca19d95f86585e20e"
TEN_PERCENT_TRAJECTORY_RUN_ID = "100m-2b-sft-s0-10pct-peak3000-001"
TEN_PERCENT_TRAJECTORY_WANDB_NAME = "100M / 2B parent / SFT S0 / 10% / peak-through-3000"

# Verified private Kaggle publication accepted for the 100M/2B 10% S0 run.
# The publication round-trip reported this exact tree identity. Training binds
# the split identities below because Kaggle extraction does not retain the
# publisher's tree-hash envelope as an input file.
TEN_PERCENT_PUBLISHED_TREE_SHA256 = (
    "c7550a377978231bfcc4d158ab11f8e2604e45921c5acb4e37e9557f12590b4d"
)
TEN_PERCENT_PUBLISHED_FILE_COUNT = 22
TEN_PERCENT_PUBLISHED_TOTAL_BYTES = 773_987_135
TEN_PERCENT_PUBLISHED_SPLITS = {
    "train": {
        "loss_bearing_target_tokens": 200_099_738,
        "manifest_sha256": "feefc3244bd8a2f369eec85e4a95410c2daf479016c04cf02c8042ca5a4010d3",
        "build_report_sha256": "8a131988c43349fb360f56dd41f7f552e9c1533c2550701db67b37ece6e820d7",
    },
    "validation": {
        "loss_bearing_target_tokens": 2_105_945,
        "manifest_sha256": "26cb522729b4525498559d1ce131a181c30fd8fff573f3464e09030be803d09e",
        "build_report_sha256": "37e3c4d98d1e7ed1ec077e0e92b7d79327c4ee2b473f0c4e86f0ec5e4d6c324d",
    },
    "test": {
        # Test is the frozen completed-4% held-out split. The publication
        # snippet did not include its report hash/count, so bind its immutable
        # split-manifest identity, which is already part of ADR-0122.
        "manifest_sha256": "48e99ee51c201da398e227742ca7e023064a408c486cce16e20427d1ec7634d2",
    },
}

# Kaggle's two Python/DDP workers share one host-memory budget. Bound glibc
# arena growth and omit qualification-only optimizer tensor cloning in this
# execution path; neither setting changes optimizer state or model updates.
KAGGLE_SFT_PROCESS_ENV = (
    "MALLOC_ARENA_MAX=2",
    "MALLOC_TRIM_THRESHOLD_=131072",
    "SMALL_LLM_DISABLE_OPTIMIZER_TELEMETRY=1",
)


def _is_capacity_aware_10pct(
    profile: base.SFTProfileSpec,
    *,
    parent_consumed_tokens: int,
) -> bool:
    return (
        profile.model_parameters == 100_000_000
        and profile.parent_training_tokens == 2_000_000_000
        and parent_consumed_tokens == TEN_PERCENT_PARENT_TARGETS
        and profile.sft_fraction_numerator * 10 == profile.sft_fraction_denominator
    )


def _read_capacity_aware_manifest(output: Path) -> dict[str, object]:
    manifest_path = output / "bundle-manifest.json"
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as error:
        raise base.RuntimeFailure("existing 10% S0 bundle manifest is invalid") from error
    if not isinstance(payload, dict):
        raise base.RuntimeFailure("existing 10% S0 bundle manifest is not an object")
    return payload


def _verify_capacity_aware_bundle(output: Path) -> None:
    payload = _read_capacity_aware_manifest(output)
    recipe = payload.get("s0_scaling_recipe")
    if not isinstance(recipe, dict) or recipe.get("name") != TEN_PERCENT_RECIPE:
        raise base.RuntimeFailure(
            "existing 10% S0 bundle does not carry the ADR-0122 capacity-aware recipe"
        )
    if payload.get("train_target_tokens_requested") != TEN_PERCENT_TRAIN_TARGETS:
        raise base.RuntimeFailure("existing ADR-0122 bundle has the wrong train target horizon")
    splits = payload.get("splits")
    expected = recipe.get("expected_heldout_manifest_sha256")
    if not isinstance(splits, dict) or not isinstance(expected, dict):
        raise base.RuntimeFailure("existing ADR-0122 bundle has incomplete frozen held-out metadata")
    for split in ("validation", "test"):
        split_payload = splits.get(split)
        if (
            not isinstance(split_payload, dict)
            or split_payload.get("manifest_sha256") != expected.get(split)
        ):
            raise base.RuntimeFailure(
                f"existing ADR-0122 bundle does not match frozen {split} identity"
            )


def _verify_published_10pct_training_bundle(output: Path) -> None:
    """Bind training to the exact private Kaggle artifact accepted after publication."""

    _verify_capacity_aware_bundle(output)
    payload = _read_capacity_aware_manifest(output)
    splits = payload.get("splits")
    if not isinstance(splits, dict):
        raise base.RuntimeFailure("published 10% S0 bundle has no split metadata")

    for split, expected_fields in TEN_PERCENT_PUBLISHED_SPLITS.items():
        observed = splits.get(split)
        if not isinstance(observed, dict):
            raise base.RuntimeFailure(f"published 10% S0 bundle is missing {split} metadata")
        for field, expected in expected_fields.items():
            actual = observed.get(field)
            if actual != expected:
                raise base.RuntimeFailure(
                    "10% S0 training dataset identity mismatch: "
                    f"split={split} field={field} expected={expected} actual={actual}"
                )

    print(
        "[sft-dataset-preflight] "
        f"recipe={TEN_PERCENT_RECIPE} train_manifest="
        f"{TEN_PERCENT_PUBLISHED_SPLITS['train']['manifest_sha256']} "
        f"publication_tree={TEN_PERCENT_PUBLISHED_TREE_SHA256} status=verified",
        flush=True,
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
    exact_parent_tokens = base._exact_parent_tokens(profile, parent_consumed_tokens)
    capacity_aware = _is_capacity_aware_10pct(
        profile,
        parent_consumed_tokens=exact_parent_tokens,
    )
    worktree_profile = (
        replace(profile, launch_commit=TEN_PERCENT_BUILD_COMMIT)
        if capacity_aware
        else profile
    )
    worktree = base._prepare_worktree(worktree_profile)
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

    expected_targets = base._expected_sft_targets(profile, exact_parent_tokens)
    if base._verify_existing_bundle_budget(output, expected_targets=expected_targets):
        if capacity_aware:
            _verify_capacity_aware_bundle(output)
    else:
        if output.exists():
            raise base.RuntimeFailure(
                f"refusing to replace incomplete/non-bundle SFT output directory: {output}"
            )
        if capacity_aware:
            command = base._uv_prefix() + [
                "python", "-m", "post_training.sft.s0_10pct_bundle",
                "--prepared-dir", str(prepared),
                "--replay-root", str(replay),
                "--output-dir", str(output),
                "--parent-consumed-tokens", str(exact_parent_tokens),
                "--optimizer-target-tokens", "32768",
                "--context-length", "2048",
                "--seed", "17",
            ]
        else:
            command = base._uv_prefix() + [
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
            ]
        base._run(command, cwd=worktree)
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
    exact_parent_tokens = base._exact_parent_tokens(profile, None)
    capacity_aware = _is_capacity_aware_10pct(
        profile,
        parent_consumed_tokens=exact_parent_tokens,
    )
    dataset_profile = profile
    if capacity_aware:
        profile = replace(
            profile,
            sft_run_id=TEN_PERCENT_TRAJECTORY_RUN_ID,
            wandb_run_id=TEN_PERCENT_TRAJECTORY_RUN_ID,
            wandb_run_name=TEN_PERCENT_TRAJECTORY_WANDB_NAME,
            launch_commit=TEN_PERCENT_TRAIN_COMMIT,
        )
    worktree = base._prepare_worktree(profile)
    bundle = base._find_bundle(dataset_dir, dataset_profile)
    if capacity_aware:
        if float(profile.learning_rate) != 3e-5:
            raise base.RuntimeFailure("100M/2B 10% SFT peak LR is frozen at 3e-5")
        _verify_published_10pct_training_bundle(bundle)

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
        "--remote-rolling-latest-only",
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

    runner = "dual_t4_sft_10pct.py" if capacity_aware else "dual_t4_sft.py"
    command = ["env", *KAGGLE_SFT_PROCESS_ENV] + base._uv_prefix(wandb=True) + dual_t4_runtime.qualified_runtime_uv_args() + [
        "python", "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=2",
        str(worktree / "kaggle" / runner),
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
    "TEN_PERCENT_BUILD_COMMIT",
    "TEN_PERCENT_PARENT_TARGETS",
    "TEN_PERCENT_PUBLISHED_FILE_COUNT",
    "TEN_PERCENT_PUBLISHED_SPLITS",
    "TEN_PERCENT_PUBLISHED_TOTAL_BYTES",
    "TEN_PERCENT_PUBLISHED_TREE_SHA256",
    "TEN_PERCENT_RECIPE",
    "TEN_PERCENT_TRAIN_COMMIT",
    "TEN_PERCENT_TRAIN_TARGETS",
    "TEN_PERCENT_TRAJECTORY_RUN_ID",
    "TEN_PERCENT_TRAJECTORY_WANDB_NAME",
    "evaluate",
    "prepare",
    "publish",
    "train",
]
