#!/usr/bin/env python3
"""Kaggle runtime for the first 100M/2B reasoning-SFT experiments."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import dual_t4_runtime
import sft_runtime as base
import sft_scaled_runtime as scaled

# This commit contains the R-SFT launcher, DDP adapter, tokenizer/model transition,
# and shared SFT runtime contract consumed inside the detached Kaggle worktree.
IMPLEMENTATION_COMMIT = "96a8fc399fd919f54e73d8b9c4689e698e476cc7"
PARENT_RUN_ID = "100m-2b-sft-s0-001"
DEFAULT_MICROBATCH_SIZE = 2
DEFAULT_CADENCE_STEPS = 250
DEFAULT_LEARNING_RATE = 3e-5


@dataclass(frozen=True, slots=True)
class RSFTProfile(base.SFTProfileSpec):
    @property
    def run_root(self) -> Path:
        return base.WORK / f"small-llm-{self.sft_run_id}"

    @property
    def default_bundle(self) -> Path:
        # Production R-SFT bundles are attached explicitly. This property only
        # keeps the inherited profile contract complete.
        return base.WORK / f"{self.sft_run_id}-bundle"


def resolve_profile(
    model_parameters: int,
    parent_training_tokens: int,
    *,
    run_id: str,
    delimiter_format: str,
    learning_rate: float = DEFAULT_LEARNING_RATE,
) -> RSFTProfile:
    if (model_parameters, parent_training_tokens) != (100_000_000, 2_000_000_000):
        raise base.RuntimeFailure("the first Kaggle R-SFT launcher supports only the 100M/2B parent")
    if not run_id or run_id.strip() != run_id:
        raise base.RuntimeFailure("--run-id must be a non-empty stable identifier without outer whitespace")
    if delimiter_format not in {"atomic", "textual"}:
        raise base.RuntimeFailure("--delimiter-format must be atomic or textual")
    if not isinstance(learning_rate, float) or learning_rate <= 0:
        raise base.RuntimeFailure("R-SFT learning rate must be positive")
    return RSFTProfile(
        model_parameters=model_parameters,
        parent_training_tokens=parent_training_tokens,
        model_label="100M",
        token_label="2B",
        token_key="2b",
        parent_run_id=PARENT_RUN_ID,
        sft_run_id=run_id,
        wandb_run_id=run_id,
        wandb_run_name=f"100M / 2B S0 parent / R-SFT R0 / {delimiter_format}",
        dataset_slug=run_id,
        known_parent_consumed_tokens=None,
        launch_commit=IMPLEMENTATION_COMMIT,
        # Internal compatibility values only. dual_t4_rsft.py replaces the
        # inherited fraction budget with the bundle's exact target count.
        sft_fraction_numerator=1,
        sft_fraction_denominator=2,
        microbatch_size=DEFAULT_MICROBATCH_SIZE,
        cadence_steps=DEFAULT_CADENCE_STEPS,
        learning_rate=learning_rate,
    )


def _bundle_target_budget(bundle: Path) -> int:
    try:
        payload = json.loads((bundle / "bundle-manifest.json").read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as error:
        raise base.RuntimeFailure(f"invalid R-SFT bundle manifest: {bundle}") from error
    if not isinstance(payload, dict):
        raise base.RuntimeFailure("R-SFT bundle manifest must be a JSON object")
    value = payload.get("train_target_tokens_requested")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise base.RuntimeFailure("R-SFT bundle has no positive train_target_tokens_requested")
    return value


def build_train_command(
    profile: RSFTProfile,
    *,
    worktree: Path,
    bundle: Path,
    delimiter_format: str,
    token_spec: Path,
    parent_repo_id: str,
    checkpoint_repo_id: str,
    wandb_entity: str | None,
    max_steps_this_session: int | None,
) -> list[str]:
    trainer_args = [
        "--dataset-dir", str(bundle),
        "--checkpoint-dir", str(profile.checkpoint_dir),
        "--sft-run-id", profile.sft_run_id,
        "--parent-repo-id", parent_repo_id,
        "--parent-run-id", profile.parent_run_id,
        "--parent-pointer", "latest",
        "--checkpoint-repo-id", checkpoint_repo_id,
        "--device", "cuda",
        "--precision", "fp16",
        "--microbatch-size", str(profile.microbatch_size),
        "--learning-rate", str(profile.learning_rate),
        "--checkpoint-every-steps", str(profile.cadence_steps),
        "--evaluation-every-steps", str(profile.cadence_steps),
        "--remote-publish-every-steps", str(profile.cadence_steps),
        "--validation-blocks", str(scaled.INLINE_VALIDATION_BLOCKS),
        "--behavior-cases", str(scaled.INLINE_BEHAVIOR_CASES),
        "--wandb-mode", "online",
        "--wandb-project", "Small-LLM",
        "--wandb-run-id", profile.wandb_run_id,
        "--wandb-run-name", profile.wandb_run_name,
    ]
    if wandb_entity:
        trainer_args += ["--wandb-entity", wandb_entity]
    if max_steps_this_session is not None:
        trainer_args += ["--max-steps-this-session", str(max_steps_this_session)]

    return [
        "env",
        *scaled.KAGGLE_SFT_PROCESS_ENV,
        *base._uv_prefix(wandb=True),
        *dual_t4_runtime.qualified_runtime_uv_args(),
        "python",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=2",
        str(worktree / "kaggle" / "dual_t4_rsft.py"),
        "--worktree",
        str(worktree),
        "--rsft-delimiter-format",
        delimiter_format,
        "--rsft-token-spec",
        str(token_spec),
        *trainer_args,
    ]


def train(
    profile: RSFTProfile,
    *,
    dataset_dir: str,
    delimiter_format: str,
    token_spec: str,
    parent_repo_id: str | None,
    checkpoint_repo_id: str | None,
    max_steps_this_session: int | None,
    wandb_entity: str | None,
    dry_run: bool = False,
) -> int:
    bundle = base._find_bundle(dataset_dir)
    exact_targets = _bundle_target_budget(bundle)
    token_spec_path = Path(token_spec).expanduser().resolve()
    if not token_spec_path.is_file() or token_spec_path.is_symlink():
        raise base.RuntimeFailure(f"R-SFT token spec is missing or unsafe: {token_spec_path}")

    parent_repo = (
        parent_repo_id
        or os.environ.get("SMALL_LLM_SFT_HF_REPO_ID")
        or os.environ.get("SMALL_LLM_HF_REPO_ID")
    )
    checkpoint_repo = (
        checkpoint_repo_id
        or os.environ.get("SMALL_LLM_RSFT_HF_REPO_ID")
        or os.environ.get("SMALL_LLM_SFT_HF_REPO_ID")
        or os.environ.get("SMALL_LLM_HF_REPO_ID")
    )
    if dry_run:
        parent_repo = parent_repo or "<SMALL_LLM_SFT_HF_REPO_ID>"
        checkpoint_repo = checkpoint_repo or "<SMALL_LLM_RSFT_HF_REPO_ID>"
    if not parent_repo:
        raise base.RuntimeFailure(
            "pass --parent-repo-id or set SMALL_LLM_SFT_HF_REPO_ID/SMALL_LLM_HF_REPO_ID"
        )
    if not checkpoint_repo:
        raise base.RuntimeFailure(
            "pass --checkpoint-repo-id or set SMALL_LLM_RSFT_HF_REPO_ID/SMALL_LLM_SFT_HF_REPO_ID"
        )
    entity = wandb_entity or os.environ.get("WANDB_ENTITY")

    if dry_run:
        worktree = base.REPO
    else:
        worktree = base._prepare_worktree(profile)
        base._wandb_preflight(profile, worktree=worktree, entity=entity)

    command = build_train_command(
        profile,
        worktree=worktree,
        bundle=bundle,
        delimiter_format=delimiter_format,
        token_spec=token_spec_path,
        parent_repo_id=parent_repo,
        checkpoint_repo_id=checkpoint_repo,
        wandb_entity=entity,
        max_steps_this_session=max_steps_this_session,
    )
    if dry_run:
        print(
            json.dumps(
                {
                    "schema": "small-llm-rsft-kaggle-dry-run-v1",
                    "topology": "2xTesla-T4-DDP",
                    "stage": "r_sft_r0",
                    "parent_run_id": profile.parent_run_id,
                    "run_id": profile.sft_run_id,
                    "delimiter_format": delimiter_format,
                    "bundle": str(bundle),
                    "bundle_target_tokens": exact_targets,
                    "budget_mode": "bundle-exact-one-pass",
                    "microbatch_size_per_rank": profile.microbatch_size,
                    "learning_rate": profile.learning_rate,
                    "command": command,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    return base._run(command, cwd=worktree)


__all__ = [
    "DEFAULT_CADENCE_STEPS",
    "DEFAULT_LEARNING_RATE",
    "DEFAULT_MICROBATCH_SIZE",
    "IMPLEMENTATION_COMMIT",
    "PARENT_RUN_ID",
    "RSFTProfile",
    "build_train_command",
    "resolve_profile",
    "train",
]
