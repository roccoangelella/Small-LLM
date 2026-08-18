#!/usr/bin/env python3
"""Kaggle runtime for 100M/2B reasoning supervised fine-tuning."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import dual_t4_runtime
import rsft_prepare
import sft_runtime as base
import sft_scaled_runtime as scaled

# Pinned implementation used by the detached Kaggle worktree. Re-pinned after
# the repeated-epoch adapter lands.
IMPLEMENTATION_COMMIT = "28b854b58068ba30d1557c317483da524639a2a0"
PARENT_RUN_ID = "100m-2b-sft-s0-001"
PRODUCTION_RUN_ID = "100m-2b-rsft-r0-001"
PILOT_RUN_IDS = {
    "atomic": "100m-2b-rsft-r0-atomic-pilot-001",
    "textual": "100m-2b-rsft-r0-textual-pilot-001",
}
DEFAULT_MICROBATCH_SIZE = 2
DEFAULT_CADENCE_STEPS = 250
DEFAULT_LEARNING_RATE = 3e-5
PRODUCTION_OPTIMIZER_TARGET_TOKENS = 32_768
CANONICAL_SPECIAL_TOKENS = {
    "reasoning_start": {"id": 50_257, "text": "<think>"},
    "reasoning_end": {"id": 50_258, "text": "</think>"},
    "answer_start": {"id": 50_259, "text": "<answer>"},
}


@dataclass(frozen=True, slots=True)
class RSFTProfile(base.SFTProfileSpec):
    @property
    def run_root(self) -> Path:
        return base.WORK / f"small-llm-{self.sft_run_id}"

    @property
    def default_bundle(self) -> Path:
        return base.WORK / f"{self.sft_run_id}-bundle"


def default_pilot_run_id(delimiter_format: str) -> str:
    try:
        return PILOT_RUN_IDS[delimiter_format]
    except KeyError as error:
        raise base.RuntimeFailure("--delimiter-format must be atomic or textual") from error


def default_experiment_run_id(delimiter_format: str, *, num_epochs: int) -> str:
    if isinstance(num_epochs, bool) or not isinstance(num_epochs, int) or num_epochs <= 0:
        raise base.RuntimeFailure("--num-epochs must be a positive integer")
    if num_epochs == 1:
        return default_pilot_run_id(delimiter_format)
    if delimiter_format not in PILOT_RUN_IDS:
        raise base.RuntimeFailure("--delimiter-format must be atomic or textual")
    return f"100m-2b-rsft-r0-{delimiter_format}-repeat-e{num_epochs}-001"


def resolve_profile(
    model_parameters: int,
    parent_training_tokens: int,
    *,
    run_id: str,
    delimiter_format: str,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    num_epochs: int = 1,
) -> RSFTProfile:
    if (model_parameters, parent_training_tokens) != (100_000_000, 2_000_000_000):
        raise base.RuntimeFailure("the Kaggle R-SFT launcher supports only the 100M/2B parent")
    if not run_id or run_id.strip() != run_id:
        raise base.RuntimeFailure("--run-id must be a non-empty stable identifier without outer whitespace")
    if delimiter_format not in {"atomic", "textual"}:
        raise base.RuntimeFailure("R-SFT delimiter format must be atomic or textual")
    if not isinstance(learning_rate, float) or learning_rate <= 0:
        raise base.RuntimeFailure("R-SFT learning rate must be positive")
    if isinstance(num_epochs, bool) or not isinstance(num_epochs, int) or num_epochs <= 0:
        raise base.RuntimeFailure("R-SFT --num-epochs must be a positive integer")
    return RSFTProfile(
        model_parameters=model_parameters,
        parent_training_tokens=parent_training_tokens,
        model_label="100M",
        token_label="2B",
        token_key="2b",
        parent_run_id=PARENT_RUN_ID,
        sft_run_id=run_id,
        wandb_run_id=run_id,
        wandb_run_name=(
            f"100M / 2B S0 parent / R-SFT R0 / {delimiter_format} / epochs={num_epochs}"
        ),
        dataset_slug=run_id,
        known_parent_consumed_tokens=None,
        launch_commit=IMPLEMENTATION_COMMIT,
        sft_fraction_numerator=1,
        sft_fraction_denominator=2,
        microbatch_size=DEFAULT_MICROBATCH_SIZE,
        cadence_steps=DEFAULT_CADENCE_STEPS,
        learning_rate=learning_rate,
    )


def _read_json(path: Path, *, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as error:
        raise base.RuntimeFailure(f"{label} is missing or invalid: {path}") from error
    if not isinstance(payload, Mapping):
        raise base.RuntimeFailure(f"{label} must be a JSON object: {path}")
    return dict(payload)


def _read_bundle_manifest(bundle: Path) -> dict[str, object]:
    return _read_json(bundle / "bundle-manifest.json", label="R-SFT bundle manifest")


def _bundle_target_budget(bundle: Path) -> int:
    payload = _read_bundle_manifest(bundle)
    value = payload.get("train_target_tokens_requested")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise base.RuntimeFailure("R-SFT bundle has no positive train_target_tokens_requested")
    return value


def _metadata_special_tokens(payload: Mapping[str, object]) -> dict[str, dict[str, object]]:
    special = payload.get("special_tokens")
    if not isinstance(special, Mapping):
        raise base.RuntimeFailure("R-SFT reasoning-token metadata has no special_tokens map")
    normalized: dict[str, dict[str, object]] = {}
    for role in CANONICAL_SPECIAL_TOKENS:
        row = special.get(role)
        if not isinstance(row, Mapping):
            raise base.RuntimeFailure(f"R-SFT reasoning-token metadata has no {role!r} entry")
        normalized[role] = dict(row)
    return normalized


def _require_canonical_token_metadata(payload: Mapping[str, object], *, label: str) -> None:
    if payload.get("base_encoding") != "gpt2" or payload.get("semantic_vocab_size") != 50_260:
        raise base.RuntimeFailure(f"{label} does not use the frozen GPT-2 + 3-token R-SFT vocabulary")
    actual = _metadata_special_tokens(payload)
    if actual != CANONICAL_SPECIAL_TOKENS:
        raise base.RuntimeFailure(
            f"{label} does not match the frozen production reasoning-token contract: {actual}"
        )


def _require_canonical_token_spec(path: Path) -> None:
    payload = _read_json(path, label="R-SFT token spec")
    compact_keys = {"reasoning_start", "reasoning_end", "answer_start"}
    if set(payload) == compact_keys:
        compact = {
            "reasoning_start": payload["reasoning_start"],
            "reasoning_end": payload["reasoning_end"],
            "answer_start": payload["answer_start"],
        }
        expected = {
            "reasoning_start": "<think>",
            "reasoning_end": "</think>",
            "answer_start": "<answer>",
        }
        if compact != expected:
            raise base.RuntimeFailure(
                f"production R-SFT token spec must be exactly {expected}, got {compact}"
            )
        return
    nested = payload.get("reasoning_tokenizer")
    if isinstance(nested, Mapping):
        payload = dict(nested)
    _require_canonical_token_metadata(payload, label="production R-SFT token spec")


def _require_atomic_production_bundle(bundle: Path) -> int:
    manifest = _read_bundle_manifest(bundle)
    rsft = manifest.get("rsft")
    if not isinstance(rsft, Mapping):
        raise base.RuntimeFailure("production R-SFT bundle has no rsft metadata")
    if rsft.get("stage") != "r_sft_r0":
        raise base.RuntimeFailure("production R-SFT bundle has the wrong stage")
    if rsft.get("delimiter_format") != "atomic":
        raise base.RuntimeFailure(
            "production R-SFT requires atomic special-token serialization; textual bundles are ablation-only"
        )
    if rsft.get("contract") != "atomic-production-v1":
        raise base.RuntimeFailure(
            "production R-SFT requires an atomic-production-v1 bundle; pilot bundles are not accepted"
        )
    reasoning_tokenizer = rsft.get("reasoning_tokenizer")
    if not isinstance(reasoning_tokenizer, Mapping):
        raise base.RuntimeFailure("production R-SFT bundle has no reasoning-tokenizer metadata")
    _require_canonical_token_metadata(
        reasoning_tokenizer,
        label="production R-SFT bundle reasoning tokenizer",
    )
    if manifest.get("optimizer_target_tokens") != PRODUCTION_OPTIMIZER_TARGET_TOKENS:
        raise base.RuntimeFailure(
            "production R-SFT bundle must use the frozen 32,768-target optimizer block; "
            "2,048-target bundles are pilot-ablation-only"
        )
    return _bundle_target_budget(bundle)


def _resolve_token_spec(
    explicit: str | None,
    *,
    bundle: Path,
    worktree: Path,
) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser().resolve())
    candidates.append(bundle / "reasoning-tokens.json")
    candidates.append((worktree / rsft_prepare.TOKEN_SPEC_RELATIVE_PATH).resolve())
    for candidate in candidates:
        if candidate.is_file() and not candidate.is_symlink():
            return candidate
    raise base.RuntimeFailure(
        "R-SFT token spec is missing; the canonical spec should exist in the bundle or pinned worktree"
    )


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
    num_epochs: int,
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
        "--rsft-num-epochs",
        str(num_epochs),
        *trainer_args,
    ]


def train(
    profile: RSFTProfile,
    *,
    dataset_dir: str | None,
    delimiter_format: str,
    token_spec: str | None,
    s0_bundle: str | None,
    parent_repo_id: str | None,
    checkpoint_repo_id: str | None,
    max_steps_this_session: int | None,
    wandb_entity: str | None,
    num_epochs: int = 1,
    production: bool = False,
    dry_run: bool = False,
) -> int:
    if isinstance(num_epochs, bool) or not isinstance(num_epochs, int) or num_epochs <= 0:
        raise base.RuntimeFailure("R-SFT --num-epochs must be a positive integer")
    if production and num_epochs != 1:
        raise base.RuntimeFailure(
            "canonical production R-SFT remains one-pass; use the ablation lane for explicit repeat experiments"
        )

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

    worktree = base.REPO if dry_run else base._prepare_worktree(profile)
    preparation: dict[str, object] | None = None
    if production:
        if delimiter_format != "atomic":
            raise base.RuntimeFailure("production R-SFT is atomic-only")
        if not dataset_dir:
            raise base.RuntimeFailure(
                "production R-SFT requires --dataset-dir pointing to a frozen atomic-production-v1 bundle"
            )
        bundle = base._find_bundle(dataset_dir)
        exact_targets: int | None = _require_atomic_production_bundle(bundle)
    elif dataset_dir:
        bundle = base._find_bundle(dataset_dir)
        exact_targets = _bundle_target_budget(bundle)
    elif dry_run:
        preparation = rsft_prepare.preparation_plan(worktree=worktree, s0_bundle=s0_bundle)
        bundle = Path(str(preparation[f"{delimiter_format}_bundle"]))
        exact_targets = None
    else:
        matched_root = rsft_prepare.prepare_pilot_bundles(
            worktree=worktree,
            s0_bundle=s0_bundle,
        )
        bundle = base._find_bundle(str(matched_root / delimiter_format))
        exact_targets = _bundle_target_budget(bundle)

    token_spec_path = _resolve_token_spec(
        token_spec,
        bundle=bundle,
        worktree=worktree,
    )
    if production:
        _require_canonical_token_spec(token_spec_path)

    if not dry_run:
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
        num_epochs=num_epochs,
    )
    if dry_run:
        total_targets = None if exact_targets is None else exact_targets * num_epochs
        print(
            json.dumps(
                {
                    "schema": "small-llm-rsft-kaggle-dry-run-v4",
                    "topology": "2xTesla-T4-DDP",
                    "stage": "r_sft_r0",
                    "contract": "atomic-production-v1" if production else (
                        "pilot-ablation-v1" if num_epochs == 1 else "pilot-repeat-v1"
                    ),
                    "parent_run_id": profile.parent_run_id,
                    "run_id": profile.sft_run_id,
                    "delimiter_format": delimiter_format,
                    "bundle": str(bundle),
                    "bundle_target_tokens_one_pass": exact_targets,
                    "requested_target_tokens_all_epochs": total_targets,
                    "num_epochs": num_epochs,
                    "budget_mode": "bundle-exact-one-pass" if num_epochs == 1 else "bundle-exact-repeat",
                    "microbatch_size_per_rank": profile.microbatch_size,
                    "learning_rate": profile.learning_rate,
                    "auto_preparation": preparation,
                    "command": command,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    return base._run(command, cwd=worktree)


__all__ = [
    "CANONICAL_SPECIAL_TOKENS",
    "DEFAULT_CADENCE_STEPS",
    "DEFAULT_LEARNING_RATE",
    "DEFAULT_MICROBATCH_SIZE",
    "IMPLEMENTATION_COMMIT",
    "PARENT_RUN_ID",
    "PILOT_RUN_IDS",
    "PRODUCTION_OPTIMIZER_TARGET_TOKENS",
    "PRODUCTION_RUN_ID",
    "RSFTProfile",
    "build_train_command",
    "default_experiment_run_id",
    "default_pilot_run_id",
    "resolve_profile",
    "train",
]
