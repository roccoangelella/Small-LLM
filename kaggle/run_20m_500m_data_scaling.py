#!/usr/bin/env python3
"""Fail-closed 500M profile overlay for the proven 20M/100M Kaggle launcher.

The training mechanics remain identical to the qualified 100M path.  This module
binds a distinct finite dataset identity, W&B identity, output namespace, and
500M manifest envelope while reusing exact-resume, numerical gates, validation,
and checkpoint publication code.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import run_20m_100m_data_scaling as base

DEFAULT_COMMIT = "__PIN_20M_500M_LAUNCH_COMMIT__"
DATASET_RUN_ID = "20m-500m-dataset-001"
PROFILE = "20m-500m-data-scaling-v1"
TARGET_SOURCE_TOKENS = 500_000_000
MINIMUM_SOURCE_TOKENS = 450_000_000
MAXIMUM_SOURCE_TOKENS = 550_000_000
CHECKPOINT_SOURCE_TOKENS = 20_000_000
RUN_NAME = "20M model on 500M tokens"
WANDB_RUN_ID = "20m-500m-data-001"
ROOT = base.common.WORK / "small-llm-20m-500m-data-scaling"
WORKTREE = ROOT / "launch-worktree"
EVIDENCE = ROOT / ("evidence-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
CHECKPOINTS = ROOT / "checkpoints"
SUMMARY = base.common.WORK / "small_llm_20m_500m_data_scaling_summary.json"

_EXPECTED_BASE = {
    "DATASET_RUN_ID": "20m-100m-dataset-001",
    "PROFILE": "20m-100m-data-scaling-v1",
}
for _name, _expected in _EXPECTED_BASE.items():
    if getattr(base, _name, None) != _expected:
        raise RuntimeError(
            f"500M launcher base contract changed: {_name}="
            f"{getattr(base, _name, None)!r}, expected {_expected!r}"
        )

_original_wandb_preflight_command = base.wandb_preflight_command
_original_trainer_command = base.trainer_command
_builtin_print = print


def arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--launch-commit",
        default=os.environ.get("SMALL_LLM_500M_LAUNCH_COMMIT", DEFAULT_COMMIT),
    )
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument(
        "--max-steps-this-session",
        type=int,
        default=base.MAX_STEPS_PER_SESSION,
    )
    return parser.parse_args(argv)


def profile_match(root: Path) -> tuple[bool, dict[str, Any]]:
    manifest_path = root / "manifest.json"
    drive_path = root / "drive_manifest.json"
    row: dict[str, Any] = {
        "root": str(root),
        "manifest": manifest_path.is_file(),
        "drive_manifest": drive_path.is_file(),
        "train": (root / "train").is_dir(),
        "validation": (root / "validation").is_dir(),
    }
    if not all(row[key] for key in ("manifest", "drive_manifest", "train", "validation")):
        return False, row
    manifest = base.read_object(manifest_path, "dataset manifest")
    production = manifest.get("production")
    top = {
        "schema_version": 2,
        "sequence_format": "context_plus_one",
        "context_length": 2048,
        "stored_tokens_per_sequence": 2049,
        "sequences_per_block": 16,
        "target_shard_bytes": 8 * 1024 * 1024,
    }
    prod = {
        "run_id": DATASET_RUN_ID,
        "target_source_tokens": TARGET_SOURCE_TOKENS,
        "minimum_source_tokens": MINIMUM_SOURCE_TOKENS,
        "maximum_source_tokens": MAXIMUM_SOURCE_TOKENS,
        "checkpoint_source_tokens": CHECKPOINT_SOURCE_TOKENS,
        "target_reached": True,
        "remote_required": True,
    }
    matched = all(manifest.get(key) == value for key, value in top.items())
    matched = matched and isinstance(production, Mapping)
    if isinstance(production, Mapping):
        matched = matched and all(production.get(key) == value for key, value in prod.items())
    row["run_id"] = production.get("run_id") if isinstance(production, Mapping) else None
    if matched:
        row["manifest_sha256"] = base.common.sha256(manifest_path)
        row["drive_manifest_sha256"] = base.common.sha256(drive_path)
    return bool(matched), row


def find_dataset(explicit: Path | None) -> tuple[Path, list[dict[str, Any]]]:
    roots = (
        [explicit.resolve()]
        if explicit
        else sorted({path.parent for path in base.common.INPUT.rglob("manifest.json")})
    )
    inspected: list[dict[str, Any]] = []
    matches: list[Path] = []
    for root in roots:
        matched, row = profile_match(root)
        inspected.append(row)
        if matched:
            matches.append(root)
    if len(matches) != 1:
        raise base.LaunchFailure(
            f"Expected exactly one attached 500M dataset; found {len(matches)}.\n"
            + json.dumps(inspected, indent=2)
        )
    return matches[0], inspected


def wandb_preflight_command(
    uv: str,
    evidence: Path,
    entity: str | None = None,
) -> tuple[list[str], Path, Path]:
    command, root, result = _original_wandb_preflight_command(uv, evidence, entity)
    command[command.index("--run-name") + 1] = RUN_NAME
    return command, root, result


def trainer_command(
    uv: str,
    dataset: Path,
    plan: Mapping[str, Any],
    checkpoint_dir: Path,
    *,
    additional_steps: int,
    microbatch: int,
    online: bool,
    entity: str | None = None,
    resume: str | None = None,
) -> list[str]:
    command = _original_trainer_command(
        uv,
        dataset,
        plan,
        checkpoint_dir,
        additional_steps=additional_steps,
        microbatch=microbatch,
        online=online,
        entity=entity,
        resume=resume,
    )
    if "--wandb-run-name" in command:
        command[command.index("--wandb-run-name") + 1] = RUN_NAME
    command = ["500m-tokens" if item == "100m-tokens" else item for item in command]
    return command


def profile_print(*values: object, **kwargs: object) -> None:
    replaced = [
        value.replace("100M-token run completed", "500M-token run completed")
        if isinstance(value, str)
        else value
        for value in values
    ]
    _builtin_print(*replaced, **kwargs)


def install_profile() -> None:
    base.DEFAULT_COMMIT = DEFAULT_COMMIT
    base.DATASET_RUN_ID = DATASET_RUN_ID
    base.PROFILE = PROFILE
    base.ROOT = ROOT
    base.WORKTREE = WORKTREE
    base.EVIDENCE = EVIDENCE
    base.CHECKPOINTS = CHECKPOINTS
    base.SUMMARY = SUMMARY
    base.WANDB_RUN_ID = WANDB_RUN_ID
    base.arguments = arguments
    base.profile_match = profile_match
    base.find_dataset = find_dataset
    base.wandb_preflight_command = wandb_preflight_command
    base.trainer_command = trainer_command
    base.print = profile_print


install_profile()

# Re-export the training surface used by the console adapter, entry point, and
# offline tests.  Functions imported from the base module read the overlaid
# globals above at call time.
LaunchFailure = base.LaunchFailure
common = base.common
MAX_STEPS_PER_SESSION = base.MAX_STEPS_PER_SESSION
LOCAL_EVERY = base.LOCAL_EVERY
EVAL_EVERY = base.EVAL_EVERY
REMOTE_EVERY = base.REMOTE_EVERY
WANDB_INIT_TIMEOUT_SECONDS = base.WANDB_INIT_TIMEOUT_SECONDS
validate_wandb_preflight_result = base.validate_wandb_preflight_result
compare_probes = base.compare_probes
segment_plan = base.segment_plan
validate_plan = base.validate_plan
main = base.main


def configure_runtime(*, durability_every: int, max_steps_per_session: int, wandb_run_id: str) -> None:
    """Apply entry-point runtime overrides to both this overlay and its base."""

    global LOCAL_EVERY, EVAL_EVERY, REMOTE_EVERY, MAX_STEPS_PER_SESSION, WANDB_RUN_ID
    LOCAL_EVERY = EVAL_EVERY = REMOTE_EVERY = durability_every
    MAX_STEPS_PER_SESSION = max_steps_per_session
    WANDB_RUN_ID = wandb_run_id
    base.LOCAL_EVERY = durability_every
    base.EVAL_EVERY = durability_every
    base.REMOTE_EVERY = durability_every
    base.MAX_STEPS_PER_SESSION = max_steps_per_session
    base.WANDB_RUN_ID = wandb_run_id


if __name__ == "__main__":
    raise SystemExit(main())
