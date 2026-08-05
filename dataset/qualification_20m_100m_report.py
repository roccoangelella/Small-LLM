"""Derive the exact one-pass trainer plan for the 20M/100M experiment."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Mapping

from dataset import config
from dataset import qualification_20m_report as shared
from dataset.qualification_20m_100m import (
    CHECKPOINT_SOURCE_TOKENS,
    CONTEXT_LENGTH,
    MAXIMUM_SOURCE_TOKENS,
    MINIMUM_SOURCE_TOKENS,
    SEQUENCES_PER_BLOCK,
    TARGET_SHARD_BYTES,
    TARGET_SOURCE_TOKENS,
)
from dataset.src.storage import write_json_atomic

PROFILE = "20m-100m-one-pass-v1"


def _validate_identity(manifest: Mapping[str, object]) -> None:
    expected_top_level = {
        "schema_version": 2,
        "sequence_format": "context_plus_one",
        "context_length": CONTEXT_LENGTH,
        "stored_tokens_per_sequence": CONTEXT_LENGTH + 1,
        "sequences_per_block": SEQUENCES_PER_BLOCK,
        "target_shard_bytes": TARGET_SHARD_BYTES,
    }
    for key, expected in expected_top_level.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"100M qualification manifest {key} mismatch: "
                f"expected {expected!r}, got {manifest.get(key)!r}"
            )

    production = manifest.get("production")
    if not isinstance(production, Mapping):
        raise ValueError("100M qualification manifest has no production identity")
    expected_production = {
        "target_source_tokens": TARGET_SOURCE_TOKENS,
        "minimum_source_tokens": MINIMUM_SOURCE_TOKENS,
        "maximum_source_tokens": MAXIMUM_SOURCE_TOKENS,
        "checkpoint_source_tokens": CHECKPOINT_SOURCE_TOKENS,
        "remote_required": True,
    }
    for key, expected in expected_production.items():
        if production.get(key) != expected:
            raise ValueError(
                f"100M qualification production {key} mismatch: "
                f"expected {expected!r}, got {production.get(key)!r}"
            )

    accepted = shared._require_integer(
        manifest.get("accepted_source_tokens"),
        name="accepted_source_tokens",
        minimum=1,
    )
    if not MINIMUM_SOURCE_TOKENS <= accepted <= MAXIMUM_SOURCE_TOKENS:
        raise ValueError("accepted source tokens fall outside 100M bounds")
    if production.get("target_reached") != (accepted >= TARGET_SOURCE_TOKENS):
        raise ValueError("production target_reached disagrees with accepted source tokens")


def derive_plan(
    manifest: Mapping[str, object],
    *,
    manifest_path: Path | None = None,
    drive_manifest_path: Path | None = None,
) -> dict[str, object]:
    """Validate the fixed profile and return exact one-pass launch values."""

    _validate_identity(manifest)
    train = shared._split_summary(manifest, split="train")
    validation = shared._split_summary(manifest, split="validation")
    planned_updates = int(train["block_count"])
    if planned_updates < 20:
        raise ValueError("100M qualification dataset has fewer than 20 train blocks")
    if int(validation["block_count"]) == 0:
        raise ValueError("100M qualification dataset has no validation block")

    warmup_updates = max(16, math.ceil(planned_updates * 0.05))
    decay_updates = math.ceil(planned_updates * 0.20)
    if warmup_updates + decay_updates >= planned_updates:
        raise ValueError("100M qualification dataset is too small for fixed WSD phases")
    stable_updates = planned_updates - warmup_updates - decay_updates

    full_update_tokens = CONTEXT_LENGTH * SEQUENCES_PER_BLOCK
    total_train_tokens = int(train["target_tokens"])
    warmup_tokens = warmup_updates * full_update_tokens
    tokens_before_decay = (planned_updates - decay_updates) * full_update_tokens
    if not 0 < warmup_tokens < tokens_before_decay < total_train_tokens:
        raise ValueError("manifest target tokens are incompatible with the WSD schedule")
    stable_tokens = tokens_before_decay - warmup_tokens
    decay_tokens = total_train_tokens - tokens_before_decay

    accepted = int(manifest["accepted_source_tokens"])
    validation_source = shared._require_integer(
        manifest.get("validation_source_tokens"),
        name="validation_source_tokens",
    )
    if validation_source > accepted:
        raise ValueError("validation source tokens exceed accepted source tokens")

    identity: dict[str, object] = {}
    if manifest_path is not None:
        identity["manifest_path"] = str(manifest_path)
        identity["manifest_sha256"] = shared._sha256(manifest_path)
    if drive_manifest_path is not None:
        drive_manifest = shared._read_object(
            drive_manifest_path,
            label="Drive manifest",
        )
        drive_run_id, drive_shard_count = shared._verify_drive_manifest(
            drive_manifest,
            manifest=manifest,
            train=train,
            validation=validation,
        )
        identity["drive_manifest_path"] = str(drive_manifest_path)
        identity["drive_manifest_sha256"] = shared._sha256(drive_manifest_path)
        identity["drive_run_id"] = drive_run_id
        identity["drive_shard_count"] = drive_shard_count

    return {
        "version": 1,
        "qualification_profile": PROFILE,
        "accepted_source_tokens": accepted,
        "train_source_tokens": accepted - validation_source,
        "validation_source_tokens": validation_source,
        "context_length": CONTEXT_LENGTH,
        "sequences_per_block": SEQUENCES_PER_BLOCK,
        "target_shard_bytes": TARGET_SHARD_BYTES,
        "train": train,
        "validation": validation,
        "trainer": {
            "steps": planned_updates,
            "passes": 1,
            "full_block_target_tokens": full_update_tokens,
            "schedule": "wsd",
            "warmup_updates": warmup_updates,
            "stable_updates": stable_updates,
            "decay_updates": decay_updates,
            "warmup_tokens": warmup_tokens,
            "stable_tokens": stable_tokens,
            "decay_tokens": decay_tokens,
            "minimum_lr_ratio": 0.1,
            "validation_blocks": int(validation["block_count"]),
            "checkpoint_every_steps": 250,
            "evaluation_every_steps": 500,
            "remote_publish_every_steps": 500,
            "microbatch_size": 4,
        },
        "identity": identity,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Derive the exact 20M-model / 100M-token trainer plan."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--drive-manifest", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest_path = args.dataset_dir / config.MANIFEST_FILENAME
    try:
        manifest = shared._read_object(
            manifest_path,
            label="100M qualification manifest",
        )
        plan = derive_plan(
            manifest,
            manifest_path=manifest_path,
            drive_manifest_path=args.drive_manifest,
        )
        if args.output is not None:
            write_json_atomic(args.output, plan)
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    except Exception as error:  # noqa: BLE001 - concise report boundary
        print(
            f"100M qualification report error: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
