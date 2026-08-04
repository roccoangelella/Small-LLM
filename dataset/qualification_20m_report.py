"""Derive the exact one-pass trainer plan from a completed qualification manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Mapping

from dataset import config
from dataset.qualification_20m import (
    CHECKPOINT_SOURCE_TOKENS,
    CONTEXT_LENGTH,
    MAXIMUM_SOURCE_TOKENS,
    MINIMUM_SOURCE_TOKENS,
    SEQUENCES_PER_BLOCK,
    TARGET_SHARD_BYTES,
    TARGET_SOURCE_TOKENS,
)
from dataset.src.storage import write_json_atomic


def _read_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as error:
        raise ValueError(f"cannot read {label}: {path}") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must contain a JSON object")
    return dict(payload)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"manifest has invalid {name}")
    return value


def _require_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"manifest has invalid {name}")
    return value


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
                f"qualification manifest {key} mismatch: "
                f"expected {expected!r}, got {manifest.get(key)!r}"
            )

    production = manifest.get("production")
    if not isinstance(production, Mapping):
        raise ValueError("qualification manifest has no production identity")
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
                f"qualification production {key} mismatch: "
                f"expected {expected!r}, got {production.get(key)!r}"
            )

    accepted = _require_integer(
        manifest.get("accepted_source_tokens"),
        name="accepted_source_tokens",
        minimum=1,
    )
    if not MINIMUM_SOURCE_TOKENS <= accepted <= MAXIMUM_SOURCE_TOKENS:
        raise ValueError("accepted source tokens fall outside qualification bounds")
    if production.get("target_reached") != (accepted >= TARGET_SOURCE_TOKENS):
        raise ValueError("production target_reached disagrees with accepted source tokens")


def _split_summary(
    manifest: Mapping[str, object],
    *,
    split: str,
) -> dict[str, object]:
    raw_shards = manifest.get("shards")
    if not isinstance(raw_shards, list):
        raise ValueError("qualification manifest has no shard list")
    shards = [
        item
        for item in raw_shards
        if isinstance(item, Mapping) and item.get("split") == split
    ]
    shards.sort(
        key=lambda item: _require_integer(
            item.get("first_block_id"), name="first_block_id"
        )
    )
    if not shards:
        return {
            "shard_count": 0,
            "block_count": 0,
            "sequence_count": 0,
            "stored_tokens": 0,
            "target_tokens": 0,
            "block_ids": [],
            "shards": [],
        }

    expected_first = 0
    block_ids: list[int] = []
    sequence_count = 0
    stored_tokens = 0
    shard_rows: list[dict[str, object]] = []
    for shard in shards:
        first = _require_integer(shard.get("first_block_id"), name="first_block_id")
        last = _require_integer(shard.get("last_block_id"), name="last_block_id")
        if first != expected_first or last < first:
            raise ValueError(f"{split} shard block ranges are not contiguous")
        block_ids.extend(range(first, last + 1))
        expected_first = last + 1
        sequences = _require_integer(
            shard.get("sequence_count"), name="sequence_count", minimum=1
        )
        tokens = _require_integer(
            shard.get("token_count"), name="token_count", minimum=1
        )
        if tokens != sequences * (CONTEXT_LENGTH + 1):
            raise ValueError(f"{split} shard token count disagrees with sequence count")
        filename = _require_string(shard.get("filename"), name="filename")
        checksum = _require_string(shard.get("checksum"), name="checksum")
        if len(checksum) != 64 or any(character not in "0123456789abcdef" for character in checksum):
            raise ValueError(f"{split} shard has an invalid checksum")
        sequence_count += sequences
        stored_tokens += tokens
        shard_rows.append(
            {
                "filename": filename,
                "byte_size": _require_integer(
                    shard.get("byte_size"), name="byte_size", minimum=1
                ),
                "checksum": checksum,
                "first_block_id": first,
                "last_block_id": last,
                "sequence_count": sequences,
                "target_tokens": sequences * CONTEXT_LENGTH,
            }
        )

    block_count = len(block_ids)
    minimum_sequences = (block_count - 1) * SEQUENCES_PER_BLOCK + 1
    maximum_sequences = block_count * SEQUENCES_PER_BLOCK
    if not minimum_sequences <= sequence_count <= maximum_sequences:
        raise ValueError(
            f"{split} sequence count is incompatible with full blocks plus one final partial block"
        )
    return {
        "shard_count": len(shards),
        "block_count": block_count,
        "sequence_count": sequence_count,
        "stored_tokens": stored_tokens,
        "target_tokens": sequence_count * CONTEXT_LENGTH,
        "block_ids": block_ids,
        "shards": shard_rows,
    }


def _verify_drive_manifest(
    drive_manifest: Mapping[str, object],
    *,
    manifest: Mapping[str, object],
    train: Mapping[str, object],
    validation: Mapping[str, object],
) -> tuple[str, int]:
    production = manifest.get("production")
    if not isinstance(production, Mapping):
        raise ValueError("qualification manifest has no production identity")
    run_id = _require_string(production.get("run_id"), name="production run_id")
    configuration_hash = _require_string(
        production.get("configuration_hash"), name="production configuration_hash"
    )
    schema_hash = _require_string(
        production.get("schema_hash"), name="production schema_hash"
    )
    if drive_manifest.get("version") != 1:
        raise ValueError("Drive manifest has an unsupported version")
    if drive_manifest.get("run_id") != run_id:
        raise ValueError("Drive manifest run ID disagrees with the dataset manifest")
    if drive_manifest.get("configuration_hash") != configuration_hash:
        raise ValueError("Drive manifest configuration hash disagrees with the dataset")
    if drive_manifest.get("schema_hash") != schema_hash:
        raise ValueError("Drive manifest schema hash disagrees with the dataset")

    local_rows = list(train.get("shards", [])) + list(validation.get("shards", []))
    local_by_name = {
        str(item["filename"]): item
        for item in local_rows
        if isinstance(item, Mapping)
    }
    drive_shards = drive_manifest.get("shards")
    if not isinstance(drive_shards, list) or any(
        not isinstance(item, Mapping) for item in drive_shards
    ):
        raise ValueError("Drive manifest has an invalid shard list")
    drive_by_name: dict[str, Mapping[str, object]] = {}
    file_ids: set[str] = set()
    for item in drive_shards:
        assert isinstance(item, Mapping)
        filename = _require_string(item.get("filename"), name="Drive filename")
        file_id = _require_string(item.get("drive_file_id"), name="Drive file ID")
        if filename in drive_by_name or file_id in file_ids:
            raise ValueError("Drive manifest contains a duplicate shard or file ID")
        if item.get("remote_durable") is not True:
            raise ValueError(f"Drive shard is not remotely durable: {filename}")
        if item.get("configuration_hash") != configuration_hash:
            raise ValueError(f"Drive shard has a configuration mismatch: {filename}")
        if item.get("schema_hash") != schema_hash:
            raise ValueError(f"Drive shard has a schema mismatch: {filename}")
        drive_by_name[filename] = item
        file_ids.add(file_id)

    if set(drive_by_name) != set(local_by_name):
        raise ValueError("Drive and local manifests reference different shard filenames")
    for filename, local in local_by_name.items():
        remote = drive_by_name[filename]
        if remote.get("byte_size") != local.get("byte_size"):
            raise ValueError(f"Drive shard byte size mismatch: {filename}")
        if remote.get("local_sha256") != local.get("checksum"):
            raise ValueError(f"Drive shard checksum mismatch: {filename}")
    return run_id, len(drive_by_name)


def derive_plan(
    manifest: Mapping[str, object],
    *,
    manifest_path: Path | None = None,
    drive_manifest_path: Path | None = None,
) -> dict[str, object]:
    """Validate the fixed profile and return exact one-pass launch values."""

    _validate_identity(manifest)
    train = _split_summary(manifest, split="train")
    validation = _split_summary(manifest, split="validation")
    planned_updates = int(train["block_count"])
    if planned_updates < 20:
        raise ValueError("qualification dataset has fewer than 20 train blocks")
    if int(validation["block_count"]) == 0:
        raise ValueError("qualification dataset has no validation block")

    warmup_updates = max(16, math.ceil(planned_updates * 0.05))
    decay_updates = math.ceil(planned_updates * 0.20)
    if warmup_updates + decay_updates >= planned_updates:
        raise ValueError("qualification dataset is too small for the fixed WSD phases")
    stable_updates = planned_updates - warmup_updates - decay_updates

    full_update_tokens = CONTEXT_LENGTH * SEQUENCES_PER_BLOCK
    total_train_tokens = int(train["target_tokens"])
    warmup_tokens = warmup_updates * full_update_tokens
    tokens_before_decay = (planned_updates - decay_updates) * full_update_tokens
    if not 0 < warmup_tokens < tokens_before_decay < total_train_tokens:
        raise ValueError("manifest target tokens are incompatible with the derived WSD schedule")
    stable_tokens = tokens_before_decay - warmup_tokens
    decay_tokens = total_train_tokens - tokens_before_decay

    accepted = int(manifest["accepted_source_tokens"])
    validation_source = _require_integer(
        manifest.get("validation_source_tokens"),
        name="validation_source_tokens",
    )
    if validation_source > accepted:
        raise ValueError("validation source tokens exceed accepted source tokens")

    identity: dict[str, object] = {}
    if manifest_path is not None:
        identity["manifest_path"] = str(manifest_path)
        identity["manifest_sha256"] = _sha256(manifest_path)
    if drive_manifest_path is not None:
        drive_manifest = _read_object(drive_manifest_path, label="Drive manifest")
        drive_run_id, drive_shard_count = _verify_drive_manifest(
            drive_manifest,
            manifest=manifest,
            train=train,
            validation=validation,
        )
        identity["drive_manifest_path"] = str(drive_manifest_path)
        identity["drive_manifest_sha256"] = _sha256(drive_manifest_path)
        identity["drive_run_id"] = drive_run_id
        identity["drive_shard_count"] = drive_shard_count

    return {
        "version": 1,
        "qualification_profile": "20m-one-pass-v1",
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
        },
        "identity": identity,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Derive the exact 20M trainer plan from its verified manifest."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--drive-manifest", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest_path = args.dataset_dir / config.MANIFEST_FILENAME
    try:
        manifest = _read_object(manifest_path, label="qualification manifest")
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
            f"qualification report error: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
