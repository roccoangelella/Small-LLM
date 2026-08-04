"""Fail-closed token-by-token verification for the 20M qualification dataset.

The general schema-v2 verifier validates shard identities, sizes, checksums,
block continuity, source-token attribution, and Drive durability markers.  This
entry point adds the qualification-specific geometry checks and decodes every
stored uint16 token so ``--full-scan`` evidence is literal rather than sampled.
"""

from __future__ import annotations

import argparse
import json
import struct
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from dataset import config
from dataset.qualification_20m import (
    CONTEXT_LENGTH,
    MAXIMUM_SOURCE_TOKENS,
    MINIMUM_SOURCE_TOKENS,
    SEQUENCES_PER_BLOCK,
    TARGET_SHARD_BYTES,
    TARGET_SOURCE_TOKENS,
)
from dataset.src.storage import read_json
from dataset.src.verify import verify as verify_dataset

SCAN_CHUNK_BYTES = 8 * 1024 * 1024
_MAX_REPORTED_BAD_TOKENS = 10


def _integer(
    value: object,
    label: str,
    problems: list[str],
    *,
    minimum: int = 0,
) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        problems.append(f"{label} must be an integer >= {minimum}")
        return None
    return value


def _safe_shard_path(root: Path, filename: object, problems: list[str]) -> Path | None:
    if not isinstance(filename, str) or not filename:
        problems.append("shard filename must be a non-empty string")
        return None
    candidate = (root / filename).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        problems.append(f"shard path escapes dataset directory: {filename}")
        return None
    return candidate


def _scan_uint16_file(path: Path) -> tuple[int, int, list[str]]:
    """Decode every little-endian uint16 token without loading the shard at once."""

    token_count = 0
    out_of_range = 0
    details: list[str] = []
    byte_offset = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(SCAN_CHUNK_BYTES)
            if not chunk:
                break
            if len(chunk) % 2:
                details.append(f"odd byte count while scanning {path.name}")
                break
            for local_index, (value,) in enumerate(struct.iter_unpack("<H", chunk)):
                if not (config.TOKEN_MIN <= value <= config.TOKEN_MAX):
                    out_of_range += 1
                    if len(details) < _MAX_REPORTED_BAD_TOKENS:
                        details.append(
                            f"token at {path.name}:{byte_offset + local_index * 2}={value} "
                            f"is outside {config.TOKEN_MIN}..{config.TOKEN_MAX}"
                        )
            token_count += len(chunk) // 2
            byte_offset += len(chunk)
    return token_count, out_of_range, details


def verify_qualification_dataset(dataset_dir: Path) -> dict[str, Any]:
    """Return complete structural, identity, geometry, and token-scan evidence."""

    root = dataset_dir.resolve()
    problems: list[str] = []

    try:
        structural = verify_dataset(root, full_scan=False)
    except (FileNotFoundError, OSError, TypeError, ValueError) as error:
        return {
            "passed": False,
            "complete": False,
            "full_scan": True,
            "dataset_dir": str(root),
            "problems": [f"structural verifier failed: {error}"],
        }
    problems.extend(structural.problems)

    manifest_path = root / config.MANIFEST_FILENAME
    try:
        manifest = read_json(manifest_path)
    except (OSError, TypeError, ValueError) as error:
        problems.append(f"cannot read manifest.json: {error}")
        manifest = {}
    if not isinstance(manifest, dict):
        problems.append("manifest.json must contain an object")
        manifest = {}

    expected_manifest_values = {
        "schema_version": 2,
        "sequence_format": "context_plus_one",
        "context_length": CONTEXT_LENGTH,
        "stored_tokens_per_sequence": CONTEXT_LENGTH + 1,
        "sequences_per_block": SEQUENCES_PER_BLOCK,
        "target_shard_bytes": TARGET_SHARD_BYTES,
    }
    for key, expected in expected_manifest_values.items():
        if manifest.get(key) != expected:
            problems.append(f"manifest {key} must equal {expected!r}")

    accepted_source_tokens = _integer(
        manifest.get("accepted_source_tokens"),
        "manifest accepted_source_tokens",
        problems,
    )
    if accepted_source_tokens is not None and not (
        MINIMUM_SOURCE_TOKENS <= accepted_source_tokens <= MAXIMUM_SOURCE_TOKENS
    ):
        problems.append(
            "accepted source tokens are outside the fixed "
            f"{MINIMUM_SOURCE_TOKENS}..{MAXIMUM_SOURCE_TOKENS} range"
        )

    production = manifest.get("production")
    if not isinstance(production, Mapping):
        problems.append("manifest production must be an object")
        production = {}
    expected_production_values = {
        "target_source_tokens": TARGET_SOURCE_TOKENS,
        "minimum_source_tokens": MINIMUM_SOURCE_TOKENS,
        "maximum_source_tokens": MAXIMUM_SOURCE_TOKENS,
        "remote_required": True,
        "target_reached": True,
        "completion_reason": "target_reached",
    }
    for key, expected in expected_production_values.items():
        if production.get(key) != expected:
            problems.append(f"production {key} must equal {expected!r}")

    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        problems.append("manifest shards must be a non-empty list")
        shards = []

    scanned_tokens = 0
    split_stored_tokens = {"train": 0, "validation": 0}
    split_sequence_counts = {"train": 0, "validation": 0}
    per_cluster_source_tokens: defaultdict[int, int] = defaultdict(int)
    train_source_tokens = 0

    for index, raw_entry in enumerate(shards):
        if not isinstance(raw_entry, Mapping):
            problems.append(f"shard entry {index} must be an object")
            continue
        entry = dict(raw_entry)
        path = _safe_shard_path(root, entry.get("filename"), problems)
        if path is None or not path.is_file():
            if path is not None:
                problems.append(f"missing shard {path}")
            continue

        split = entry.get("split")
        if split not in split_stored_tokens:
            problems.append(f"invalid split for {path.name}: {split!r}")
            continue

        byte_size = _integer(entry.get("byte_size"), f"{path.name} byte_size", problems)
        sequence_count = _integer(
            entry.get("sequence_count"),
            f"{path.name} sequence_count",
            problems,
            minimum=1,
        )
        manifest_token_count = _integer(
            entry.get("token_count"),
            f"{path.name} token_count",
            problems,
            minimum=1,
        )
        first_block = _integer(
            entry.get("first_block_id"),
            f"{path.name} first_block_id",
            problems,
        )
        last_block = _integer(
            entry.get("last_block_id"),
            f"{path.name} last_block_id",
            problems,
        )

        actual_size = path.stat().st_size
        if byte_size is not None and byte_size != actual_size:
            problems.append(f"{path.name} byte size disagrees with manifest")
        if actual_size % 2:
            problems.append(f"{path.name} has an odd byte size")

        actual_token_count, bad_count, bad_details = _scan_uint16_file(path)
        scanned_tokens += actual_token_count
        problems.extend(bad_details)
        if bad_count:
            problems.append(f"{bad_count} tokens in {path.name} are outside the vocabulary range")

        if manifest_token_count is not None and manifest_token_count != actual_token_count:
            problems.append(f"{path.name} token_count disagrees with decoded tokens")
        if sequence_count is not None:
            expected_tokens = sequence_count * (CONTEXT_LENGTH + 1)
            if actual_token_count != expected_tokens:
                problems.append(
                    f"{path.name} has {actual_token_count} tokens; "
                    f"expected {expected_tokens} from sequence geometry"
                )
        if (
            sequence_count is not None
            and first_block is not None
            and last_block is not None
        ):
            if last_block < first_block:
                problems.append(f"{path.name} has a reversed block range")
            else:
                block_count = last_block - first_block + 1
                expected_blocks = (
                    sequence_count + SEQUENCES_PER_BLOCK - 1
                ) // SEQUENCES_PER_BLOCK
                if block_count != expected_blocks:
                    problems.append(
                        f"{path.name} block range contains {block_count} blocks; "
                        f"sequence geometry requires {expected_blocks}"
                    )

        split_stored_tokens[split] += actual_token_count
        if sequence_count is not None:
            split_sequence_counts[split] += sequence_count

        raw_cluster_counts = entry.get("shard_cluster_source_tokens")
        if not isinstance(raw_cluster_counts, Mapping):
            problems.append(f"{path.name} source-token attribution must be an object")
            continue
        shard_source_tokens = 0
        for raw_cluster, raw_count in raw_cluster_counts.items():
            try:
                cluster = int(raw_cluster)
            except (TypeError, ValueError):
                problems.append(f"{path.name} has invalid cluster key {raw_cluster!r}")
                continue
            count = _integer(
                raw_count,
                f"{path.name} cluster {cluster} source tokens",
                problems,
            )
            if cluster not in config.ACCEPTED_CLUSTER_IDS:
                problems.append(f"{path.name} attributes tokens to excluded cluster {cluster}")
            if count is not None:
                per_cluster_source_tokens[cluster] += count
                shard_source_tokens += count
        if split == "train":
            train_source_tokens += shard_source_tokens

    source_total = sum(per_cluster_source_tokens.values())
    if accepted_source_tokens is not None and source_total != accepted_source_tokens:
        problems.append(
            "aggregated per-cluster source tokens disagree with accepted_source_tokens"
        )

    scheduler = manifest.get("scheduler")
    if not isinstance(scheduler, Mapping):
        problems.append("manifest scheduler must be an object")
    else:
        scheduled_train_tokens = _integer(
            scheduler.get("total_emitted_source_tokens"),
            "scheduler total_emitted_source_tokens",
            problems,
        )
        if scheduled_train_tokens is not None and scheduled_train_tokens != train_source_tokens:
            problems.append(
                "aggregated train source tokens disagree with scheduler total"
            )

    if split_stored_tokens["train"] != structural.train_token_count:
        problems.append("decoded train-token total disagrees with structural verifier")
    if split_stored_tokens["validation"] != structural.validation_token_count:
        problems.append("decoded validation-token total disagrees with structural verifier")

    return {
        "passed": not problems,
        "complete": structural.complete,
        "full_scan": True,
        "dataset_dir": str(root),
        "shard_count": len(shards),
        "scanned_tokens": scanned_tokens,
        "train_stored_tokens": split_stored_tokens["train"],
        "validation_stored_tokens": split_stored_tokens["validation"],
        "train_sequence_count": split_sequence_counts["train"],
        "validation_sequence_count": split_sequence_counts["validation"],
        "accepted_source_tokens": accepted_source_tokens,
        "per_cluster_source_tokens": {
            str(cluster): per_cluster_source_tokens[cluster]
            for cluster in sorted(per_cluster_source_tokens)
        },
        "unavailable_counters": [
            "accepted_document_count",
            "inserted_eod_count",
        ],
        "problems": problems,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Token-by-token verifier for the fixed 20M qualification dataset."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = verify_qualification_dataset(args.dataset_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
