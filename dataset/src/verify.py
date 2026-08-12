"""Fail-closed verifier for the active schema-v2 sharded dataset format."""

from __future__ import annotations

import sys
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dataset import config

from .storage import read_json, sha256_file


@dataclass
class VerifyReport:
    passed: bool
    complete: bool
    output_dir: str
    train_token_count: int
    validation_token_count: int
    total_written_tokens: int
    accepted_source_tokens: int
    inserted_eod_count: int
    accepted_document_count: int
    per_cluster: dict[str, dict[str, Any]] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "complete": self.complete,
            "output_dir": self.output_dir,
            "train_token_count": self.train_token_count,
            "validation_token_count": self.validation_token_count,
            "total_written_tokens": self.total_written_tokens,
            "accepted_source_tokens": self.accepted_source_tokens,
            "inserted_eod_count": self.inserted_eod_count,
            "accepted_document_count": self.accepted_document_count,
            "per_cluster": self.per_cluster,
            "problems": self.problems,
        }


def verify(output_dir: Path, *, full_scan: bool = False) -> VerifyReport:
    """Validate the active context-plus-one shard manifest and local files."""

    output_dir = output_dir.resolve()
    manifest_path = output_dir / config.MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest artefact: {manifest_path}")
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("manifest.json must contain a JSON object")
    if manifest.get("sequence_format") != "context_plus_one":
        raise ValueError(
            "unsupported legacy dataset format; active verification requires schema-v2 context_plus_one shards"
        )
    return _verify_stream_cache(output_dir, manifest, full_scan=full_scan)


def _scan_uint16_ranges(path: Path, *, chunk_bytes: int = 16 * 1024 * 1024) -> str | None:
    """Return a problem if any stored uint16 token lies outside the GPT-2 vocabulary."""

    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            if len(block) % 2:
                return f"odd byte size for {path.name}"
            values = array("H")
            values.frombytes(block)
            if sys.byteorder != "little":
                values.byteswap()
            invalid = next((token for token in values if token >= config.VOCAB_SIZE), None)
            if invalid is not None:
                return f"token id {invalid} outside vocabulary in {path.name}"
    return None


def _verify_stream_cache(
    output_dir: Path,
    manifest: dict[str, Any],
    *,
    full_scan: bool,
) -> VerifyReport:
    problems: list[str] = []

    if manifest.get("schema_version") != 2:
        problems.append("unsupported streaming-cache schema")
    context_length = manifest.get("context_length")
    stored_tokens = manifest.get("stored_tokens_per_sequence")
    if not isinstance(context_length, int) or context_length <= 0:
        problems.append("invalid context length")
    elif stored_tokens != context_length + 1:
        problems.append("context-plus-one geometry is inconsistent")
    shards = manifest.get("shards")
    if not isinstance(shards, list):
        problems.append("streaming manifest shards must be a list")
        shards = []

    blocks_by_split: dict[str, list[int]] = {"train": [], "validation": []}
    source_total = 0
    train_source_total = 0
    train_tokens = 0
    validation_tokens = 0
    per_cluster_total: dict[str, int] = {}

    for entry in shards:
        if not isinstance(entry, dict):
            problems.append("invalid shard entry")
            continue
        filename = entry.get("filename")
        if not isinstance(filename, str) or not filename:
            problems.append("shard entry has invalid filename")
            continue
        path = output_dir / filename
        if not path.is_file():
            problems.append(f"missing local shard {path}")
            continue

        byte_size = path.stat().st_size
        if byte_size != entry.get("byte_size"):
            problems.append(f"size mismatch for {path.name}")
        if byte_size % 2:
            problems.append(f"odd byte size for {path.name}")
        if sha256_file(path) != entry.get("checksum"):
            problems.append(f"checksum mismatch for {path.name}")
        if full_scan:
            range_problem = _scan_uint16_ranges(path)
            if range_problem is not None:
                problems.append(range_problem)

        split = entry.get("split")
        if split not in blocks_by_split:
            problems.append(f"invalid split for {path.name}")
            continue
        first = entry.get("first_block_id")
        last = entry.get("last_block_id")
        if not isinstance(first, int) or not isinstance(last, int) or last < first:
            problems.append(f"invalid block range for {path.name}")
        else:
            blocks_by_split[split].extend(range(first, last + 1))

        per_cluster = entry.get("shard_cluster_source_tokens", {})
        shard_source_total = 0
        if not isinstance(per_cluster, dict):
            problems.append(f"invalid source-token attribution for {path.name}")
        else:
            try:
                for cluster, value in per_cluster.items():
                    tokens = int(value)
                    if tokens < 0:
                        raise ValueError
                    shard_source_total += tokens
                    key = str(cluster)
                    per_cluster_total[key] = per_cluster_total.get(key, 0) + tokens
            except (TypeError, ValueError):
                problems.append(f"invalid source-token attribution for {path.name}")
        source_total += shard_source_total

        if split == "train":
            train_tokens += byte_size // 2
            train_source_total += shard_source_total
        else:
            validation_tokens += byte_size // 2

    for split, blocks in blocks_by_split.items():
        if blocks and (
            len(blocks) != len(set(blocks))
            or sorted(blocks) != list(range(min(blocks), max(blocks) + 1))
        ):
            problems.append(f"duplicate or gapped {split} block ranges")

    if any(output_dir.rglob("*.tmp")) or any(output_dir.rglob("*.part")):
        problems.append("incomplete .tmp or .part file is present")

    accepted_source_tokens = manifest.get("accepted_source_tokens")
    if not isinstance(accepted_source_tokens, int) or accepted_source_tokens < 0:
        problems.append("invalid accepted_source_tokens")
        accepted_source_tokens = source_total
    elif source_total != accepted_source_tokens:
        problems.append("per-shard source-token total disagrees with accepted source tokens")

    scheduler = manifest.get("scheduler", {})
    if isinstance(scheduler, dict):
        emitted = scheduler.get("total_emitted_source_tokens")
        if not isinstance(emitted, int) or train_source_total != emitted:
            problems.append("train-shard source-token total disagrees with scheduler")

    # Historical filename retained by current producer/checkpoint compatibility.
    durability_manifest_path = output_dir / "drive_manifest.json"
    if durability_manifest_path.exists():
        durability = read_json(durability_manifest_path)
        remote_shards = durability.get("shards", []) if isinstance(durability, dict) else []
        if not isinstance(durability, dict) or not isinstance(remote_shards, list):
            problems.append("legacy durability manifest is invalid")
        elif any(
            not entry.get("remote_durable")
            for entry in remote_shards
            if isinstance(entry, dict)
        ):
            problems.append("legacy durability manifest contains unverified shard")

    complete = manifest.get("complete", True) is True
    if not complete:
        problems.append("manifest is not complete")

    return VerifyReport(
        passed=not problems,
        complete=complete,
        output_dir=str(output_dir),
        train_token_count=train_tokens,
        validation_token_count=validation_tokens,
        total_written_tokens=train_tokens + validation_tokens,
        accepted_source_tokens=int(accepted_source_tokens),
        inserted_eod_count=int(manifest.get("inserted_eod_count", 0) or 0),
        accepted_document_count=int(manifest.get("accepted_document_count", 0) or 0),
        per_cluster={
            cluster: {"source_tokens": tokens}
            for cluster, tokens in sorted(per_cluster_total.items(), key=lambda item: int(item[0]))
        },
        problems=problems,
    )
