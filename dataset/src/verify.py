"""Memory-efficient verifier for a built token corpus.

The ordinary verifier streams SHA-256 over each artefact and samples tokens via
``mmap``.  It never loads either corpus file into RAM.  ``--full-scan`` checks
every token and is intended for bounded smoke corpora, not the 90B-token build.
"""

from __future__ import annotations

import hashlib
import logging
import mmap
import random
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dataset import config

from .checkpoint import Progress, run_signature_hash
from .manifest import sha256_file
from .storage import read_json
from .workplan import load_work_plan


LOGGER = logging.getLogger(__name__)
SAMPLE_POSITIONS = 4096


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
    """Validate manifest, checkpoint, work plan, hashes, sizes, and token ranges."""

    output_dir = output_dir.resolve()
    streaming_manifest = output_dir / config.MANIFEST_FILENAME
    if streaming_manifest.is_file():
        candidate = read_json(streaming_manifest)
        if isinstance(candidate, dict) and candidate.get("sequence_format") == "context_plus_one":
            return _verify_stream_cache(output_dir, candidate)
    paths = {
        "train": output_dir / config.TRAIN_FILENAME,
        "validation": output_dir / config.VALIDATION_FILENAME,
        "manifest": output_dir / config.MANIFEST_FILENAME,
        "progress": output_dir / config.PROGRESS_FILENAME,
        "work_plan": output_dir / config.WORK_PLAN_FILENAME,
    }
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label} artefact: {path}")

    manifest = read_json(paths["manifest"])
    if not isinstance(manifest, dict):
        raise ValueError("manifest.json must contain a JSON object")
    if manifest.get("schema_version") != config.MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported manifest schema version {manifest.get('schema_version')!r}; "
            f"expected {config.MANIFEST_SCHEMA_VERSION}"
        )

    problems: list[str] = []
    _validate_manifest_policy(manifest, problems)
    counts = _mapping(manifest, "counts", problems)
    hashes = _mapping(manifest, "hashes", problems)

    try:
        progress = Progress.load(paths["progress"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid progress.json: {error}") from error
    try:
        plan = load_work_plan(paths["work_plan"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid work_plan.json: {error}") from error

    train_size = paths["train"].stat().st_size
    validation_size = paths["validation"].stat().st_size
    _validate_sizes(
        train_size=train_size,
        validation_size=validation_size,
        counts=counts,
        progress=progress,
        problems=problems,
    )

    # Stream every hash without retaining file contents.
    actual_hashes = {
        "train_sha256": sha256_file(paths["train"]),
        "validation_sha256": sha256_file(paths["validation"]),
        "work_plan_sha256": sha256_file(paths["work_plan"]),
    }
    for key, actual in actual_hashes.items():
        if hashes.get(key) != actual:
            problems.append(f"{key} mismatch")

    if manifest.get("work_plan_hash") != plan.hash:
        problems.append("manifest work_plan_hash does not match work_plan.json")
    if progress.work_plan_hash != plan.hash:
        problems.append("progress work_plan_hash does not match work_plan.json")
    if plan.dataset != config.DATASET_REPOSITORY:
        problems.append("work plan dataset does not match the frozen repository")
    if plan.revision != config.DATASET_REVISION:
        problems.append("work plan revision does not match the frozen revision")
    if plan.selection_seed != config.SELECTION_SEED:
        problems.append("work plan selection seed does not match frozen policy")
    if plan.source_glob != config.SOURCE_DATA_GLOB:
        problems.append("work plan source glob does not match frozen policy")
    if manifest.get("region_bytes") != plan.region_bytes:
        problems.append("manifest region_bytes does not match work_plan.json")
    expected_source_files = [
        {"path": source.path, "size": source.size} for source in plan.source_files
    ]
    if manifest.get("source_files") != expected_source_files:
        problems.append("manifest source_files do not match work_plan.json")

    # Both files are opened and memory-mapped independently.  Empty files are
    # still checked for readability, but POSIX cannot mmap a zero-length file.
    for path, size in (
        (paths["train"], train_size),
        (paths["validation"], validation_size),
    ):
        problems.extend(_check_token_ranges(path, size, full_scan=full_scan))

    _validate_progress_and_counts(
        manifest=manifest,
        counts=counts,
        progress=progress,
        train_size=train_size,
        validation_size=validation_size,
        problems=problems,
    )

    return VerifyReport(
        passed=not problems,
        complete=progress.complete,
        output_dir=str(output_dir),
        train_token_count=train_size // 2,
        validation_token_count=validation_size // 2,
        total_written_tokens=(train_size + validation_size) // 2,
        accepted_source_tokens=progress.accepted_source_tokens,
        inserted_eod_count=progress.inserted_eod_count,
        accepted_document_count=progress.accepted_document_count,
        per_cluster=counts.get("per_cluster", {}),
        problems=problems,
    )


def _verify_stream_cache(output_dir: Path, manifest: dict[str, Any]) -> VerifyReport:
    """Validate schema-v2 cache shards without assuming legacy train.bin files."""
    problems: list[str] = []
    if manifest.get("schema_version") != 2:
        problems.append("unsupported streaming-cache schema")
    if manifest.get("stored_tokens_per_sequence") != int(manifest.get("context_length", -1)) + 1:
        problems.append("context-plus-one geometry is inconsistent")
    shards = manifest.get("shards")
    if not isinstance(shards, list):
        problems.append("streaming manifest shards must be a list")
        shards = []
    blocks_by_split: dict[str, list[int]] = {"train": [], "validation": []}
    source_total = train_source_total = 0
    train_tokens = validation_tokens = 0
    for entry in shards:
        if not isinstance(entry, dict):
            problems.append("invalid shard entry")
            continue
        path = output_dir / str(entry.get("filename", ""))
        if not path.is_file():
            problems.append(f"missing local shard {path}")
            continue
        if path.stat().st_size != entry.get("byte_size"):
            problems.append(f"size mismatch for {path.name}")
        if sha256_file(path) != entry.get("checksum"):
            problems.append(f"checksum mismatch for {path.name}")
        if path.stat().st_size % 2:
            problems.append(f"odd byte size for {path.name}")
        split = entry.get("split")
        if split not in blocks_by_split:
            problems.append(f"invalid split for {path.name}")
            continue
        first, last = entry.get("first_block_id"), entry.get("last_block_id")
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
                shard_source_total = sum(int(value) for value in per_cluster.values())
            except (TypeError, ValueError):
                problems.append(f"invalid source-token attribution for {path.name}")
            source_total += shard_source_total
        if split == "train":
            train_tokens += path.stat().st_size // 2
            train_source_total += shard_source_total
        else:
            validation_tokens += path.stat().st_size // 2
    for split, blocks in blocks_by_split.items():
        if blocks and (
            len(blocks) != len(set(blocks))
            or sorted(blocks) != list(range(min(blocks), max(blocks) + 1))
        ):
            problems.append(f"duplicate or gapped {split} block ranges")
    if any(output_dir.rglob("*.tmp")) or any(output_dir.rglob("*.part")):
        problems.append("incomplete .tmp or .part file is present")
    scheduler = manifest.get("scheduler", {})
    if source_total != int(manifest.get("accepted_source_tokens", -1)):
        problems.append("per-shard source-token total disagrees with accepted source tokens")
    if isinstance(scheduler, dict) and train_source_total != int(scheduler.get("total_emitted_source_tokens", -1)):
        problems.append("train-shard source-token total disagrees with scheduler")
    drive_manifest_path = output_dir / "drive_manifest.json"
    if drive_manifest_path.exists():
        drive = read_json(drive_manifest_path)
        if not isinstance(drive, dict) or any(not entry.get("remote_durable") for entry in drive.get("shards", []) if isinstance(entry, dict)):
            problems.append("Drive manifest contains unverified shard")
    return VerifyReport(not problems, True, str(output_dir), train_tokens, validation_tokens,
                        train_tokens + validation_tokens, source_total, 0, 0, problems=problems)


def _validate_manifest_policy(manifest: dict[str, Any], problems: list[str]) -> None:
    required = (
        "complete",
        "dataset_repository",
        "source_revision",
        "source_files",
        "work_plan_hash",
        "selection_seed",
        "accepted_cluster_ids",
        "excluded_cluster_ids",
        "validation_split_rule",
        "tokenizer",
        "eod_token_id",
        "binary_format",
        "counts",
        "targets",
        "timestamps",
        "hashes",
        "license",
        "attribution",
        "known_limitation",
    )
    for key in required:
        if key not in manifest:
            problems.append(f"manifest is missing {key!r}")

    expected_pairs = (
        ("dataset_repository", config.DATASET_REPOSITORY),
        ("source_revision", config.DATASET_REVISION),
        ("source_glob", config.SOURCE_DATA_GLOB),
        ("selection_seed", config.SELECTION_SEED),
        ("eod_token_id", config.EOD_TOKEN_ID),
    )
    for key, expected in expected_pairs:
        if manifest.get(key) != expected:
            problems.append(f"manifest {key} does not match frozen policy")
    if manifest.get("accepted_cluster_ids") != sorted(config.ACCEPTED_CLUSTER_IDS):
        problems.append("manifest accepted_cluster_ids do not match frozen policy")
    if manifest.get("excluded_cluster_ids") != sorted(config.EXCLUDED_CLUSTER_IDS):
        problems.append("manifest excluded_cluster_ids do not match frozen policy")

    tokenizer = manifest.get("tokenizer")
    if not isinstance(tokenizer, dict):
        problems.append("manifest tokenizer must be an object")
    else:
        if tokenizer.get("id") != config.TOKENIZER_ID:
            problems.append("manifest tokenizer ID does not match frozen policy")
        if tokenizer.get("vocab_size") != config.VOCAB_SIZE:
            problems.append("manifest tokenizer vocabulary size does not match policy")

    binary = manifest.get("binary_format")
    if not isinstance(binary, dict):
        problems.append("manifest binary_format must be an object")
    else:
        expected_binary = {
            "int_type": config.INT_TYPE,
            "byte_order": config.BYTE_ORDER,
            "bytes_per_token": 2,
            "header": "none",
            "framing": "none",
            "compression": "none",
        }
        for key, expected in expected_binary.items():
            if binary.get(key) != expected:
                problems.append(f"manifest binary_format.{key} != {expected!r}")

    split = manifest.get("validation_split_rule")
    if not isinstance(split, dict):
        problems.append("manifest validation_split_rule must be an object")
    else:
        if split.get("probability") != config.VALIDATION_PROBABILITY:
            problems.append("manifest validation probability does not match frozen policy")
        if split.get("hash_version") != config.SPLIT_HASH_VERSION:
            problems.append("manifest split hash version does not match frozen policy")


def _mapping(
    payload: dict[str, Any], key: str, problems: list[str]
) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        problems.append(f"manifest {key} must be an object")
        return {}
    return value


def _integer(mapping: dict[str, Any], key: str, problems: list[str]) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        problems.append(f"manifest count {key} must be an integer")
        return -1
    return value


def _counter_integer(
    mapping: dict[str, Any],
    key: str,
    label: str,
    problems: list[str],
) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        problems.append(f"{label}.{key} must be a non-negative integer")
        return 0
    return value


def _validate_sizes(
    *,
    train_size: int,
    validation_size: int,
    counts: dict[str, Any],
    progress: Progress,
    problems: list[str],
) -> None:
    if train_size % 2:
        problems.append(f"{config.TRAIN_FILENAME} size {train_size} is not divisible by two")
    if validation_size % 2:
        problems.append(
            f"{config.VALIDATION_FILENAME} size {validation_size} is not divisible by two"
        )

    expected = {
        "train_file_size": train_size,
        "validation_file_size": validation_size,
        "train_token_count": train_size // 2,
        "validation_token_count": validation_size // 2,
    }
    for key, actual in expected.items():
        if _integer(counts, key, problems) != actual:
            problems.append(f"manifest {key} does not match the binary files")

    if progress.confirmed_train_byte_size != train_size:
        problems.append("progress confirmed_train_byte_size does not match train.bin")
    if progress.confirmed_validation_byte_size != validation_size:
        problems.append(
            "progress confirmed_validation_byte_size does not match validation.bin"
        )
    if progress.train_written_tokens * 2 != train_size:
        problems.append("progress train_written_tokens does not match train.bin")
    if progress.validation_written_tokens * 2 != validation_size:
        problems.append("progress validation_written_tokens does not match validation.bin")


def _validate_progress_and_counts(
    *,
    manifest: dict[str, Any],
    counts: dict[str, Any],
    progress: Progress,
    train_size: int,
    validation_size: int,
    problems: list[str],
) -> None:
    manifest_complete = manifest.get("complete")
    if not isinstance(manifest_complete, bool):
        problems.append("manifest complete must be a boolean")
        manifest_complete = False
    if manifest_complete != progress.complete:
        problems.append(
            f"manifest.complete={manifest_complete} but progress.complete={progress.complete}"
        )

    direct_counts = {
        "train_source_token_count": progress.train_source_tokens,
        "validation_source_token_count": progress.validation_source_tokens,
        "train_inserted_eod_count": progress.train_inserted_eod_count,
        "validation_inserted_eod_count": progress.validation_inserted_eod_count,
        "train_document_count": progress.train_document_count,
        "validation_document_count": progress.validation_document_count,
        "total_written_tokens": (train_size + validation_size) // 2,
        "accepted_source_tokens": progress.accepted_source_tokens,
        "inserted_eod_count": progress.inserted_eod_count,
        "accepted_document_count": progress.accepted_document_count,
        "inspected_document_count": progress.inspected_document_count,
        "source_bytes_processed": progress.source_bytes_processed,
    }
    for key, expected in direct_counts.items():
        if _integer(counts, key, problems) != expected:
            problems.append(f"manifest {key} does not match progress.json")

    if (
        progress.train_written_tokens + progress.validation_written_tokens
        != progress.accepted_source_tokens + progress.inserted_eod_count
    ):
        problems.append("written tokens do not equal source tokens plus inserted EODs")
    if (
        progress.train_source_tokens + progress.validation_source_tokens
        != progress.accepted_source_tokens
    ):
        problems.append("split source-token counts do not sum to accepted source tokens")
    if (
        progress.train_document_count + progress.validation_document_count
        != progress.accepted_document_count
    ):
        problems.append("split document counts do not sum to accepted documents")
    if (
        progress.train_inserted_eod_count + progress.validation_inserted_eod_count
        != progress.inserted_eod_count
    ):
        problems.append("split EOD counts do not sum to inserted EOD count")
    if progress.accepted_document_count > progress.inspected_document_count:
        problems.append("accepted document count exceeds inspected document count")

    per_cluster = counts.get("per_cluster")
    if not isinstance(per_cluster, dict):
        problems.append("manifest per_cluster must be an object")
    else:
        source_sum = 0
        document_sum = 0
        for cluster_id in range(1, 21):
            counters = per_cluster.get(str(cluster_id))
            if not isinstance(counters, dict):
                problems.append(f"manifest per_cluster is missing cluster {cluster_id}")
                continue
            progress_counters = progress.per_cluster.get(str(cluster_id), {})
            for key in ("documents", "source_tokens", "written_tokens", "inserted_eods"):
                value = _counter_integer(
                    counters, key, f"per_cluster.{cluster_id}", problems
                )
                if value != int(progress_counters.get(key, 0)):
                    problems.append(
                        f"manifest per_cluster.{cluster_id}.{key} "
                        "does not match progress.json"
                    )
            source_sum += _counter_integer(
                counters,
                "source_tokens",
                f"per_cluster.{cluster_id}",
                problems,
            )
            document_sum += _counter_integer(
                counters,
                "documents",
                f"per_cluster.{cluster_id}",
                problems,
            )
        if source_sum != progress.accepted_source_tokens:
            problems.append("per-cluster source tokens do not sum to accepted source tokens")
        if document_sum != progress.accepted_document_count:
            problems.append("per-cluster documents do not sum to accepted documents")
        cluster_11 = per_cluster.get("11", {})
        if (
            isinstance(cluster_11, dict)
            and (
                _counter_integer(cluster_11, "documents", "per_cluster.11", problems)
                or _counter_integer(
                    cluster_11, "source_tokens", "per_cluster.11", problems
                )
            )
        ):
            problems.append("excluded cluster 11 has accepted documents or tokens")

    if counts.get("structural_rejections") != progress.structural_rejections:
        problems.append("manifest structural_rejections do not match progress.json")
    if counts.get("cluster_exclusions") != progress.cluster_exclusions:
        problems.append("manifest cluster_exclusions do not match progress.json")

    if progress.run_config_hash != run_signature_hash(progress.run_config):
        problems.append("progress configuration hash is invalid")
    if progress.dataset != config.DATASET_REPOSITORY:
        problems.append("progress dataset does not match frozen policy")
    if progress.revision != config.DATASET_REVISION:
        problems.append("progress revision does not match frozen policy")

    targets = manifest.get("targets")
    if not isinstance(targets, dict):
        problems.append("manifest targets must be an object")
        targets = {}
    target = targets.get("target_accepted_source_tokens")
    minimum = targets.get("minimum_accepted_source_tokens")
    maximum = targets.get("maximum_accepted_source_tokens")
    expected_targets = {
        "target_accepted_source_tokens": progress.run_config.get(
            "target_accepted_source_tokens"
        ),
        "minimum_accepted_source_tokens": progress.run_config.get(
            "minimum_accepted_source_tokens"
        ),
        "maximum_accepted_source_tokens": progress.run_config.get(
            "maximum_accepted_source_tokens"
        ),
    }
    for key, expected in expected_targets.items():
        if targets.get(key) != expected:
            problems.append(f"manifest target {key} does not match progress.json")
    if progress.complete:
        if not isinstance(target, int) or progress.accepted_source_tokens < target:
            problems.append("complete corpus has not reached its accepted-source-token target")
        if not isinstance(minimum, int) or progress.accepted_source_tokens < minimum:
            problems.append("complete corpus is below its minimum acceptable size")
        if not isinstance(maximum, int) or progress.accepted_source_tokens > maximum:
            problems.append("complete corpus exceeds its accepted-source-token maximum")

    timestamps = manifest.get("timestamps")
    if not isinstance(timestamps, dict):
        problems.append("manifest timestamps must be an object")
    elif progress.complete and not timestamps.get("completion"):
        problems.append("complete corpus has no completion timestamp")
    elif not progress.complete and timestamps.get("completion") is not None:
        problems.append("incomplete corpus has a misleading completion timestamp")


def _check_token_ranges(path: Path, size: int, *, full_scan: bool) -> list[str]:
    """Memory-map and confirm tokens are explicit little-endian uint16 values."""

    problems: list[str] = []
    with path.open("rb") as handle:
        if size == 0:
            return problems
        usable_size = size - (size % 2)
        if usable_size == 0:
            return problems
        checked = 0
        out_of_range = 0
        unpack = struct.Struct("<H")
        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapping:
            if full_scan or usable_size <= 4 * 1024 * 1024:
                offsets = range(0, usable_size, 2)
            else:
                seed = hashlib.sha256(
                    f"{path.name}:{size}".encode("utf-8")
                ).digest()
                rng = random.Random(seed)
                offsets = (
                    rng.randrange(usable_size // 2) * 2
                    for _ in range(SAMPLE_POSITIONS)
                )
            for offset in offsets:
                value = unpack.unpack_from(mapping, offset)[0]
                if not (config.TOKEN_MIN <= value <= config.TOKEN_MAX):
                    out_of_range += 1
                    if out_of_range <= 10:
                        problems.append(
                            f"token at {path.name}:{offset}={value} is outside "
                            f"{config.TOKEN_MIN}..{config.TOKEN_MAX}"
                        )
                checked += 1
        if out_of_range:
            problems.append(
                f"{out_of_range} checked tokens in {path.name} were out of range"
            )
        LOGGER.debug("verified %d token positions in %s", checked, path.name)
    return problems
