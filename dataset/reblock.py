"""Byte-preserving schema-v2 reblocking for the Modal training geometry.

This tool changes only physical shard/block grouping.  Stored context+1 uint16
sequences are copied in exact split order; no source data is downloaded and no
tokenization, packing, split assignment, or mixture scheduling is repeated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath, PureWindowsPath

from dataset import config
from dataset.qualification import derive_plan, get_profile, profile_payload
from dataset.src.storage import write_json_atomic
from dataset.src.verify import verify

SOURCE_PROFILE = "20m-2b"
DEFAULT_TARGET_PROFILE = "modal-2b-b64"
COPY_CHUNK_BYTES = 16 * 1024 * 1024


def _read_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(payload)


def _stable_hash(value: Mapping[str, object]) -> str:
    encoded = json.dumps(dict(value), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_relative(value: object) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise ValueError(f"unsafe shard filename: {value!r}")
    posix, windows = PurePosixPath(value), PureWindowsPath(value)
    if posix.is_absolute() or windows.is_absolute() or windows.drive:
        raise ValueError(f"unsafe shard filename: {value!r}")
    if any(part in {"", ".", ".."} for part in value.split("/")):
        raise ValueError(f"unsafe shard filename: {value!r}")
    return Path(*value.split("/"))


def _integer(entry: Mapping[str, object], key: str, *, minimum: int = 0) -> int:
    value = entry.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"invalid shard {key}: {value!r}")
    return value


def _cluster_counts(entry: Mapping[str, object]) -> dict[int, int]:
    raw = entry.get("shard_cluster_source_tokens")
    if not isinstance(raw, Mapping):
        raise ValueError("source shard has no source-token attribution")
    counts: dict[int, int] = {}
    for raw_cluster, raw_value in raw.items():
        cluster = int(raw_cluster)
        value = int(raw_value)
        if value < 0:
            raise ValueError("source-token attribution cannot be negative")
        counts[cluster] = counts.get(cluster, 0) + value
    return counts


def _split_shards(manifest: Mapping[str, object], split: str) -> list[dict[str, object]]:
    raw = manifest.get("shards")
    if not isinstance(raw, list):
        raise ValueError("source manifest has no shard list")
    rows = [dict(item) for item in raw if isinstance(item, Mapping) and item.get("split") == split]
    rows.sort(key=lambda item: _integer(item, "first_block_id"))
    if not rows:
        raise ValueError(f"source manifest has no {split} shards")
    expected = 0
    for row in rows:
        first = _integer(row, "first_block_id")
        last = _integer(row, "last_block_id")
        if first != expected or last < first:
            raise ValueError(f"source {split} shard block ranges are not contiguous")
        expected = last + 1
    return rows


def _group_shards(
    shards: Sequence[Mapping[str, object]],
    *,
    sequences_per_block: int,
    target_shard_bytes: int,
) -> list[list[Mapping[str, object]]]:
    groups: list[list[Mapping[str, object]]] = []
    current: list[Mapping[str, object]] = []
    current_sequences = 0
    current_bytes = 0
    for index, shard in enumerate(shards):
        current.append(shard)
        current_sequences += _integer(shard, "sequence_count", minimum=1)
        current_bytes += _integer(shard, "byte_size", minimum=1)
        is_last = index == len(shards) - 1
        next_bytes = 0 if is_last else _integer(shards[index + 1], "byte_size", minimum=1)
        aligned = current_sequences % sequences_per_block == 0
        target_boundary = current_bytes >= target_shard_bytes or current_bytes + next_bytes > target_shard_bytes
        if is_last or (aligned and target_boundary):
            groups.append(current)
            current = []
            current_sequences = 0
            current_bytes = 0
    if current:
        raise RuntimeError("internal reblock grouping left an unfinished shard group")
    for group in groups[:-1]:
        sequences = sum(_integer(item, "sequence_count", minimum=1) for item in group)
        if sequences % sequences_per_block:
            raise RuntimeError("non-final reblocked shard would end inside a prepared block")
    return groups


def _copy_group(
    *,
    source_root: Path,
    staging_root: Path,
    split: str,
    output_index: int,
    group: Sequence[Mapping[str, object]],
    block_cursor: int,
    sequences_per_block: int,
    context_length: int,
    cumulative_counts: dict[int, int],
    stream_digest: hashlib._Hash,
) -> tuple[dict[str, object], int]:
    relative = Path(split) / f"{split}-{output_index:06d}.bin"
    destination = staging_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists() or destination.exists():
        raise FileExistsError(f"reblock destination already exists: {destination}")

    digest = hashlib.sha256()
    sequence_count = 0
    group_counts: dict[int, int] = {}
    with temporary.open("xb") as output:
        for source_shard in group:
            sequence_count += _integer(source_shard, "sequence_count", minimum=1)
            for cluster, count in _cluster_counts(source_shard).items():
                group_counts[cluster] = group_counts.get(cluster, 0) + count
            source_path = source_root / _safe_relative(source_shard.get("filename"))
            with source_path.open("rb") as input_handle:
                while chunk := input_handle.read(COPY_CHUNK_BYTES):
                    output.write(chunk)
                    digest.update(chunk)
                    stream_digest.update(chunk)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, destination)

    token_count = sequence_count * (context_length + 1)
    byte_size = token_count * 2
    actual_size = destination.stat().st_size
    if actual_size != byte_size:
        raise RuntimeError(
            f"reblocked shard byte size mismatch: expected {byte_size}, got {actual_size}"
        )
    blocks = math.ceil(sequence_count / sequences_per_block)
    first_block_id = block_cursor
    last_block_id = first_block_id + blocks - 1
    for cluster, count in group_counts.items():
        cumulative_counts[cluster] = cumulative_counts.get(cluster, 0) + count
    row = {
        "filename": relative.as_posix(),
        "split": split,
        "byte_size": byte_size,
        "token_count": token_count,
        "sequence_count": sequence_count,
        "checksum": digest.hexdigest(),
        "first_block_id": first_block_id,
        "last_block_id": last_block_id,
        "context_length": context_length,
        "int_type": config.INT_TYPE,
        "byte_order": config.BYTE_ORDER,
        "cumulative_cluster_source_tokens": {
            str(cluster): count for cluster, count in sorted(cumulative_counts.items())
        },
        "shard_cluster_source_tokens": {
            str(cluster): count for cluster, count in sorted(group_counts.items())
        },
    }
    return row, last_block_id + 1


def _reblock_split(
    *,
    source_root: Path,
    staging_root: Path,
    manifest: Mapping[str, object],
    split: str,
    sequences_per_block: int,
    target_shard_bytes: int,
    context_length: int,
) -> tuple[list[dict[str, object]], str, int]:
    source_shards = _split_shards(manifest, split)
    groups = _group_shards(
        source_shards,
        sequences_per_block=sequences_per_block,
        target_shard_bytes=target_shard_bytes,
    )
    output: list[dict[str, object]] = []
    block_cursor = 0
    cumulative_counts: dict[int, int] = {}
    stream_digest = hashlib.sha256()
    for index, group in enumerate(groups):
        row, block_cursor = _copy_group(
            source_root=source_root,
            staging_root=staging_root,
            split=split,
            output_index=index,
            group=group,
            block_cursor=block_cursor,
            sequences_per_block=sequences_per_block,
            context_length=context_length,
            cumulative_counts=cumulative_counts,
            stream_digest=stream_digest,
        )
        if index < len(groups) - 1 and int(row["sequence_count"]) % sequences_per_block:
            raise RuntimeError("only the final split shard may contain a partial target block")
        output.append(row)
    return output, stream_digest.hexdigest(), block_cursor


def reblock_dataset(
    source_dir: Path,
    output_dir: Path,
    *,
    target_profile_key: str = DEFAULT_TARGET_PROFILE,
) -> dict[str, object]:
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    if source_dir == output_dir:
        raise ValueError("source and output dataset directories must differ")
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    source_manifest_path = source_dir / config.MANIFEST_FILENAME
    source_manifest = _read_object(source_manifest_path)

    report = verify(source_dir, full_scan=False)
    if not report.passed:
        raise RuntimeError("source dataset verification failed: " + "; ".join(report.problems))
    source_plan = derive_plan(
        source_manifest,
        profile=SOURCE_PROFILE,
        manifest_path=source_manifest_path,
    )
    target = get_profile(target_profile_key)
    if target.run_id is None:
        raise ValueError("target reblock profile must have a durable run ID")
    if target.context_length != int(source_plan["context_length"]):
        raise ValueError("target reblock profile changes context length")
    if target.sequences_per_block <= int(source_plan["sequences_per_block"]):
        raise ValueError("target reblock profile must increase prepared-block size")

    staging = output_dir.with_name(output_dir.name + ".reblock-tmp")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        train_rows, train_stream_sha, train_blocks = _reblock_split(
            source_root=source_dir,
            staging_root=staging,
            manifest=source_manifest,
            split="train",
            sequences_per_block=target.sequences_per_block,
            target_shard_bytes=target.target_shard_bytes,
            context_length=target.context_length,
        )
        validation_rows, validation_stream_sha, validation_blocks = _reblock_split(
            source_root=source_dir,
            staging_root=staging,
            manifest=source_manifest,
            split="validation",
            sequences_per_block=target.sequences_per_block,
            target_shard_bytes=target.target_shard_bytes,
            context_length=target.context_length,
        )

        source_manifest_sha = hashlib.sha256(source_manifest_path.read_bytes()).hexdigest()
        schema_hash = _stable_hash(
            {
                "stream_cache_schema_version": 2,
                "sequence_format": "context_plus_one",
                "context_length": target.context_length,
                "stored_sequence_tokens": target.context_length + 1,
                "sequences_per_block": target.sequences_per_block,
                "int_type": config.INT_TYPE,
                "byte_order": config.BYTE_ORDER,
                "eod_token_id": config.EOD_TOKEN_ID,
            }
        )
        configuration_hash = _stable_hash(
            {
                "version": 1,
                "operation": "byte_preserving_reblock",
                "source_manifest_sha256": source_manifest_sha,
                "source_profile": SOURCE_PROFILE,
                "target_profile": profile_payload(target),
            }
        )

        target_manifest = dict(source_manifest)
        production = target_manifest.get("production")
        if not isinstance(production, Mapping):
            raise ValueError("source manifest has no production identity")
        source_run_id = production.get("run_id")
        target_production = dict(production)
        target_production.update(
            {
                "run_id": target.run_id,
                "configuration_hash": configuration_hash,
                "schema_hash": schema_hash,
                # This derived corpus is intended to become remotely durable in
                # the named Modal Volume before training starts.  The companion
                # compatibility manifest below binds those exact bytes.
                "remote_required": True,
                "completion_reason": "byte_preserving_reblock",
            }
        )
        target_manifest.update(
            {
                "sequences_per_block": target.sequences_per_block,
                "target_shard_bytes": target.target_shard_bytes,
                "shards": train_rows + validation_rows,
                "last_durable_block_id": train_blocks - 1,
                "last_durable_train_block_id": train_blocks - 1,
                "last_durable_validation_block_id": validation_blocks - 1,
                "production": target_production,
                "reblock": {
                    "version": 1,
                    "byte_preserving": True,
                    "source_profile": SOURCE_PROFILE,
                    "source_run_id": source_run_id,
                    "source_manifest_sha256": source_manifest_sha,
                    "source_sequences_per_block": int(source_plan["sequences_per_block"]),
                    "target_sequences_per_block": target.sequences_per_block,
                    "train_stream_sha256": train_stream_sha,
                    "validation_stream_sha256": validation_stream_sha,
                    "remote_transport": "modal_volume",
                    "modal_volume": "small-llm-data",
                },
            }
        )
        write_json_atomic(staging / config.MANIFEST_FILENAME, target_manifest)

        drive_manifest = {
            "version": 1,
            "run_id": target.run_id,
            "configuration_hash": configuration_hash,
            "schema_hash": schema_hash,
            "transport": "modal_volume",
            "volume_name": "small-llm-data",
            "shards": [
                {
                    "filename": row["filename"],
                    # Qualification treats this field as an opaque durable-object
                    # identifier.  For the Modal-derived corpus it names the
                    # object in the immutable data Volume rather than Google Drive.
                    "drive_file_id": f"modal-volume:{target.run_id}:{row['filename']}",
                    "byte_size": row["byte_size"],
                    "local_sha256": row["checksum"],
                    "remote_durable": True,
                    "configuration_hash": configuration_hash,
                    "schema_hash": schema_hash,
                }
                for row in train_rows + validation_rows
            ],
        }
        drive_path = staging / "drive_manifest.json"
        write_json_atomic(drive_path, drive_manifest)

        target_report = verify(staging, full_scan=False)
        if not target_report.passed:
            raise RuntimeError(
                "reblocked dataset verification failed: " + "; ".join(target_report.problems)
            )
        target_plan = derive_plan(
            target_manifest,
            profile=target,
            manifest_path=staging / config.MANIFEST_FILENAME,
            drive_manifest_path=drive_path,
        )
        for split in ("train", "validation"):
            source_tokens = int(source_plan[split]["target_tokens"])
            target_tokens = int(target_plan[split]["target_tokens"])
            if source_tokens != target_tokens:
                raise RuntimeError(
                    f"reblock changed {split} target tokens: {source_tokens} -> {target_tokens}"
                )

        output_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, output_dir)
        return {
            "status": "complete",
            "source_dir": str(source_dir),
            "output_dir": str(output_dir),
            "source_profile": SOURCE_PROFILE,
            "target_profile": target.key,
            "run_id": target.run_id,
            "sequences_per_block": target.sequences_per_block,
            "train_blocks": int(target_plan["train"]["block_count"]),
            "validation_blocks": int(target_plan["validation"]["block_count"]),
            "train_target_tokens": int(target_plan["train"]["target_tokens"]),
            "validation_target_tokens": int(target_plan["validation"]["target_tokens"]),
            "train_stream_sha256": train_stream_sha,
            "validation_stream_sha256": validation_stream_sha,
        }
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Reblock a verified 2B schema-v2 corpus without retokenizing or redownloading it."
    )
    result.add_argument("--source-dir", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--profile", default=DEFAULT_TARGET_PROFILE)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        result = reblock_dataset(args.source_dir, args.output_dir, target_profile_key=args.profile)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception as error:  # noqa: BLE001 - concise CLI boundary
        print(f"dataset reblock error: {type(error).__name__}: {error}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
