"""Validate schema-v2/Drive manifests and map blocks into immutable shards."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib, json
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Mapping

@dataclass(frozen=True, slots=True)
class BlockLocation:
    block_id: int
    path: Path
    offset: int
    byte_size: int
    sequences: int
    checksum: str

def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def safe_path(value: object) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise ValueError(f"unsafe shard filename: {value!r}")
    posix, windows = PurePosixPath(value), PureWindowsPath(value)
    if posix.is_absolute() or windows.is_absolute() or windows.drive:
        raise ValueError(f"unsafe shard filename: {value!r}")
    if any(part in {"", ".", ".."} for part in value.split("/")):
        raise ValueError(f"unsafe shard filename: {value!r}")
    return Path(*value.split("/"))

def manifest_identity(manifest: Mapping[str, object], context: int, block_size: int) -> str:
    raw = manifest.get("shards")
    if not isinstance(raw, list):
        raise ValueError("dataset manifest shards must be a list")
    shards = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise ValueError("dataset manifest contains a malformed shard")
        shards.append({"filename": item.get("filename"), "split": item.get("split"),
            "byte_size": item.get("byte_size"), "sequence_count": item.get("sequence_count"),
            "checksum": item.get("checksum", item.get("local_sha256")),
            "first_block_id": item.get("first_block_id"), "last_block_id": item.get("last_block_id")})
    shards.sort(key=lambda x: (str(x["split"]), int(x["first_block_id"] or -1), str(x["filename"])))
    value = {"schema_version": 2, "sequence_format": "context_plus_one",
             "context_length": context, "stored_tokens_per_sequence": context + 1,
             "sequences_per_block": block_size, "shards": shards}
    raw_json = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw_json).hexdigest()

def build_locations(root: Path, manifest: Mapping[str, object], *, split: str,
                    context_length: int, sequences_per_block: int) -> tuple[BlockLocation, ...]:
    raw = manifest.get("shards")
    if not isinstance(raw, list):
        raise ValueError("dataset manifest shards must be a list")
    shards = [x for x in raw if isinstance(x, Mapping) and x.get("split") == split]
    if len(shards) != sum(isinstance(x, Mapping) and x.get("split") == split for x in raw):
        raise ValueError("dataset manifest contains a malformed shard")
    shards.sort(key=lambda x: (int(x.get("first_block_id", -1)), str(x.get("filename", ""))))
    record_bytes, expected, locations = (context_length + 1) * 2, 0, []
    for shard in shards:
        first, last = shard.get("first_block_id"), shard.get("last_block_id")
        count, size = shard.get("sequence_count"), shard.get("byte_size")
        checksum = shard.get("checksum", shard.get("local_sha256"))
        if any(isinstance(x, bool) or not isinstance(x, int) for x in (first, last, count, size)):
            raise ValueError("dataset shard has non-integer geometry")
        assert all(isinstance(x, int) for x in (first, last, count, size))
        if first != expected or last < first or count <= 0 or size != count * record_bytes:
            raise ValueError("dataset shard block ranges or geometry are invalid")
        if not isinstance(checksum, str) or len(checksum) != 64:
            raise ValueError("dataset shard checksum is invalid")
        blocks = last - first + 1
        final_count = count - (blocks - 1) * sequences_per_block
        if not 1 <= final_count <= sequences_per_block:
            raise ValueError("shard cannot be partitioned into prepared blocks")
        path, offset = root / safe_path(shard.get("filename")), 0
        for block_id in range(first, last + 1):
            sequences = final_count if block_id == last else sequences_per_block
            byte_size = sequences * record_bytes
            locations.append(BlockLocation(block_id, path, offset, byte_size, sequences, checksum))
            offset += byte_size
        if offset != size:
            raise ValueError("shard block offsets do not cover the file")
        expected = last + 1
    if any(item.sequences != sequences_per_block for item in locations[:-1]):
        raise ValueError("only the final split block may be partial")
    return tuple(locations)
