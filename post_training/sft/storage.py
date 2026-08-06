"""Immutable sharded SFT block storage with exact target masks and resume state."""

from __future__ import annotations

from array import array
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import struct
import sys
from typing import Iterable, Mapping

from .config import SFTDataConfig
from .schema import SFTBlock, TokenizedSFTRecord

_MAGIC = b"SFTB1\0"
_PREFIX = struct.Struct("<6sIQQ")
_SCHEMA_VERSION = 1


def _canonical_json(payload: Mapping[str, object]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def encode_sft_block(block: SFTBlock) -> bytes:
    lengths = [len(record.token_ids) for record in block.records]
    header = {
        "schema_version": _SCHEMA_VERSION,
        "block_id": block.block_id,
        "split": block.split,
        "record_ids": [record.record_id for record in block.records],
        "sources": [record.source for record in block.records],
        "lengths": lengths,
        "target_counts": [record.target_token_count for record in block.records],
        "target_token_count": block.target_token_count,
        "serialized_token_count": block.serialized_token_count,
    }
    header_bytes = _canonical_json(header)
    tokens = array("H")
    masks = bytearray()
    for record in block.records:
        tokens.extend(record.token_ids)
        masks.extend(1 if value else 0 for value in record.target_mask)
    if sys.byteorder != "little":  # pragma: no cover
        tokens.byteswap()
    token_bytes = tokens.tobytes()
    prefix = _PREFIX.pack(
        _MAGIC,
        len(header_bytes),
        len(token_bytes),
        len(masks),
    )
    return prefix + header_bytes + token_bytes + bytes(masks)


def decode_sft_block(payload: bytes) -> SFTBlock:
    if len(payload) < _PREFIX.size:
        raise ValueError("SFT block payload is truncated")
    magic, header_size, token_bytes_size, mask_size = _PREFIX.unpack_from(payload)
    if magic != _MAGIC:
        raise ValueError("SFT block magic mismatch")
    expected = _PREFIX.size + header_size + token_bytes_size + mask_size
    if len(payload) != expected:
        raise ValueError("SFT block byte geometry mismatch")
    cursor = _PREFIX.size
    try:
        header = json.loads(payload[cursor : cursor + header_size])
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("SFT block header is invalid") from error
    cursor += header_size
    if not isinstance(header, Mapping) or header.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError("SFT block schema mismatch")

    token_bytes = payload[cursor : cursor + token_bytes_size]
    cursor += token_bytes_size
    masks = payload[cursor:]
    if token_bytes_size % 2:
        raise ValueError("SFT block token bytes are not uint16-aligned")
    tokens = array("H")
    tokens.frombytes(token_bytes)
    if sys.byteorder != "little":  # pragma: no cover
        tokens.byteswap()

    lengths = header.get("lengths")
    record_ids = header.get("record_ids")
    sources = header.get("sources")
    target_counts = header.get("target_counts")
    if not all(isinstance(value, list) for value in (lengths, record_ids, sources, target_counts)):
        raise ValueError("SFT block record metadata is malformed")
    count = len(lengths)
    if not (len(record_ids) == len(sources) == len(target_counts) == count):
        raise ValueError("SFT block record metadata lengths disagree")

    token_cursor = mask_cursor = 0
    records: list[TokenizedSFTRecord] = []
    for length, record_id, source, target_count in zip(
        lengths, record_ids, sources, target_counts, strict=True
    ):
        if isinstance(length, bool) or not isinstance(length, int) or length < 2:
            raise ValueError("SFT block contains an invalid record length")
        mask_length = length - 1
        record_tokens = tuple(int(value) for value in tokens[token_cursor : token_cursor + length])
        record_mask_bytes = masks[mask_cursor : mask_cursor + mask_length]
        if len(record_tokens) != length or len(record_mask_bytes) != mask_length:
            raise ValueError("SFT block record payload is truncated")
        if any(value not in (0, 1) for value in record_mask_bytes):
            raise ValueError("SFT block target mask is not binary")
        record = TokenizedSFTRecord(
            record_id=str(record_id),
            source=str(source),
            split=str(header["split"]),  # type: ignore[arg-type]
            token_ids=record_tokens,
            target_mask=tuple(bool(value) for value in record_mask_bytes),
            metadata={},
        )
        if record.target_token_count != target_count:
            raise ValueError("SFT block target count mismatch")
        records.append(record)
        token_cursor += length
        mask_cursor += mask_length

    if token_cursor != len(tokens) or mask_cursor != len(masks):
        raise ValueError("SFT block has unreferenced token or mask bytes")
    block = SFTBlock(
        block_id=int(header["block_id"]),
        split=str(header["split"]),  # type: ignore[arg-type]
        records=tuple(records),
    )
    if block.target_token_count != header.get("target_token_count"):
        raise ValueError("SFT block aggregate target count mismatch")
    return block


@dataclass(frozen=True, slots=True)
class StoredBlock:
    block_id: int
    split: str
    shard: str
    offset: int
    byte_size: int
    target_token_count: int
    serialized_token_count: int
    record_count: int
    cumulative_target_tokens: int

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "StoredBlock":
        return cls(
            block_id=int(payload["block_id"]),
            split=str(payload["split"]),
            shard=str(payload["shard"]),
            offset=int(payload["offset"]),
            byte_size=int(payload["byte_size"]),
            target_token_count=int(payload["target_token_count"]),
            serialized_token_count=int(payload["serialized_token_count"]),
            record_count=int(payload["record_count"]),
            cumulative_target_tokens=int(payload["cumulative_target_tokens"]),
        )


class SFTDatasetWriter:
    """Write a complete immutable dataset through a temporary sibling directory."""

    def __init__(self, output_dir: Path | str, config: SFTDataConfig) -> None:
        self.output_dir = Path(output_dir)
        self.config = config

    def write(self, blocks: Iterable[SFTBlock]) -> dict[str, object]:
        if self.output_dir.exists():
            raise FileExistsError(f"refusing to replace existing SFT dataset: {self.output_dir}")
        temporary = self.output_dir.with_name(f".{self.output_dir.name}.tmp")
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir(parents=True)

        block_entries: list[dict[str, object]] = []
        shard_entries: list[dict[str, object]] = []
        source_target_tokens: dict[str, int] = {}
        cumulative_targets = 0
        shard_index = -1
        shard_path: Path | None = None
        shard_handle = None
        shard_bytes = 0

        def open_shard() -> None:
            nonlocal shard_index, shard_path, shard_handle, shard_bytes
            shard_index += 1
            shard_path = temporary / f"shard-{shard_index:05d}.sft"
            shard_handle = shard_path.open("wb")
            shard_bytes = 0

        def close_shard() -> None:
            nonlocal shard_handle
            if shard_handle is None or shard_path is None:
                return
            shard_handle.flush()
            shard_handle.close()
            shard_entries.append(
                {
                    "path": shard_path.name,
                    "byte_size": shard_path.stat().st_size,
                    "sha256": _sha256_file(shard_path),
                }
            )
            shard_handle = None

        try:
            expected_id = 0
            for block in blocks:
                if block.block_id != expected_id:
                    raise ValueError("SFT blocks must be contiguous from zero")
                encoded = encode_sft_block(block)
                if shard_handle is None:
                    open_shard()
                elif shard_bytes and shard_bytes + len(encoded) > self.config.shard_target_bytes:
                    close_shard()
                    open_shard()
                assert shard_handle is not None and shard_path is not None
                offset = shard_handle.tell()
                shard_handle.write(encoded)
                shard_bytes += len(encoded)
                cumulative_targets += block.target_token_count
                for record in block.records:
                    source_target_tokens[record.source] = (
                        source_target_tokens.get(record.source, 0)
                        + record.target_token_count
                    )
                block_entries.append(
                    {
                        "block_id": block.block_id,
                        "split": block.split,
                        "shard": shard_path.name,
                        "offset": offset,
                        "byte_size": len(encoded),
                        "target_token_count": block.target_token_count,
                        "serialized_token_count": block.serialized_token_count,
                        "record_count": len(block.records),
                        "cumulative_target_tokens": cumulative_targets,
                    }
                )
                expected_id += 1
            close_shard()
            if not block_entries:
                raise RuntimeError("cannot publish an empty SFT dataset")

            manifest_without_hash: dict[str, object] = {
                "schema": "small-llm-sft",
                "schema_version": _SCHEMA_VERSION,
                "config": self.config.as_dict(),
                "blocks": block_entries,
                "shards": shard_entries,
                "totals": {
                    "blocks": len(block_entries),
                    "records": sum(int(item["record_count"]) for item in block_entries),
                    "loss_bearing_target_tokens": cumulative_targets,
                    "serialized_tokens": sum(
                        int(item["serialized_token_count"]) for item in block_entries
                    ),
                    "source_target_tokens": source_target_tokens,
                },
            }
            manifest_hash = _sha256_bytes(_canonical_json(manifest_without_hash))
            manifest = {**manifest_without_hash, "manifest_sha256": manifest_hash}
            (temporary / "manifest.json").write_bytes(
                json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n"
            )
            temporary.rename(self.output_dir)
            return manifest
        except BaseException:
            if shard_handle is not None:
                shard_handle.close()
            shutil.rmtree(temporary, ignore_errors=True)
            raise


class SFTShardReader:
    """Verified block reader compatible with ``TrainingSession``."""

    def __init__(
        self,
        root: Path | str,
        *,
        split: str = "train",
        verify_checksums: bool = True,
        pad_token_id: int = 50_256,
    ) -> None:
        self.root = Path(root)
        try:
            manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError("SFT manifest is missing or invalid") from error
        if not isinstance(manifest, Mapping):
            raise RuntimeError("SFT manifest must be an object")
        supplied_hash = manifest.get("manifest_sha256")
        without_hash = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
        expected_hash = _sha256_bytes(_canonical_json(without_hash))
        if supplied_hash != expected_hash:
            raise RuntimeError("SFT manifest self-hash mismatch")
        if manifest.get("schema") != "small-llm-sft" or manifest.get("schema_version") != _SCHEMA_VERSION:
            raise RuntimeError("unsupported SFT dataset schema")
        if split not in {"train", "validation", "test"}:
            raise ValueError("invalid SFT reader split")

        raw_blocks = manifest.get("blocks")
        raw_shards = manifest.get("shards")
        if not isinstance(raw_blocks, list) or not isinstance(raw_shards, list):
            raise RuntimeError("SFT manifest blocks/shards are malformed")
        self._blocks = [
            StoredBlock.from_mapping(item)
            for item in raw_blocks
            if isinstance(item, Mapping) and item.get("split") == split
        ]
        self._index = {item.block_id: index for index, item in enumerate(self._blocks)}
        self._shard_hashes = {
            str(item["path"]): str(item["sha256"])
            for item in raw_shards
            if isinstance(item, Mapping)
        }
        if len(self._index) != len(self._blocks):
            raise RuntimeError("duplicate SFT block IDs")
        self.split = split
        self.verify_checksums = verify_checksums
        self.pad_token_id = pad_token_id
        self.manifest_identity = str(supplied_hash)
        self.last_acknowledged_block_id = -1
        self._outstanding = None
        self._verified: set[str] = set()

    @property
    def block_count(self) -> int:
        return len(self._blocks)

    @property
    def block_target_counts(self) -> tuple[int, ...]:
        return tuple(item.target_token_count for item in self._blocks)

    def _read(self, item: StoredBlock):
        path = self.root / item.shard
        if item.shard not in self._verified:
            if not path.is_file() or path.is_symlink():
                raise FileNotFoundError(f"missing or unsafe SFT shard: {path}")
            if self.verify_checksums and _sha256_file(path) != self._shard_hashes.get(item.shard):
                raise RuntimeError(f"SFT shard checksum mismatch: {path}")
            self._verified.add(item.shard)
        with path.open("rb") as handle:
            handle.seek(item.offset)
            payload = handle.read(item.byte_size)
        if len(payload) != item.byte_size:
            raise RuntimeError("short SFT shard read")
        block = decode_sft_block(payload)
        if (
            block.block_id != item.block_id
            or block.target_token_count != item.target_token_count
        ):
            raise RuntimeError("SFT manifest and block payload disagree")
        return block.to_token_batch(pad_token_id=self.pad_token_id)

    def next_batch(self, timeout: float | None = None):
        del timeout
        if self._outstanding is not None:
            raise RuntimeError("the previous SFT block has not been acknowledged")
        index = self._index.get(self.last_acknowledged_block_id + 1)
        if index is None:
            raise StopIteration
        item = self._blocks[index]
        self._outstanding = self._read(item)
        return self._outstanding

    def acknowledge(self, block_id: int) -> None:
        if self._outstanding is None or self._outstanding.block_id != block_id:
            raise ValueError("only the current SFT block may be acknowledged")
        self.last_acknowledged_block_id = block_id
        self._outstanding = None

    def iter_from_start(self):
        for item in self._blocks:
            yield self._read(item)

    def pipeline_state(self) -> dict[str, object]:
        return {
            "schema": "small-llm-sft-pipeline",
            "manifest_identity": self.manifest_identity,
            "split": self.split,
            "last_consumed_block_id": self.last_acknowledged_block_id,
            "gradient_accumulation_position": 0,
            "consumer": {
                "kind": "sft_shard_reader",
                "manifest_identity": self.manifest_identity,
                "split": self.split,
            },
        }

    def load_pipeline_state(self, state: Mapping[str, object]) -> None:
        if state.get("manifest_identity") != self.manifest_identity:
            raise RuntimeError("SFT resume manifest identity mismatch")
        if state.get("split") != self.split:
            raise RuntimeError("SFT resume split mismatch")
        cursor = state.get("last_consumed_block_id")
        if isinstance(cursor, bool) or not isinstance(cursor, int) or cursor < -1:
            raise RuntimeError("SFT resume cursor is invalid")
        if cursor >= 0 and cursor not in self._index:
            raise RuntimeError("SFT resume cursor is outside this dataset")
        self.last_acknowledged_block_id = cursor
        self._outstanding = None


__all__ = [
    "SFTDatasetWriter",
    "SFTShardReader",
    "StoredBlock",
    "decode_sft_block",
    "encode_sft_block",
]
