"""Immutable schema-v2 shard and restored-cache reader."""
from __future__ import annotations
from pathlib import Path
from typing import Iterator, Mapping
from .decode import PreparedBlockDecoder
from .shard_config import load_manifest, resolve_geometry
from .shard_io import read_location
from .shard_layout import BlockLocation, build_locations, manifest_identity, safe_path
from .shard_state import load_pipeline_state, load_reader_state, pipeline_state, reader_state
from .types import TokenBatch

class SchemaV2ShardReader:
    """Read immutable shards in exact block order, optionally through a rolling cache."""
    def __init__(self, root: Path | str, *, split: str = "train",
                 sequences_per_block: int | None = None, semantic_vocab_size: int = 50_257,
                 verify_checksums: bool = True, manifest_path: Path | str | None = None,
                 context_length: int | None = None, cache_manager: object | None = None) -> None:
        self.root = Path(root)
        path = Path(manifest_path) if manifest_path is not None else self.root / "manifest.json"
        manifest = load_manifest(path)
        if split not in {"train", "validation"}:
            raise ValueError("split must be train or validation")
        context_length, block_size = resolve_geometry(manifest,
            context_length=context_length, sequences_per_block=sequences_per_block)
        self.split, self.context_length = split, context_length
        self.sequences_per_block, self.verify_checksums = block_size, verify_checksums
        self.cache_manager = cache_manager
        self.decoder = PreparedBlockDecoder(context_length=context_length,
            semantic_vocab_size=semantic_vocab_size, expected_split=split)
        self.manifest_identity = manifest_identity(manifest, context_length, block_size)
        self._locations = build_locations(self.root, manifest, split=split,
            context_length=context_length, sequences_per_block=block_size)
        self._index = {item.block_id: index for index, item in enumerate(self._locations)}
        planned = getattr(cache_manager, "planned_block_count", None)
        shard_for_block = getattr(cache_manager, "shard_for_block", None)
        self._dynamic_frontier = (
            split == "train"
            and isinstance(planned, int)
            and not isinstance(planned, bool)
            and planned > 0
            and callable(shard_for_block)
        )
        self._planned_block_count = int(planned) if self._dynamic_frontier else len(self._locations)
        if self._dynamic_frontier and self._locations:
            if self._locations[-1].block_id >= self._planned_block_count:
                raise ValueError("bootstrap manifest extends beyond the frozen incremental horizon")
        self._verified: set[Path] = set()
        self.last_acknowledged_block_id = -1
        self._outstanding: TokenBatch | None = None

    @classmethod
    def from_restored_checkpoint(cls, checkpoint_root: Path | str, cache_root: Path | str, *,
                                 context_length: int, sequences_per_block: int,
                                 split: str = "train", semantic_vocab_size: int = 50_257,
                                 verify_checksums: bool = True) -> "SchemaV2ShardReader":
        return cls(cache_root, split=split, sequences_per_block=sequences_per_block,
            semantic_vocab_size=semantic_vocab_size, verify_checksums=verify_checksums,
            manifest_path=Path(checkpoint_root) / "drive_manifest.json",
            context_length=context_length)

    @property
    def block_count(self) -> int:
        return self._planned_block_count

    def _dynamic_locations(self, block_id: int) -> tuple[BlockLocation, ...]:
        if not self._dynamic_frontier or self.cache_manager is None:
            raise RuntimeError("dynamic shard locations requested for a static reader")
        shard_for_block = getattr(self.cache_manager, "shard_for_block", None)
        if not callable(shard_for_block):
            raise RuntimeError("incremental cache manager has no shard_for_block method")
        shard = shard_for_block(block_id)
        filename = getattr(shard, "filename", None)
        first = getattr(shard, "first_block_id", None)
        last = getattr(shard, "last_block_id", None)
        count = getattr(shard, "sequence_count", None)
        byte_size = getattr(shard, "byte_size", None)
        checksum = getattr(shard, "checksum", None)
        if (
            not isinstance(filename, str)
            or isinstance(first, bool) or not isinstance(first, int)
            or isinstance(last, bool) or not isinstance(last, int)
            or isinstance(count, bool) or not isinstance(count, int)
            or isinstance(byte_size, bool) or not isinstance(byte_size, int)
            or not isinstance(checksum, str) or len(checksum) != 64
        ):
            raise RuntimeError("incremental frontier returned malformed shard metadata")
        if first < 0 or last < first or not first <= block_id <= last or count <= 0:
            raise RuntimeError("incremental frontier returned invalid block geometry")
        record_bytes = (self.context_length + 1) * 2
        if byte_size != count * record_bytes:
            raise RuntimeError("incremental frontier shard byte/sequence counts disagree")
        blocks = last - first + 1
        final_count = count - (blocks - 1) * self.sequences_per_block
        if not 1 <= final_count <= self.sequences_per_block:
            raise RuntimeError("incremental frontier shard cannot be partitioned into prepared blocks")
        if last < self._planned_block_count - 1 and final_count != self.sequences_per_block:
            raise RuntimeError("incremental READY prefix contains a partial non-terminal training block")
        path = self.root / safe_path(filename)
        result: list[BlockLocation] = []
        offset = 0
        for current in range(first, last + 1):
            sequences = final_count if current == last else self.sequences_per_block
            current_bytes = sequences * record_bytes
            result.append(BlockLocation(current, path, offset, current_bytes, sequences, checksum))
            offset += current_bytes
        if offset != byte_size:
            raise RuntimeError("incremental frontier shard offsets do not cover the object")
        return tuple(result)

    def _read(self, item: BlockLocation, *, peers: tuple[BlockLocation, ...] | None = None) -> TokenBatch:
        if self.cache_manager is not None:
            ensure = getattr(self.cache_manager, "ensure_block", None)
            if not callable(ensure):
                raise RuntimeError("configured shard cache manager has no ensure_block method")
            ensure(item.block_id)
        return read_location(self, item, peer_locations=peers)

    def _item_for_block(self, block_id: int) -> tuple[BlockLocation, tuple[BlockLocation, ...] | None]:
        index = self._index.get(block_id)
        if index is not None:
            return self._locations[index], None
        if not self._dynamic_frontier:
            raise StopIteration
        peers = self._dynamic_locations(block_id)
        item = next((location for location in peers if location.block_id == block_id), None)
        if item is None:
            raise RuntimeError(f"incremental shard does not contain requested block {block_id}")
        return item, peers

    def next_batch(self, timeout: float | None = None) -> TokenBatch:
        del timeout
        if self._outstanding is not None:
            raise RuntimeError("the previous block has not been acknowledged")
        next_block = self.last_acknowledged_block_id + 1
        if next_block >= self.block_count:
            raise StopIteration
        item, peers = self._item_for_block(next_block)
        self._outstanding = self._read(item, peers=peers)
        return self._outstanding

    def acknowledge(self, block_id: int) -> None:
        if self._outstanding is None or block_id != self._outstanding.block_id:
            raise ValueError("only the current trained block may be acknowledged")
        self.last_acknowledged_block_id, self._outstanding = block_id, None
        if self.cache_manager is not None:
            acknowledge = getattr(self.cache_manager, "acknowledge", None)
            if not callable(acknowledge):
                raise RuntimeError("configured shard cache manager has no acknowledge method")
            acknowledge(block_id)

    def state_dict(self) -> dict[str, object]:
        return reader_state(self)

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        load_reader_state(self, state)
        if self.cache_manager is not None:
            restore = getattr(self.cache_manager, "restore_after_acknowledged", None)
            if not callable(restore):
                raise RuntimeError("configured shard cache manager has no restore_after_acknowledged method")
            restore(self.last_acknowledged_block_id)

    def pipeline_state(self) -> dict[str, object]:
        return pipeline_state(self)

    def load_pipeline_state(self, state: Mapping[str, object]) -> None:
        load_pipeline_state(self, state)
        if self.cache_manager is not None:
            restore = getattr(self.cache_manager, "restore_after_acknowledged", None)
            if not callable(restore):
                raise RuntimeError("configured shard cache manager has no restore_after_acknowledged method")
            restore(self.last_acknowledged_block_id)

    def iter_from_start(self, maximum_blocks: int | None = None) -> Iterator[TokenBatch]:
        if maximum_blocks is not None and maximum_blocks < 0:
            raise ValueError("maximum_blocks must be non-negative")
        limit = self.block_count if maximum_blocks is None else min(self.block_count, maximum_blocks)
        for block_id in range(limit):
            item, peers = self._item_for_block(block_id)
            yield self._read(item, peers=peers)
