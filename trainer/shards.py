"""Immutable schema-v2 shard and restored-cache reader."""
from __future__ import annotations
from pathlib import Path
from typing import Iterator, Mapping
from .decode import PreparedBlockDecoder
from .shard_config import load_manifest, resolve_geometry
from .shard_io import read_location
from .shard_layout import BlockLocation, build_locations, manifest_identity
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
        return len(self._locations)

    def _read(self, item: BlockLocation) -> TokenBatch:
        if self.cache_manager is not None:
            ensure = getattr(self.cache_manager, "ensure_block", None)
            if not callable(ensure):
                raise RuntimeError("configured shard cache manager has no ensure_block method")
            ensure(item.block_id)
        return read_location(self, item)

    def next_batch(self, timeout: float | None = None) -> TokenBatch:
        del timeout
        if self._outstanding is not None:
            raise RuntimeError("the previous block has not been acknowledged")
        index = self._index.get(self.last_acknowledged_block_id + 1)
        if index is None:
            raise StopIteration
        self._outstanding = self._read(self._locations[index])
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
        for index, item in enumerate(self._locations):
            if maximum_blocks is not None and index >= maximum_blocks:
                break
            yield self._read(item)
