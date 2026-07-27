"""Checkpoint-aware JSONL shard writer and selected-output reader."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Iterator

from dataset import config

from .models import SelectionState
from .storage import read_jsonl


LOGGER = logging.getLogger(__name__)


class JsonlShardWriter:
    """Append output with checkpoint-aware recovery that prevents duplicates."""

    def __init__(self, output_dir: Path, state: SelectionState) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state = state
        self.handle: Any | None = None
        self.shard_id = state.next_shard_id
        self.shard_name: str | None = None
        self.shard_bytes = 0
        self._discard_uncheckpointed_shards()
        if state.active_shard is not None:
            self.shard_name = state.active_shard
            path = self.output_dir / self.shard_name
            if not path.exists():
                raise RuntimeError(f"Checkpoint refers to missing shard: {path}")
            self.handle = path.open("r+b")
            self.handle.truncate(state.active_shard_bytes)
            self.handle.seek(0, os.SEEK_END)
            self.shard_bytes = state.active_shard_bytes
        else:
            self._open_new_shard()

    def _discard_uncheckpointed_shards(self) -> None:
        """Remove only shards created after the last durable checkpoint."""

        stale_paths = []
        for path in self.output_dir.glob("part-*.jsonl"):
            try:
                shard_id = int(path.stem.removeprefix("part-"))
            except ValueError:
                continue
            if shard_id > self.state.next_shard_id:
                stale_paths.append(path)
        for path in stale_paths:
            LOGGER.warning("Discarding uncheckpointed shard during resume: %s", path)
            path.unlink()

    def _open_new_shard(self) -> None:
        self.shard_name = f"part-{self.shard_id:05d}.jsonl"
        self.handle = (self.output_dir / self.shard_name).open("ab")
        self.shard_bytes = 0

    def write(self, row: dict[str, Any]) -> int:
        """Append one row, rolling only at a document boundary."""

        encoded = (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        if self.shard_bytes and self.shard_bytes + len(encoded) > config.OUTPUT_SHARD_MAX_BYTES:
            self.handle.close()
            self.shard_id += 1
            self._open_new_shard()
        self.handle.write(encoded)
        self.shard_bytes += len(encoded)
        return len(encoded)

    def checkpoint(self, state: SelectionState) -> None:
        """Flush the active shard and update state fields ready for persistence."""

        if self.handle is None or self.shard_name is None:
            raise RuntimeError("Writer is closed")
        self.handle.flush()
        os.fsync(self.handle.fileno())
        state.next_shard_id = self.shard_id
        state.active_shard = self.shard_name
        state.active_shard_bytes = self.shard_bytes

    def close(self) -> None:
        """Close the active shard without altering the last checkpoint."""

        if self.handle is not None:
            self.handle.close()
            self.handle = None


def iter_selected_documents() -> Iterator[dict[str, Any]]:
    """Read selected JSONL shards in deterministic shard/line order."""

    for path in sorted(config.OUTPUT_DIR.glob("part-*.jsonl")):
        yield from read_jsonl(path)
