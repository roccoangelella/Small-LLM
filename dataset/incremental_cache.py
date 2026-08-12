"""Low-overhead dynamic cache for an incremental READY shard frontier."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping

from dataset.incremental_frontier import (
    SHARD_FRONTIER_FILENAME,
    FrontierShard,
    IncrementalRollingShardCache as _BaseIncrementalRollingShardCache,
    _frontier_shards,
    _train_index_for_block,
)


class IncrementalRollingShardCache(_BaseIncrementalRollingShardCache):
    """Poll HF only when a requested block is beyond the locally known READY prefix."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._cached_train: list[FrontierShard] = []
        self._cached_producer_complete = False
        local = self.root / SHARD_FRONTIER_FILENAME
        if local.is_file():
            try:
                payload = json.loads(local.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                payload = None
            if isinstance(payload, Mapping):
                self._accept_frontier(dict(payload), require_remote_monotonic=False)

    def _accept_frontier(
        self,
        frontier: dict[str, object],
        *,
        require_remote_monotonic: bool = True,
    ) -> None:
        if frontier.get("run_id") != self.run_id:
            raise RuntimeError("cached incremental frontier belongs to a different run")
        if frontier.get("contract_sha256") != self.contract.get("contract_sha256"):
            raise RuntimeError("cached incremental frontier belongs to a different run contract")
        train = _frontier_shards(frontier, "ready_train_shards")
        old_payload = [row.as_dict() for row in self._cached_train]
        new_payload = [row.as_dict() for row in train]
        if require_remote_monotonic and new_payload[: len(old_payload)] != old_payload:
            raise RuntimeError("remote training frontier regressed or mutated")
        if len(new_payload) < len(old_payload):
            raise RuntimeError("incremental training frontier lost READY shards")
        self._cached_train = train
        self._cached_producer_complete = bool(frontier.get("producer_complete"))
        self._last_frontier = frontier

    def _refresh_ready_prefix(self) -> None:
        frontier = super()._frontier()
        self._accept_frontier(frontier)

    def _cached_shard(self, block_id: int) -> FrontierShard | None:
        index = _train_index_for_block(self._cached_train, block_id)
        return None if index is None else self._cached_train[index]

    def _wait_for_shard(self, block_id: int) -> FrontierShard:
        cached = self._cached_shard(block_id)
        if cached is not None:
            return cached
        while True:
            self._refresh_ready_prefix()
            cached = self._cached_shard(block_id)
            if cached is not None:
                return cached
            if self._cached_producer_complete:
                raise RuntimeError(f"producer completed without required train block {block_id}")
            time.sleep(self.poll_seconds)


__all__ = ["IncrementalRollingShardCache"]
