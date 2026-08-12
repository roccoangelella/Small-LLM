"""Network-free cache tests for dynamic incremental shard boundaries."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from dataset.incremental_cache import IncrementalRollingShardCache
from dataset.incremental_frontier import SHARD_FRONTIER_FILENAME


class FakeStore:
    def __init__(self, frontier: dict[str, object]) -> None:
        self.bucket_id = "fake"
        self.frontier = frontier
        self.blobs: dict[str, bytes] = {}
        self.downloads: list[str] = []

    @staticmethod
    def object_key(run_id: str, logical_name: str) -> str:
        return f"run/{run_id}/{logical_name}"

    def _read_json(self, key: str) -> dict[str, object] | None:
        if key.endswith(SHARD_FRONTIER_FILENAME):
            return json.loads(json.dumps(self.frontier))
        return None

    def download_shard(
        self,
        *,
        run_id: str,
        logical_name: str,
        file_id: str,
        destination: Path,
        byte_size: int,
        sha256: str,
    ) -> None:
        assert file_id == self.object_key(run_id, logical_name)
        payload = self.blobs[file_id]
        assert len(payload) == byte_size
        assert hashlib.sha256(payload).hexdigest() == sha256
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        self.downloads.append(logical_name)


def _row(index: int, payload: bytes) -> dict[str, object]:
    return {
        "filename": f"train/train-{index:06d}.bin",
        "split": "train",
        "byte_size": len(payload),
        "checksum": hashlib.sha256(payload).hexdigest(),
        "first_block_id": index,
        "last_block_id": index,
        "sequence_count": 2,
    }


def test_successor_prefetch_is_promoted_without_duplicate_download(tmp_path: Path) -> None:
    run_id = "dataset-001"
    contract = {
        "run_id": run_id,
        "contract_sha256": "c" * 64,
        "planned_train_blocks": 3,
    }
    payloads = [b"a" * 16, b"b" * 16, b"c" * 16]
    rows = [_row(index, payload) for index, payload in enumerate(payloads)]
    remote_frontier = {
        "version": 1,
        "run_id": run_id,
        "contract_sha256": contract["contract_sha256"],
        "ready_train_shards": rows,
        "frozen_validation_shards": [],
        "producer_complete": False,
    }
    store = FakeStore(remote_frontier)
    for row, payload in zip(rows, payloads, strict=True):
        store.blobs[store.object_key(run_id, str(row["filename"]))] = payload

    # CPU staging has already placed current + successor and cached a frontier
    # that does not yet know about shard 2.
    (tmp_path / "train").mkdir()
    for index in (0, 1):
        (tmp_path / str(rows[index]["filename"])).write_bytes(payloads[index])
    local_frontier = dict(remote_frontier)
    local_frontier["ready_train_shards"] = rows[:2]
    (tmp_path / SHARD_FRONTIER_FILENAME).write_text(
        json.dumps(local_frontier), encoding="utf-8"
    )

    cache = IncrementalRollingShardCache(
        root=tmp_path,
        run_id=run_id,
        contract=contract,
        store=store,
        prefetch_shards=1,
        poll_seconds=0.001,
    )
    try:
        cache.ensure_block(0)
        cache.acknowledge(0)
        assert not (tmp_path / str(rows[0]["filename"])).exists()
        assert (tmp_path / str(rows[1]["filename"])).is_file()

        cache.ensure_block(1)
        cache.acknowledge(1)
        assert store.downloads.count(str(rows[2]["filename"])) == 1
        assert not (tmp_path / str(rows[1]["filename"])).exists()
        assert (tmp_path / str(rows[2]["filename"])).is_file()
    finally:
        cache.close()
