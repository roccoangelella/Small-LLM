"""Network-free cache tests for dynamic incremental shard boundaries."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from dataset.incremental_cache import IncrementalRollingShardCache
from dataset.incremental_frontier import FrontierShard, SHARD_FRONTIER_FILENAME


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
        if file_id != self.object_key(run_id, logical_name):
            raise AssertionError("unexpected remote shard identity")
        payload = self.blobs[file_id]
        if len(payload) != byte_size or hashlib.sha256(payload).hexdigest() != sha256:
            raise AssertionError("fake remote payload does not match requested identity")
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


class IncrementalCacheTests(unittest.TestCase):
    def test_wait_timeout_tracks_ready_progress_and_zero_keeps_waiting(self) -> None:
        for timeout, progress, completes in ((3, False, False), (3, True, False), (0, False, True)):
            with self.subTest(timeout=timeout, progress=progress), tempfile.TemporaryDirectory() as tmp:
                contract = {"run_id": "test", "contract_sha256": "c" * 64,
                            "planned_train_blocks": 3}
                frontier = {"version": 1, "run_id": "test", "contract_sha256": "c" * 64,
                            "ready_train_shards": [], "frozen_validation_shards": [],
                            "producer_complete": False}
                store = FakeStore(frontier)
                cache = IncrementalRollingShardCache(
                    root=Path(tmp), run_id="test", contract=contract, store=store,
                    poll_seconds=1, wait_timeout_seconds=timeout,
                )
                clock = [0.0]

                def sleep(seconds):
                    clock[0] += seconds
                    if progress and clock[0] >= 2:
                        frontier["ready_train_shards"] = [_row(0, b"a" * 16)]
                    if completes and clock[0] >= 5:
                        frontier["ready_train_shards"] = [
                            _row(i, b"a" * 16) for i in range(3)
                        ]

                try:
                    with patch("dataset.incremental_cache.time.monotonic", side_effect=lambda: clock[0]), \
                         patch("dataset.incremental_cache.time.sleep", side_effect=sleep):
                        if completes:
                            self.assertEqual(cache._wait_for_shard(2).last_block_id, 2)
                        else:
                            with self.assertRaisesRegex(TimeoutError, "no incremental READY progress"):
                                cache._wait_for_shard(2)
                    self.assertEqual(clock[0], 5 if progress or completes else 3)
                finally:
                    cache.close()

    def test_wait_timeout_rejects_invalid_values(self) -> None:
        for timeout in (-1, float("inf"), float("nan")):
            with self.subTest(timeout=timeout), self.assertRaisesRegex(ValueError, "wait timeout"):
                IncrementalRollingShardCache(wait_timeout_seconds=timeout)

    def test_acknowledge_rejects_unsafe_filename_before_deletion(self) -> None:
        run_id = "dataset-001"
        contract = {
            "run_id": run_id,
            "contract_sha256": "c" * 64,
            "planned_train_blocks": 1,
        }
        frontier = {
            "version": 1,
            "run_id": run_id,
            "contract_sha256": contract["contract_sha256"],
            "ready_train_shards": [],
            "frozen_validation_shards": [],
            "producer_complete": True,
        }
        store = FakeStore(frontier)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outside = Path(tmp) / "outside.bin"
            outside.write_bytes(b"sentinel")
            cache = IncrementalRollingShardCache(
                root=root,
                run_id=run_id,
                contract=contract,
                store=store,
                prefetch_shards=1,
                poll_seconds=0.001,
            )
            cache._cached_train = [
                FrontierShard(
                    "../outside.bin",
                    "train",
                    1,
                    hashlib.sha256(b"x").hexdigest(),
                    0,
                    0,
                    1,
                )
            ]
            try:
                with self.assertRaisesRegex(RuntimeError, "unsafe"):
                    cache.acknowledge(0)
            finally:
                cache.close()

            self.assertTrue(outside.exists())

    def test_successor_prefetch_is_promoted_without_duplicate_download(self) -> None:
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

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "train").mkdir()
            for index in (0, 1):
                (root / str(rows[index]["filename"])).write_bytes(payloads[index])
            local_frontier = dict(remote_frontier)
            local_frontier["ready_train_shards"] = rows[:2]
            (root / SHARD_FRONTIER_FILENAME).write_text(
                json.dumps(local_frontier), encoding="utf-8"
            )

            cache = IncrementalRollingShardCache(
                root=root,
                run_id=run_id,
                contract=contract,
                store=store,
                prefetch_shards=1,
                poll_seconds=0.001,
            )
            try:
                cache.ensure_block(0)
                cache.acknowledge(0)
                self.assertFalse((root / str(rows[0]["filename"])).exists())
                self.assertTrue((root / str(rows[1]["filename"])).is_file())

                cache.ensure_block(1)
                cache.acknowledge(1)
                self.assertEqual(store.downloads.count(str(rows[2]["filename"])), 1)
                self.assertFalse((root / str(rows[1]["filename"])).exists())
                self.assertTrue((root / str(rows[2]["filename"])).is_file())
            finally:
                cache.close()


if __name__ == "__main__":
    unittest.main()
