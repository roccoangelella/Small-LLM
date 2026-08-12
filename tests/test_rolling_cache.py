"""Tests for checkpoint-aligned CPU staging and current+next rolling shard cache."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from dataset.rolling_cache import RollingShardCache, stage_dataset_window, verify_staged_dataset


class FakeShardStore:
    bucket_id = "owner/datasets"

    def __init__(self, manifest: dict[str, object], objects: dict[str, bytes]) -> None:
        self.manifest = manifest
        self.objects = objects
        self.downloads: list[str] = []

    @staticmethod
    def object_key(run_id: str, logical_name: str) -> str:
        return f"run/{run_id}/{logical_name}"

    def download_dataset_manifest(self, *, run_id: str, destination: Path):
        self.downloads.append("manifest.json")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.manifest), encoding="utf-8")
        return self.manifest

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
        self.assert_key = self.object_key(run_id, logical_name)
        if file_id != self.assert_key:
            raise RuntimeError("wrong object key")
        data = self.objects[logical_name]
        if len(data) != byte_size or hashlib.sha256(data).hexdigest() != sha256:
            raise RuntimeError("fake remote object identity mismatch")
        self.downloads.append(logical_name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)


def _row(filename: str, split: str, first: int, last: int, data: bytes) -> dict[str, object]:
    return {
        "filename": filename,
        "split": split,
        "byte_size": len(data),
        "checksum": hashlib.sha256(data).hexdigest(),
        "first_block_id": first,
        "last_block_id": last,
    }


def fixture() -> tuple[dict[str, object], dict[str, bytes]]:
    objects = {
        "train/train-000000.bin": b"train-zero",
        "train/train-000001.bin": b"train-one",
        "train/train-000002.bin": b"train-two",
        "validation/validation-000000.bin": b"validation",
    }
    manifest = {
        "production": {"run_id": "dataset-001"},
        "shards": [
            _row("train/train-000000.bin", "train", 0, 1, objects["train/train-000000.bin"]),
            _row("train/train-000001.bin", "train", 2, 3, objects["train/train-000001.bin"]),
            _row("train/train-000002.bin", "train", 4, 5, objects["train/train-000002.bin"]),
            _row(
                "validation/validation-000000.bin",
                "validation",
                0,
                0,
                objects["validation/validation-000000.bin"],
            ),
        ],
    }
    return manifest, objects


class RollingShardCacheTests(unittest.TestCase):
    def test_cpu_stage_is_checkpoint_aligned_and_keeps_validation(self) -> None:
        manifest, objects = fixture()
        store = FakeShardStore(manifest, objects)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            staged = stage_dataset_window(
                store=store,  # type: ignore[arg-type]
                run_id="dataset-001",
                destination=root,
                start_block_id=2,
                train_shards=1,
            )
            self.assertEqual(staged["status"], "ready")
            self.assertEqual(staged["staged_train_shards"], ["train/train-000001.bin"])
            self.assertFalse((root / "train/train-000000.bin").exists())
            self.assertTrue((root / "train/train-000001.bin").is_file())
            self.assertFalse((root / "train/train-000002.bin").exists())
            self.assertTrue((root / "validation/validation-000000.bin").is_file())
            verified = verify_staged_dataset(
                destination=root,
                bucket_id=store.bucket_id,
                run_id="dataset-001",
                required_train_block=2,
            )
            self.assertEqual(verified["required_train_shard"], "train/train-000001.bin")

    def test_training_complete_stage_needs_no_train_or_validation_download(self) -> None:
        manifest, objects = fixture()
        store = FakeShardStore(manifest, objects)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            staged = stage_dataset_window(
                store=store,  # type: ignore[arg-type]
                run_id="dataset-001",
                destination=root,
                start_block_id=6,
            )
            self.assertEqual(staged["status"], "training_complete")
            self.assertEqual(store.downloads, ["manifest.json"])
            verified = verify_staged_dataset(
                destination=root,
                bucket_id=store.bucket_id,
                run_id="dataset-001",
                required_train_block=6,
            )
            self.assertEqual(verified["status"], "training_complete")

    def test_current_plus_next_prefetch_and_consumed_shard_eviction(self) -> None:
        manifest, objects = fixture()
        store = FakeShardStore(manifest, objects)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stage_dataset_window(
                store=store,  # type: ignore[arg-type]
                run_id="dataset-001",
                destination=root,
                start_block_id=0,
            )
            cache = RollingShardCache(
                root=root,
                run_id="dataset-001",
                manifest=manifest,
                store=store,
                prefetch_shards=1,
                evict_consumed=True,
            )
            try:
                cache.ensure_block(0)
                # Reaching the next shard waits for the background prefetch and
                # proves no whole-dataset materialization was required.
                cache.acknowledge(1)
                self.assertFalse((root / "train/train-000000.bin").exists())
                cache.ensure_block(2)
                self.assertTrue((root / "train/train-000001.bin").is_file())
                self.assertFalse((root / "train/train-000000.bin").exists())
                cache.acknowledge(3)
                cache.ensure_block(4)
                self.assertFalse((root / "train/train-000001.bin").exists())
                self.assertTrue((root / "train/train-000002.bin").is_file())
            finally:
                cache.close()


if __name__ == "__main__":
    unittest.main()
