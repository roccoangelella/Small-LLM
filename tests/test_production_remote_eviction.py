"""Regression tests for bounded-disk production with verified remote shard eviction."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from dataset import production
from tests.production_helpers import (
    CountingShardStore,
    ReaderPatchMixin,
    documents,
    stream_config,
    work_plan,
)


class ProductionRemoteEvictionTests(ReaderPatchMixin, unittest.TestCase):
    def test_crash_resume_does_not_require_evicted_old_shards_locally(self) -> None:
        values = documents((3, 3, 4, 5))
        self.use_documents(values)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = CountingShardStore()
            policy = production.ProductionPolicy(
                "run",
                target_source_tokens=10,
                minimum_source_tokens=8,
                maximum_source_tokens=12,
                checkpoint_source_tokens=100,
            )
            with self.assertRaisesRegex(RuntimeError, "simulated production interruption"):
                production.build_production_cache(
                    root,
                    stream_config(),
                    policy,
                    work_plan(),
                    lambda _: None,
                    remote_store=store,
                    simulate_crash_after_documents=2,
                    evict_remote_shards=True,
                )
            self.assertTrue((root / "drive_manifest.json").is_file())
            self.assertEqual(list((root / "train").glob("train-*.bin")), [])
            self.assertEqual(list((root / "validation").glob("validation-*.bin")), [])

            self.use_documents(values)
            manifest = production.build_production_cache(
                root,
                stream_config(),
                policy,
                work_plan(),
                lambda _: None,
                remote_store=store,
                resume=True,
                evict_remote_shards=True,
            )
            self.assertEqual(manifest["accepted_source_tokens"], 10)
            self.assertEqual(list((root / "train").glob("train-*.bin")), [])
            self.assertEqual(list((root / "validation").glob("validation-*.bin")), [])
            self.assertGreater(store.uploads, 0)
            self.assertGreater(store.verifies, 0)

    def test_completed_remote_only_dataset_can_be_reopened_without_payload_download(self) -> None:
        values = documents((3, 3, 4, 5))
        self.use_documents(values)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = CountingShardStore()
            policy = production.ProductionPolicy(
                "run",
                target_source_tokens=10,
                minimum_source_tokens=8,
                maximum_source_tokens=12,
                checkpoint_source_tokens=4,
            )
            manifest = production.build_production_cache(
                root,
                stream_config(),
                policy,
                work_plan(),
                lambda _: None,
                remote_store=store,
                evict_remote_shards=True,
            )
            uploads = store.uploads
            verifies = store.verifies
            self.assertEqual(list(root.rglob("*.bin")), [])

            self.use_documents(values)
            same = production.build_production_cache(
                root,
                stream_config(),
                policy,
                work_plan(),
                lambda _: None,
                remote_store=store,
                resume=True,
                evict_remote_shards=True,
            )
            self.assertEqual(same, manifest)
            self.assertEqual(store.uploads, uploads)
            self.assertEqual(store.verifies, verifies)


if __name__ == "__main__":
    unittest.main()
