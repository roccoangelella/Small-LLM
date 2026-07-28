from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dataset import production
from tests.production_helpers import (
    CountingDriveStore,
    ReaderPatchMixin,
    documents,
    stream_config,
    work_plan,
)


class ProductionLimitTest(ReaderPatchMixin, unittest.TestCase):
    def test_target_stop_and_remote_manifest(self) -> None:
        self.use_documents(documents())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = CountingDriveStore()
            policy = production.ProductionPolicy(
                "run",
                target_source_tokens=10,
                minimum_source_tokens=8,
                maximum_source_tokens=12,
                checkpoint_source_tokens=100,
            )
            manifest = production.build_production_cache(
                root,
                stream_config(),
                policy,
                work_plan(),
                lambda _: None,
                remote_store=store,
            )
            self.assertEqual(manifest["accepted_source_tokens"], 10)
            self.assertTrue(manifest["production"]["target_reached"])
            self.assertEqual(manifest["production"]["completion_reason"], "target_reached")
            remote = json.loads((root / "drive_manifest.json").read_text())
            self.assertEqual(len(remote["shards"]), len(manifest["shards"]))
            self.assertTrue(all(item["remote_durable"] for item in remote["shards"]))

    def test_token_checkpoint_does_not_finalize_per_document(self) -> None:
        self.use_documents(documents((2, 2, 2, 2, 2)))
        with tempfile.TemporaryDirectory() as tmp:
            policy = production.ProductionPolicy(
                "run",
                target_source_tokens=10,
                minimum_source_tokens=1,
                maximum_source_tokens=10,
                checkpoint_source_tokens=1_000_000_000,
                remote_required=False,
            )
            manifest = production.build_production_cache(
                Path(tmp),
                stream_config(),
                policy,
                work_plan(),
                lambda _: None,
                remote_store=None,
            )
            self.assertEqual(len(manifest["shards"]), 1)

    def test_hard_maximum_never_splits_a_document(self) -> None:
        self.use_documents(documents((7, 7)))
        with tempfile.TemporaryDirectory() as tmp:
            policy = production.ProductionPolicy(
                "run",
                target_source_tokens=12,
                minimum_source_tokens=5,
                maximum_source_tokens=12,
                checkpoint_source_tokens=100,
                remote_required=False,
            )
            manifest = production.build_production_cache(
                Path(tmp),
                stream_config(),
                policy,
                work_plan(),
                lambda _: None,
                remote_store=None,
            )
            self.assertEqual(manifest["accepted_source_tokens"], 7)
            self.assertEqual(manifest["production"]["completion_reason"], "hard_maximum_guard")
            self.assertLessEqual(manifest["accepted_source_tokens"], 12)
