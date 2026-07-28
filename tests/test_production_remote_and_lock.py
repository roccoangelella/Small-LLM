from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from dataset import production
from tests.production_helpers import (
    FlakyDriveStore,
    ReaderPatchMixin,
    documents,
    stream_config,
    work_plan,
)


class ProductionRemoteTest(ReaderPatchMixin, unittest.TestCase):
    def test_transient_remote_failure_is_retried(self) -> None:
        self.use_documents(documents((3, 3, 4)))
        with tempfile.TemporaryDirectory() as tmp:
            store = FlakyDriveStore()
            policy = production.ProductionPolicy(
                "run",
                target_source_tokens=10,
                minimum_source_tokens=1,
                maximum_source_tokens=10,
                checkpoint_source_tokens=100,
            )
            manifest = production.build_production_cache(
                Path(tmp),
                stream_config(),
                policy,
                work_plan(),
                lambda _: None,
                remote_store=store,
            )
            self.assertTrue(store.failed_once)
            self.assertEqual(manifest["accepted_source_tokens"], 10)

    def test_run_lock_rejects_concurrent_writer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with production.RunLock(root):
                with self.assertRaisesRegex(RuntimeError, "another dataset process"):
                    with production.RunLock(root):
                        pass
