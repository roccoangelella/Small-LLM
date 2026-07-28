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


class ProductionResumeTest(ReaderPatchMixin, unittest.TestCase):
    def test_resume_cursor_and_remote_upload_are_idempotent(self) -> None:
        values = documents((3, 3, 4, 5))
        self.use_documents(values)
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
            with self.assertRaisesRegex(RuntimeError, "simulated production interruption"):
                production.build_production_cache(
                    root,
                    stream_config(),
                    policy,
                    work_plan(),
                    lambda _: None,
                    remote_store=store,
                    simulate_crash_after_documents=2,
                )
            uploads_after_crash = store.uploads
            self.use_documents(values)
            manifest = production.build_production_cache(
                root,
                stream_config(),
                policy,
                work_plan(),
                lambda _: None,
                remote_store=store,
                resume=True,
            )
            self.assertEqual(manifest["accepted_source_tokens"], 10)
            self.assertGreaterEqual(store.uploads, uploads_after_crash)
            uploads_after_complete = store.uploads
            self.use_documents(values)
            same = production.build_production_cache(
                root,
                stream_config(),
                policy,
                work_plan(),
                lambda _: None,
                remote_store=store,
                resume=True,
            )
            self.assertEqual(same, manifest)
            self.assertEqual(store.uploads, uploads_after_complete)
            self.assertGreater(store.verifies, 0)

    def test_resume_rejects_changed_policy(self) -> None:
        values = documents((3, 3, 4))
        self.use_documents(values)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = production.ProductionPolicy(
                "run",
                target_source_tokens=10,
                minimum_source_tokens=1,
                maximum_source_tokens=12,
                checkpoint_source_tokens=100,
                remote_required=False,
            )
            with self.assertRaises(RuntimeError):
                production.build_production_cache(
                    root,
                    stream_config(),
                    first,
                    work_plan(),
                    lambda _: None,
                    remote_store=None,
                    simulate_crash_after_documents=1,
                )
            changed = production.ProductionPolicy(
                "run",
                target_source_tokens=11,
                minimum_source_tokens=1,
                maximum_source_tokens=12,
                checkpoint_source_tokens=100,
                remote_required=False,
            )
            self.use_documents(values)
            with self.assertRaisesRegex(ValueError, "configuration does not match"):
                production.build_production_cache(
                    root,
                    stream_config(),
                    changed,
                    work_plan(),
                    lambda _: None,
                    remote_store=None,
                    resume=True,
                )

    def test_progress_backup_recovers_interrupted_final_commit(self) -> None:
        values = documents((3, 3, 4))
        self.use_documents(values)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            policy = production.ProductionPolicy(
                "run",
                target_source_tokens=10,
                minimum_source_tokens=1,
                maximum_source_tokens=10,
                checkpoint_source_tokens=100,
                remote_required=False,
            )
            with self.assertRaises(RuntimeError):
                production.build_production_cache(
                    root,
                    stream_config(),
                    policy,
                    work_plan(),
                    lambda _: None,
                    remote_store=None,
                    simulate_crash_after_documents=1,
                )
            committed = json.loads((root / "progress.json").read_text())
            (root / production.PROGRESS_BACKUP_FILENAME).write_text(json.dumps(committed))
            (root / "progress.json").write_text(json.dumps({"schema_version": 2}))
            (root / "manifest.json").write_text(json.dumps({"orphan": True}))
            orphan = root / "train" / "train-999999.bin"
            orphan.parent.mkdir(exist_ok=True)
            orphan.write_bytes(b"orphan")

            self.use_documents(values)
            manifest = production.build_production_cache(
                root,
                stream_config(),
                policy,
                work_plan(),
                lambda _: None,
                remote_store=None,
                resume=True,
            )
            self.assertEqual(manifest["accepted_source_tokens"], 10)
            self.assertFalse((root / production.PROGRESS_BACKUP_FILENAME).exists())
            self.assertFalse(orphan.exists())
