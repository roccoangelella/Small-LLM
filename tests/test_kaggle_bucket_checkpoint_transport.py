"""Kaggle contracts for Bucket latest plus the shared dedicated-best adapter."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from dataset.src.remote import build_checkpoint_manifest, sha256_path

ROOT = Path(__file__).resolve().parents[1]
KAGGLE = ROOT / "kaggle"


def _load_impl():
    if str(KAGGLE) not in sys.path:
        sys.path.insert(0, str(KAGGLE))
    spec = importlib.util.spec_from_file_location(
        "small_llm_kaggle_bucket_transport_test",
        KAGGLE / "deep_decay_10b_from_15500_impl.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _PointerStore:
    def __init__(self, checkpoint_id: str | None = None) -> None:
        self.pointer = (
            None
            if checkpoint_id is None
            else {"checkpoint_id": checkpoint_id, "last_prefix": "unused"}
        )

    def read_json(self, path: str):
        del path
        return self.pointer


class _BucketStore(_PointerStore):
    def __init__(self, checkpoint_id: str | None = None) -> None:
        super().__init__(checkpoint_id)
        self.upload_calls = 0

    def upload_tree(self, prefix: str, local_dir: Path) -> dict[str, str]:
        self.upload_calls += 1
        return {
            f"{prefix}/{path.relative_to(local_dir).as_posix()}": sha256_path(path)
            for path in sorted(local_dir.rglob("*"))
            if path.is_file()
        }

    def write_json(self, path: str, value: dict[str, object]) -> None:
        if path.endswith("/latest.json"):
            self.pointer = dict(value)

    def prune_run_checkpoints(self, *, run_id: str, checkpoint_id: str) -> dict[str, object]:
        return {
            "status": "pruned",
            "run_id": run_id,
            "checkpoint_id": checkpoint_id,
        }


class KaggleBucketCheckpointTransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_impl()

    def test_newer_bucket_wins_over_stale_legacy_pointer(self) -> None:
        bucket = _PointerStore("step-00070250")
        legacy = _PointerStore("step-00061500")
        runtime = SimpleNamespace(
            _hf_bucket_store=lambda: bucket,
            _hf_model_repo_store=lambda: legacy,
        )

        selected = self.module._remote_checkpoint_state(runtime, run_id="run")

        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual(selected["source"], "hf_bucket")
        self.assertEqual(selected["checkpoint_id"], "step-00070250")

    def test_bucket_wins_a_step_tie_and_newer_legacy_remains_migratable(self) -> None:
        bucket = _PointerStore("step-00020000")
        legacy = _PointerStore("step-00020000")
        runtime = SimpleNamespace(
            _hf_bucket_store=lambda: bucket,
            _hf_model_repo_store=lambda: legacy,
        )
        selected = self.module._remote_checkpoint_state(runtime, run_id="run")
        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual(selected["source"], "hf_bucket")

        legacy.pointer = {
            "checkpoint_id": "step-00020250",
            "last_prefix": "unused",
        }
        selected = self.module._remote_checkpoint_state(runtime, run_id="run")
        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual(selected["source"], "legacy_hf_model_repo")
        self.assertEqual(selected["checkpoint_id"], "step-00020250")

    def test_cpu_gate_publishes_and_reads_back_bucket_latest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            checkpoint_dir = run_dir / "checkpoints"
            checkpoint_id = "step-00015500"
            checkpoint = checkpoint_dir / checkpoint_id
            checkpoint.mkdir(parents=True)
            (checkpoint / "trainer_state.pkl").write_bytes(b"state")
            dataset = root / "dataset"
            dataset.mkdir()
            store = _BucketStore()

            def write_manifest(path: Path, **kwargs: object) -> dict[str, object]:
                payload = {
                    "version": 1,
                    "run_id": kwargs["run_id"],
                    "shards": [],
                    "transport": "modal-hf-bucket-checkpoint-v1",
                    "bucket_id": kwargs["bucket_id"],
                    "source_commit": kwargs["source_commit"],
                    "microbatch_size": kwargs["microbatch_size"],
                }
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}\n", encoding="utf-8")
                return payload

            runtime = SimpleNamespace(
                _hf_bucket_store=lambda: store,
                _hf_checkpoint_bucket_id=lambda: "owner/base-checkpoints",
                _write_hf_transport_manifest=write_manifest,
            )
            with (
                patch.object(self.module, "RUN_DIR", run_dir),
                patch.object(self.module, "CHECKPOINT_DIR", checkpoint_dir),
                patch.dict(
                    "os.environ",
                    {"SMALL_LLM_SOURCE_COMMIT": "abc123"},
                    clear=False,
                ),
            ):
                result = self.module._publish_latest_to_bucket(
                    runtime,
                    checkpoint_id=checkpoint_id,
                    dataset=dataset,
                )

            self.assertEqual(result["status"], "pruned")
            self.assertIsNotNone(store.pointer)
            assert store.pointer is not None
            self.assertEqual(store.pointer["checkpoint_id"], checkpoint_id)

    def test_same_checkpoint_id_does_not_hide_stale_remote_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            checkpoint_dir = run_dir / "checkpoints"
            checkpoint_id = "step-00070250"
            checkpoint = checkpoint_dir / checkpoint_id
            checkpoint.mkdir(parents=True)
            (checkpoint / "trainer_state.pkl").write_bytes(b"kaggle-migrated-state")
            transport = {
                "version": 1,
                "run_id": self.module.RUN_ID,
                "shards": [],
                "transport": "modal-hf-bucket-checkpoint-v1",
                "bucket_id": "owner/base-checkpoints",
                "source_commit": "abc123",
                "microbatch_size": self.module.MICROBATCH_SIZE,
            }
            (checkpoint / "drive_manifest.json").write_text(
                json.dumps(transport) + "\n",
                encoding="utf-8",
            )
            dataset = root / "dataset"
            dataset.mkdir()
            store = _BucketStore(checkpoint_id)
            assert store.pointer is not None
            store.pointer["checkpoint_manifest"] = {"version": 1, "files": []}

            def write_manifest(path: Path, **kwargs: object) -> dict[str, object]:
                del kwargs
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}\n", encoding="utf-8")
                return dict(transport)

            runtime = SimpleNamespace(
                _hf_bucket_store=lambda: store,
                _hf_checkpoint_bucket_id=lambda: "owner/base-checkpoints",
                _write_hf_transport_manifest=write_manifest,
            )
            environment = {
                "SMALL_LLM_SOURCE_COMMIT": "abc123",
                "SMALL_LLM_KAGGLE_PROVIDER_MIGRATION_CHECKPOINT_ID": "",
            }
            with (
                patch.object(self.module, "RUN_DIR", run_dir),
                patch.object(self.module, "CHECKPOINT_DIR", checkpoint_dir),
                patch.dict("os.environ", environment, clear=False),
            ):
                published = self.module._publish_latest_to_bucket(
                    runtime,
                    checkpoint_id=checkpoint_id,
                    dataset=dataset,
                )
                skipped = self.module._publish_latest_to_bucket(
                    runtime,
                    checkpoint_id=checkpoint_id,
                    dataset=dataset,
                )

            self.assertEqual(published["status"], "pruned")
            self.assertEqual(skipped["status"], "already_current")
            self.assertEqual(store.upload_calls, 1)
            assert store.pointer is not None
            self.assertEqual(
                store.pointer["checkpoint_manifest"],
                build_checkpoint_manifest(checkpoint),
            )


if __name__ == "__main__":
    unittest.main()
