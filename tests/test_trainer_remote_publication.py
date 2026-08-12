"""Trainer CLI integration tests for fail-closed live remote publication."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from trainer.cli import main
from trainer.remote_publication import configure_remote_publication


class _Metrics:
    def __init__(self, step: int) -> None:
        self.step = step

    def as_dict(self) -> dict[str, object]:
        return {"step": self.step, "loss": 1.0}


class _Engine:
    def __init__(self) -> None:
        self.global_step = 0


class _Session:
    def __init__(self, engine: _Engine) -> None:
        self.engine = engine
        self.saved: list[str] = []

    def step(self) -> _Metrics:
        self.engine.global_step += 1
        return _Metrics(self.engine.global_step)

    def save_checkpoint(self, coordinator, checkpoint_id: str, **kwargs) -> None:
        self.saved.append(checkpoint_id)


class _Coordinator:
    def __init__(self, session: _Session, *, fail: bool = False) -> None:
        self.session = session
        self.fail = fail
        self.published: list[str] = []

    def publish(self, publisher, *, checkpoint_id: str, drive_manifest, **kwargs):
        if checkpoint_id not in self.session.saved:
            raise AssertionError("remote publication must follow a local checkpoint")
        self.published.append(checkpoint_id)
        if self.fail:
            raise RuntimeError("remote upload failed")
        return {"checkpoint_id": checkpoint_id, "best_updated": False}


_BASE = [
    "--dataset-dir",
    "/tmp/data",
    "--checkpoint-dir",
    "/tmp/checkpoints",
    "--steps",
    "3",
    "--remote-publish-every-steps",
    "2",
    "--remote-drive-manifest",
    "/tmp/drive_manifest.json",
]


class TrainerRemotePublicationTests(unittest.TestCase):
    def _objects(self, *, fail: bool = False, rolling: bool = False):
        engine = _Engine()
        session = _Session(engine)
        coordinator = _Coordinator(session, fail=fail)
        trainer_config = SimpleNamespace(
            evaluation_every_steps=0,
            checkpoint_every_steps=0,
        )
        remote = SimpleNamespace(
            publisher=object(),
            drive_manifest={"version": 1, "run_id": "run", "shards": []},
            every_steps=2,
            rolling_latest_only=rolling,
        )
        setup_result = (object(), trainer_config, engine, session, coordinator)
        return session, coordinator, remote, setup_result

    def test_periodic_publication_and_final_publication_share_atomic_boundaries(self) -> None:
        session, coordinator, remote, setup_result = self._objects()
        with (
            patch("trainer.cli.setup", return_value=setup_result),
            patch("trainer.cli.configure_remote_publication", return_value=remote),
            patch("trainer.cli.torch.cuda.is_available", return_value=False),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(main(list(_BASE)), 0)

        self.assertEqual(session.saved, ["step-00000002", "step-00000003"])
        self.assertEqual(coordinator.published, ["step-00000002", "step-00000003"])

    def test_remote_failure_is_propagated_after_local_checkpoint(self) -> None:
        session, coordinator, remote, setup_result = self._objects(fail=True)
        with (
            patch("trainer.cli.setup", return_value=setup_result),
            patch("trainer.cli.configure_remote_publication", return_value=remote),
            patch("trainer.cli.torch.cuda.is_available", return_value=False),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            with self.assertRaisesRegex(RuntimeError, "remote upload failed"):
                main(list(_BASE))

        self.assertEqual(session.saved, ["step-00000002"])
        self.assertEqual(coordinator.published, ["step-00000002"])

    def test_rolling_mode_cleans_after_each_verified_publication(self) -> None:
        session, coordinator, remote, setup_result = self._objects(rolling=True)
        cleanup_calls: list[str] = []

        def cleanup(_remote, *, checkpoint_id: str):
            cleanup_calls.append(checkpoint_id)
            return {"status": "pruned", "checkpoint_id": checkpoint_id}

        with (
            patch("trainer.cli.setup", return_value=setup_result),
            patch("trainer.cli.configure_remote_publication", return_value=remote),
            patch("trainer.cli.cleanup_remote_publication", side_effect=cleanup),
            patch("trainer.cli.torch.cuda.is_available", return_value=False),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(main(list(_BASE) + ["--remote-rolling-latest-only"]), 0)

        self.assertEqual(coordinator.published, ["step-00000002", "step-00000003"])
        self.assertEqual(cleanup_calls, ["step-00000002", "step-00000003"])

    def test_configuration_uses_verified_manifest_and_environment_repo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "drive_manifest.json"
            manifest = {
                "version": 1,
                "run_id": "qualification-run",
                "shards": [{"filename": "train/a.bin", "remote_durable": True}],
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            args = SimpleNamespace(
                remote_publish_every_steps=50,
                remote_drive_manifest=manifest_path,
                remote_checkpoint_repo=None,
                remote_checkpoint_bucket=None,
                remote_token_env="SMALL_LLM_TEST_UNUSED_TOKEN",
                remote_checkpoint_revision=None,
                remote_create_repo=False,
                remote_create_bucket=False,
                remote_rolling_latest_only=True,
            )
            store = object()
            publisher = object()
            with (
                patch.dict("os.environ", {"SMALL_LLM_HF_REPO_ID": "owner/private"}, clear=False),
                patch("trainer.remote_publication.HuggingFaceCheckpointStore", return_value=store) as store_type,
                patch("trainer.remote_publication.TwoPhaseCheckpointPublisher", return_value=publisher) as publisher_type,
            ):
                configured = configure_remote_publication(args)

            self.assertIsNotNone(configured)
            assert configured is not None
            self.assertEqual(configured.every_steps, 50)
            self.assertEqual(configured.drive_manifest, manifest)
            self.assertTrue(configured.rolling_latest_only)
            store_type.assert_called_once_with(
                "owner/private",
                token=None,
                private=True,
                revision=None,
                create_repo=False,
            )
            publisher_type.assert_called_once_with(store, run_id="qualification-run")

    def test_configuration_can_use_storage_bucket(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "transport.json"
            manifest = {"version": 1, "run_id": "modal-run", "shards": []}
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            args = SimpleNamespace(
                remote_publish_every_steps=500,
                remote_drive_manifest=manifest_path,
                remote_checkpoint_repo=None,
                remote_checkpoint_bucket="owner/modal-checkpoints",
                remote_token_env="HF_TOKEN",
                remote_checkpoint_revision=None,
                remote_create_repo=False,
                remote_create_bucket=True,
                remote_rolling_latest_only=True,
            )
            store = object()
            publisher = object()
            with (
                patch.dict("os.environ", {"HF_TOKEN": "secret"}, clear=False),
                patch("trainer.remote_publication.HuggingFaceBucketCheckpointStore", return_value=store) as store_type,
                patch("trainer.remote_publication.TwoPhaseCheckpointPublisher", return_value=publisher) as publisher_type,
            ):
                configured = configure_remote_publication(args)

            self.assertIsNotNone(configured)
            assert configured is not None
            self.assertTrue(configured.rolling_latest_only)
            store_type.assert_called_once_with(
                "owner/modal-checkpoints",
                token="secret",
                private=True,
                create_bucket=True,
            )
            publisher_type.assert_called_once_with(store, run_id="modal-run")

    def test_configuration_rejects_non_durable_shard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "drive_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "run_id": "qualification-run",
                        "shards": [{"filename": "train/a.bin", "remote_durable": False}],
                    }
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                remote_publish_every_steps=50,
                remote_drive_manifest=manifest_path,
                remote_checkpoint_repo="owner/private",
                remote_checkpoint_bucket=None,
                remote_token_env="SMALL_LLM_TEST_UNUSED_TOKEN",
                remote_checkpoint_revision=None,
                remote_create_repo=False,
                remote_create_bucket=False,
                remote_rolling_latest_only=False,
            )
            with self.assertRaisesRegex(SystemExit, "not verified remote_durable"):
                configure_remote_publication(args)


if __name__ == "__main__":
    unittest.main()
