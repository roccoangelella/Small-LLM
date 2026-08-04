"""Trainer CLI integration tests for fail-closed live remote publication."""

from __future__ import annotations

import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from trainer.cli import main


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
    def _objects(self, *, fail: bool = False):
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


if __name__ == "__main__":
    unittest.main()
