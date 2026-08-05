"""Tests for validation-loss-based remote best-checkpoint selection."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from trainer.cli import _existing_remote_best_metric, _validation_metric


class _Store:
    def __init__(self, pointer):
        self.pointer = pointer
        self.paths: list[str] = []

    def read_json(self, path: str):
        self.paths.append(path)
        return self.pointer


class TrainerBestCheckpointTests(unittest.TestCase):
    def test_validation_loss_is_negated_for_higher_is_better_publisher(self) -> None:
        self.assertEqual(_validation_metric({"loss": 2.5}), -2.5)
        self.assertIsNone(_validation_metric(None))

    def test_existing_best_metric_is_read_on_resume(self) -> None:
        store = _Store({"metric": -1.75})
        remote = SimpleNamespace(
            publisher=SimpleNamespace(store=store),
            drive_manifest={"run_id": "pretrain-run"},
        )
        self.assertEqual(_existing_remote_best_metric(remote), -1.75)
        self.assertEqual(store.paths, ["run/pretrain-run/best.json"])

    def test_invalid_remote_metric_fails_closed(self) -> None:
        remote = SimpleNamespace(
            publisher=SimpleNamespace(store=_Store({"metric": "bad"})),
            drive_manifest={"run_id": "pretrain-run"},
        )
        with self.assertRaisesRegex(RuntimeError, "numeric metric"):
            _existing_remote_best_metric(remote)


if __name__ == "__main__":
    unittest.main()
