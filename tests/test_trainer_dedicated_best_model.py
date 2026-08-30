"""Tests for replace-only dedicated best-model publication."""
from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from trainer.best_model import get_dedicated_best_metric, publish_dedicated_best_model


class _Api:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def delete_repo(self, **kwargs):
        self.calls.append(("delete", dict(kwargs)))

    def create_repo(self, **kwargs):
        self.calls.append(("create", dict(kwargs)))

    def create_commit(self, **kwargs):
        self.calls.append(("commit", dict(kwargs)))
        return SimpleNamespace(oid="commit-1")


class DedicatedBestModelTests(unittest.TestCase):
    def test_metric_absent_when_dedicated_repo_does_not_exist(self) -> None:
        with (
            patch("huggingface_hub.HfApi", return_value=_Api()),
            patch("trainer.best_model._repo_exists", return_value=False),
        ):
            self.assertIsNone(
                get_dedicated_best_metric(
                    repo_id="owner/model-best-run",
                    run_id="run",
                    token="token",
                )
            )

    def test_existing_repo_is_marker_verified_then_deleted_before_single_commit(self) -> None:
        previous = {
            "version": 1,
            "role": "small-llm-dedicated-best-model",
            "run_id": "run",
            "checkpoint_id": "step-00000001",
            "metric": -2.0,
            "validation_loss": 2.0,
            "selection": "strict_validation_loss_improvement",
            "replacement": "delete_recreate_repository",
            "artifact_path": "models/run/step-00000001",
        }
        observed = {
            **previous,
            "checkpoint_id": "step-00000002",
            "metric": -1.5,
            "validation_loss": 1.5,
            "artifact_path": "models/run/step-00000002",
        }
        api = _Api()
        with TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "step-00000002"
            checkpoint.mkdir()
            state = checkpoint / "trainer_state.pkl"
            state.write_bytes(b"state")
            payload = checkpoint / "checkpoint.json"
            payload.write_text(
                '{"validation_metrics":{"loss":1.5}}',
                encoding="utf-8",
            )
            with (
                patch("huggingface_hub.HfApi", return_value=api),
                patch("trainer.best_model._repo_exists", return_value=True),
                patch("trainer.best_model._download_json", side_effect=[previous, observed]),
                patch("trainer.best_model.verify_local_manifest", return_value={}),
                patch("trainer.best_model._checkpoint_files", return_value=[state, payload]),
            ):
                result = publish_dedicated_best_model(
                    repo_id="owner/model-best-run",
                    run_id="run",
                    checkpoint_dir=checkpoint,
                    checkpoint_id="step-00000002",
                    metric=-1.5,
                    validation_loss=1.5,
                    token="token",
                    recreate=True,
                )

        self.assertEqual(result["status"], "replaced")
        self.assertEqual([name for name, _ in api.calls], ["delete", "create", "commit"])
        commit_kwargs = api.calls[-1][1]
        self.assertEqual(commit_kwargs["commit_message"], "Publish best run step-00000002")
        self.assertEqual(len(commit_kwargs["operations"]), 4)

    def test_existing_unmarked_repo_is_never_deleted(self) -> None:
        api = _Api()
        with TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "step-00000002"
            checkpoint.mkdir()
            (checkpoint / "checkpoint.json").write_text(
                '{"validation_metrics":{"loss":1.5}}',
                encoding="utf-8",
            )
            with (
                patch("huggingface_hub.HfApi", return_value=api),
                patch("trainer.best_model._repo_exists", return_value=True),
                patch(
                    "trainer.best_model._download_json",
                    return_value={"version": 1, "run_id": "other"},
                ),
                patch("trainer.best_model.verify_local_manifest", return_value={}),
                patch("trainer.best_model._checkpoint_files", return_value=[]),
            ):
                with self.assertRaisesRegex(RuntimeError, "not a Small-LLM dedicated"):
                    publish_dedicated_best_model(
                        repo_id="owner/model-best-run",
                        run_id="run",
                        checkpoint_dir=checkpoint,
                        checkpoint_id="step-00000002",
                        metric=-1.5,
                        validation_loss=1.5,
                        token="token",
                        recreate=True,
                    )
        self.assertEqual(api.calls, [])


if __name__ == "__main__":
    unittest.main()
