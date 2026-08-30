"""Tests for validation-loss-based remote best-checkpoint selection."""

from __future__ import annotations

import contextlib
import io
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from trainer.cli import _existing_remote_best_metric, _validation_metric, main


class _Store:
    def __init__(self, pointer=None):
        self.pointer = pointer
        self.paths: list[str] = []
        self.writes: list[tuple[str, dict[str, object]]] = []

    def read_json(self, path: str):
        self.paths.append(path)
        return self.pointer

    def write_json(self, path: str, value):
        self.writes.append((path, dict(value)))


class _Metrics:
    def as_dict(self):
        return {"step": 1, "loss": 3.0}


class _Engine:
    global_step = 0

    def evaluate(self, batches, *, maximum_batches=None):
        list(batches)
        return {
            "loss": 2.0,
            "perplexity": 7.0,
            "target_tokens": 4,
            "blocks": maximum_batches or 1,
        }


class _Session:
    def __init__(self, engine):
        self.engine = engine
        self.saved: list[str] = []

    def step(self):
        self.engine.global_step += 1
        return _Metrics()

    def save_checkpoint(self, coordinator, checkpoint_id: str, **kwargs):
        self.saved.append(checkpoint_id)
        return f"/tmp/{checkpoint_id}"


class _ValidationReader:
    def iter_from_start(self, maximum_blocks=None):
        return iter((object(),))


class _Coordinator:
    def publish(self, publisher, *, checkpoint_id: str, drive_manifest, **kwargs):
        return {
            "latest": {
                "checkpoint_id": checkpoint_id,
                "last_prefix": f"run/pretrain-run/checkpoints/{checkpoint_id}/last",
                "checkpoint_manifest": {"version": 1, "files": []},
            },
            "best_updated": False,
        }


class TrainerBestCheckpointTests(unittest.TestCase):
    def test_validation_loss_is_negated_for_higher_is_better_comparison(self) -> None:
        self.assertEqual(_validation_metric({"loss": 2.5}), -2.5)
        self.assertIsNone(_validation_metric(None))

    def test_dedicated_best_model_publishes_only_after_validation_improves(self) -> None:
        engine = _Engine()
        session = _Session(engine)
        trainer_config = SimpleNamespace(
            evaluation_every_steps=1,
            checkpoint_every_steps=1,
        )
        args = SimpleNamespace(
            steps=1,
            validation_blocks=1,
            resume=None,
            checkpoint_dir=Path("/tmp"),
            best_model_repo="owner/model-best-pretrain-run",
            best_model_recreate=True,
            remote_token_env="HF_TOKEN",
            wandb_run_id="pretrain-run",
        )
        setup_result = (
            object(),
            trainer_config,
            engine,
            session,
            _Coordinator(),
        )
        with (
            patch("trainer.cli.parse_args", return_value=args),
            patch("trainer.cli.setup", return_value=setup_result),
            patch("trainer.cli.validation_reader", return_value=_ValidationReader()),
            patch("trainer.cli.configure_remote_publication", return_value=None),
            patch("trainer.cli.get_dedicated_best_metric", return_value=None),
            patch(
                "trainer.cli.publish_dedicated_best_model",
                return_value={
                    "status": "published",
                    "repo_id": args.best_model_repo,
                    "checkpoint_id": "step-00000001",
                },
            ) as publish,
            patch("trainer.cli.configure_wandb", return_value=None),
            patch("trainer.cli.torch.cuda.is_available", return_value=False),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(main([]), 0)

        publish.assert_called_once()
        kwargs = publish.call_args.kwargs
        self.assertEqual(kwargs["checkpoint_id"], "step-00000001")
        self.assertEqual(kwargs["metric"], -2.0)
        self.assertEqual(kwargs["validation_loss"], 2.0)
        self.assertTrue(kwargs["recreate"])

    def test_dedicated_best_model_respects_persisted_historical_best(self) -> None:
        engine = _Engine()
        engine.best_validation_loss = 1.5
        session = _Session(engine)
        trainer_config = SimpleNamespace(
            evaluation_every_steps=1,
            checkpoint_every_steps=1,
        )
        args = SimpleNamespace(
            steps=1,
            validation_blocks=1,
            resume=None,
            checkpoint_dir=Path("/tmp"),
            best_model_repo="owner/model-best-pretrain-run",
            best_model_recreate=True,
            remote_token_env="HF_TOKEN",
            wandb_run_id="pretrain-run",
        )
        setup_result = (object(), trainer_config, engine, session, _Coordinator())
        with (
            patch("trainer.cli.parse_args", return_value=args),
            patch("trainer.cli.setup", return_value=setup_result),
            patch("trainer.cli.validation_reader", return_value=_ValidationReader()),
            patch("trainer.cli.configure_remote_publication", return_value=None),
            patch("trainer.cli.get_dedicated_best_metric", return_value=None),
            patch("trainer.cli.publish_dedicated_best_model") as publish,
            patch("trainer.cli.configure_wandb", return_value=None),
            patch("trainer.cli.torch.cuda.is_available", return_value=False),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(main([]), 0)
        publish.assert_not_called()

    def test_missing_best_repo_repairs_from_resumed_historical_best_checkpoint(self) -> None:
        engine = _Engine()
        engine.best_validation_loss = 2.0
        session = _Session(engine)
        trainer_config = SimpleNamespace(
            evaluation_every_steps=1,
            checkpoint_every_steps=1,
        )
        args = SimpleNamespace(
            steps=1,
            validation_blocks=1,
            resume="step-00000000",
            checkpoint_dir=Path("/tmp"),
            best_model_repo="owner/model-best-pretrain-run",
            best_model_recreate=True,
            remote_token_env="HF_TOKEN",
            wandb_run_id="pretrain-run",
        )
        setup_result = (object(), trainer_config, engine, session, _Coordinator())
        with (
            patch("trainer.cli.parse_args", return_value=args),
            patch("trainer.cli.setup", return_value=setup_result),
            patch("trainer.cli.validation_reader", return_value=_ValidationReader()),
            patch("trainer.cli.configure_remote_publication", return_value=None),
            patch("trainer.cli.get_dedicated_best_metric", return_value=None),
            patch("trainer.cli.checkpoint_validation_loss", return_value=2.0),
            patch(
                "trainer.cli.publish_dedicated_best_model",
                return_value={"status": "published", "checkpoint_id": args.resume},
            ) as publish,
            patch("trainer.cli.configure_wandb", return_value=None),
            patch("trainer.cli.torch.cuda.is_available", return_value=False),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(main([]), 0)
        publish.assert_called_once()
        self.assertEqual(publish.call_args.kwargs["checkpoint_id"], args.resume)
        self.assertEqual(publish.call_args.kwargs["validation_loss"], 2.0)

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

    def test_training_writes_pointer_only_best_after_verified_latest(self) -> None:
        store = _Store()
        remote = SimpleNamespace(
            publisher=SimpleNamespace(store=store),
            drive_manifest={"run_id": "pretrain-run"},
            every_steps=1,
        )
        engine = _Engine()
        session = _Session(engine)
        trainer_config = SimpleNamespace(
            evaluation_every_steps=1,
            checkpoint_every_steps=0,
        )
        args = SimpleNamespace(
            steps=1,
            validation_blocks=1,
            resume=None,
        )
        setup_result = (
            object(),
            trainer_config,
            engine,
            session,
            _Coordinator(),
        )
        with (
            patch("trainer.cli.parse_args", return_value=args),
            patch("trainer.cli.setup", return_value=setup_result),
            patch("trainer.cli.validation_reader", return_value=_ValidationReader()),
            patch("trainer.cli.configure_remote_publication", return_value=remote),
            patch("trainer.cli.configure_wandb", return_value=None),
            patch("trainer.cli.torch.cuda.is_available", return_value=False),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(main([]), 0)

        self.assertEqual(session.saved, ["step-00000001"])
        self.assertEqual(len(store.writes), 1)
        path, pointer = store.writes[0]
        self.assertEqual(path, "run/pretrain-run/best.json")
        self.assertEqual(pointer["metric"], -2.0)
        self.assertEqual(
            pointer["best_prefix"],
            "run/pretrain-run/checkpoints/step-00000001/last",
        )
        self.assertEqual(pointer["checkpoint_manifest"], {"version": 1, "files": []})


if __name__ == "__main__":
    unittest.main()
