"""CLI-level tests for W&B step logging and failure finalization."""

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
    def __init__(self, engine: _Engine, *, fail_after: int | None = None) -> None:
        self.engine = engine
        self.fail_after = fail_after
        self.saved: list[str] = []

    def step(self) -> _Metrics:
        if self.fail_after is not None and self.engine.global_step >= self.fail_after:
            raise RuntimeError("training failed")
        self.engine.global_step += 1
        return _Metrics(self.engine.global_step)

    def save_checkpoint(self, coordinator, checkpoint_id: str, **kwargs) -> None:
        self.saved.append(checkpoint_id)


class _Telemetry:
    def __init__(self) -> None:
        self.training_steps: list[int] = []
        self.checkpoints: list[str] = []
        self.finished: list[int] = []

    def log_training(self, metrics: _Metrics) -> None:
        self.training_steps.append(metrics.step)

    def log_checkpoint(self, *, checkpoint_id: str, **kwargs) -> None:
        self.checkpoints.append(checkpoint_id)

    def finish(self, *, exit_code: int) -> None:
        self.finished.append(exit_code)


_BASE = [
    "--dataset-dir",
    "/tmp/data",
    "--checkpoint-dir",
    "/tmp/checkpoints",
    "--steps",
    "2",
    "--wandb-mode",
    "online",
]


class TrainerWandbCLITests(unittest.TestCase):
    def _setup(self, *, fail_after: int | None = None):
        engine = _Engine()
        session = _Session(engine, fail_after=fail_after)
        trainer_config = SimpleNamespace(
            evaluation_every_steps=0,
            checkpoint_every_steps=0,
        )
        return engine, session, (object(), trainer_config, engine, session, object())

    def test_success_logs_steps_checkpoint_and_clean_finish(self) -> None:
        engine, session, setup_result = self._setup()
        telemetry = _Telemetry()
        with (
            patch("trainer.cli.setup", return_value=setup_result),
            patch("trainer.cli.configure_remote_publication", return_value=None),
            patch("trainer.cli.configure_wandb", return_value=telemetry),
            patch("trainer.cli.torch.cuda.is_available", return_value=False),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(main(list(_BASE)), 0)

        self.assertEqual(engine.global_step, 2)
        self.assertEqual(telemetry.training_steps, [1, 2])
        self.assertEqual(telemetry.checkpoints, ["step-00000002"])
        self.assertEqual(telemetry.finished, [0])
        self.assertEqual(session.saved, ["step-00000002"])

    def test_training_failure_is_preserved_and_wandb_gets_failure_status(self) -> None:
        _, _, setup_result = self._setup(fail_after=1)
        telemetry = _Telemetry()
        with (
            patch("trainer.cli.setup", return_value=setup_result),
            patch("trainer.cli.configure_remote_publication", return_value=None),
            patch("trainer.cli.configure_wandb", return_value=telemetry),
            patch("trainer.cli.torch.cuda.is_available", return_value=False),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            with self.assertRaisesRegex(RuntimeError, "training failed"):
                main(list(_BASE))

        self.assertEqual(telemetry.training_steps, [1])
        self.assertEqual(telemetry.finished, [1])


if __name__ == "__main__":
    unittest.main()
