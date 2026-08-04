"""CLI regression test for final validation on a non-evaluation boundary."""

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
        self.evaluations: list[int] = []

    def evaluate(self, batches, *, maximum_batches=None):
        list(batches)
        self.evaluations.append(self.global_step)
        return {
            "loss": 1.0,
            "perplexity": 2.0,
            "target_tokens": 4,
            "blocks": maximum_batches or 1,
        }


class _Session:
    def __init__(self, engine: _Engine) -> None:
        self.engine = engine
        self.saved_validation: list[object] = []

    def step(self):
        self.engine.global_step += 1
        return _Metrics(self.engine.global_step)

    def save_checkpoint(self, coordinator, checkpoint_id: str, **kwargs):
        self.saved_validation.append(kwargs.get("validation_metrics"))
        return f"/tmp/{checkpoint_id}"


class _ValidationReader:
    def iter_from_start(self, maximum_blocks=None):
        return iter((object(),))


class TrainerFinalValidationTests(unittest.TestCase):
    def test_non_boundary_final_step_is_validated_before_checkpoint(self) -> None:
        engine = _Engine()
        session = _Session(engine)
        trainer_config = SimpleNamespace(
            evaluation_every_steps=2,
            checkpoint_every_steps=0,
        )
        setup_result = (object(), trainer_config, engine, session, object())
        argv = [
            "--dataset-dir",
            "/tmp/data",
            "--checkpoint-dir",
            "/tmp/checkpoints",
            "--steps",
            "3",
            "--evaluation-every-steps",
            "2",
            "--validation-blocks",
            "1",
        ]
        with (
            patch("trainer.cli.setup", return_value=setup_result),
            patch("trainer.cli.validation_reader", return_value=_ValidationReader()),
            patch("trainer.cli.configure_remote_publication", return_value=None),
            patch("trainer.cli.configure_wandb", return_value=None),
            patch("trainer.cli.torch.cuda.is_available", return_value=False),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(main(argv), 0)

        self.assertEqual(engine.evaluations, [2, 3])
        self.assertEqual(len(session.saved_validation), 1)
        self.assertEqual(session.saved_validation[0]["loss"], 1.0)


if __name__ == "__main__":
    unittest.main()
