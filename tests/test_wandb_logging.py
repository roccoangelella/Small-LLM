"""Network-free tests for optional Weights & Biases telemetry."""

from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from trainer.wandb_logging import configure_wandb


class _Run:
    def __init__(self) -> None:
        self.id = "fake-run-id"
        self.defined: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.logged: list[dict[str, object]] = []
        self.finished: list[int] = []

    def define_metric(self, *args, **kwargs) -> None:
        self.defined.append((args, kwargs))

    def log(self, value) -> None:
        self.logged.append(dict(value))

    def finish(self, *, exit_code: int) -> None:
        self.finished.append(exit_code)


class _Wandb(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("wandb")
        self.run = _Run()
        self.init_kwargs: dict[str, object] | None = None

    def init(self, **kwargs):
        self.init_kwargs = dict(kwargs)
        return self.run


class _Metrics:
    def as_dict(self) -> dict[str, object]:
        return {
            "step": 3,
            "loss": 1.25,
            "gradient_clipped": True,
            "optimizer_gradient_norms": {
                "muon": 2.0,
                "adamw_decay": 1.0,
            },
        }


def _args(root: Path, **overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "wandb_mode": "online",
        "wandb_project": "Small-LLM",
        "wandb_entity": None,
        "wandb_run_name": "qualification-test",
        "wandb_run_id": None,
        "wandb_resume": "never",
        "wandb_tags": ("20m", "t4"),
        "wandb_dir": root / "wandb-local",
        "checkpoint_dir": root / "checkpoints",
        "dataset_manifest": None,
        "remote_drive_manifest": None,
        "dataset_dir": root / "dataset",
        "remote_token_env": "HF_TOKEN",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class WandbLoggingTests(unittest.TestCase):
    def test_configuration_uses_small_llm_project_and_never_logs_api_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake = _Wandb()
            with (
                patch.dict(sys.modules, {"wandb": fake}),
                patch.dict("os.environ", {"WANDB_API_KEY": "secret-value"}, clear=False),
                patch("trainer.wandb_logging._git_commit", return_value="a" * 40),
            ):
                telemetry = configure_wandb(
                    _args(root),
                    model_config={"architecture": "gdn2_hybrid"},
                    trainer_config={"precision": "fp16"},
                )

            self.assertIsNotNone(telemetry)
            self.assertEqual(fake.init_kwargs["project"], "Small-LLM")
            self.assertEqual(fake.init_kwargs["mode"], "online")
            self.assertNotIn("secret-value", repr(fake.init_kwargs))
            self.assertTrue((root / "wandb-local").is_dir())
            self.assertIn((('trainer/global_step',), {}), fake.run.defined)

    def test_training_metrics_are_flattened_on_optimizer_step_axis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fake = _Wandb()
            with patch.dict(sys.modules, {"wandb": fake}):
                telemetry = configure_wandb(
                    _args(Path(tmp)),
                    model_config={},
                    trainer_config={},
                )
            assert telemetry is not None
            telemetry.log_training(_Metrics())
            telemetry.log_validation(
                step=3,
                metrics={"loss": 1.1, "target_tokens": 128},
                elapsed_seconds=2.0,
            )
            telemetry.finish(exit_code=0)

            training = fake.run.logged[0]
            self.assertEqual(training["trainer/global_step"], 3)
            self.assertEqual(training["train/loss"], 1.25)
            self.assertEqual(training["train/optimizer_gradient_norms/muon"], 2.0)
            self.assertEqual(fake.run.logged[1]["validation/loss"], 1.1)
            self.assertEqual(fake.run.finished, [0])

    def test_disabled_mode_does_not_import_or_initialize_wandb(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            telemetry = configure_wandb(
                _args(Path(tmp), wandb_mode="disabled"),
                model_config={},
                trainer_config={},
            )
        self.assertIsNone(telemetry)

    def test_non_never_resume_requires_explicit_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fake = _Wandb()
            with patch.dict(sys.modules, {"wandb": fake}):
                with self.assertRaisesRegex(SystemExit, "--wandb-run-id"):
                    configure_wandb(
                        _args(Path(tmp), wandb_resume="must"),
                        model_config={},
                        trainer_config={},
                    )


if __name__ == "__main__":
    unittest.main()
