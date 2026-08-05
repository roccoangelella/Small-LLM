"""Network-free regression tests for the Kaggle W&B startup timeout."""

from __future__ import annotations

import os
import runpy
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ENTRYPOINT = Path(__file__).resolve().parents[1] / "kaggle" / "run_20m_100m.py"


class KaggleWandbTimeoutTests(unittest.TestCase):
    def test_entrypoint_forces_ten_minute_timeout_before_launcher_setup(self) -> None:
        observed: dict[str, str | None] = {}

        common = types.ModuleType("run_20m_one_click")
        experiment = types.ModuleType("run_20m_100m_data_scaling")
        experiment.main = lambda: 0
        console = types.ModuleType("run_20m_100m_console")

        def install_common_console(module: object) -> None:
            self.assertIs(module, common)
            observed["common"] = os.environ.get("WANDB_INIT_TIMEOUT")

        def install_experiment_console(module: object) -> None:
            self.assertIs(module, experiment)
            observed["experiment"] = os.environ.get("WANDB_INIT_TIMEOUT")

        console.install_common_console = install_common_console
        console.install_experiment_console = install_experiment_console

        with (
            patch.dict(os.environ, {"WANDB_INIT_TIMEOUT": "90"}, clear=False),
            patch.dict(
                sys.modules,
                {
                    "run_20m_one_click": common,
                    "run_20m_100m_console": console,
                    "run_20m_100m_data_scaling": experiment,
                },
            ),
        ):
            namespace = runpy.run_path(str(ENTRYPOINT), run_name="wandb_timeout_test")
            self.assertEqual(os.environ["WANDB_INIT_TIMEOUT"], "600")

        self.assertEqual(namespace["WANDB_INIT_TIMEOUT_SECONDS"], "600")
        self.assertEqual(observed, {"common": "600", "experiment": "600"})


if __name__ == "__main__":
    unittest.main()
