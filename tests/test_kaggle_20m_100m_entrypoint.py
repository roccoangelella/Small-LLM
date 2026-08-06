"""Offline tests for the pinned 20M/100M entry-point overrides."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

KAGGLE_DIR = Path(__file__).resolve().parents[1] / "kaggle"
if str(KAGGLE_DIR) not in sys.path:
    sys.path.insert(0, str(KAGGLE_DIR))

import run_20m_100m as entrypoint  # noqa: E402


class Kaggle20M100MEntrypointTests(unittest.TestCase):
    def test_full_run_and_250_step_durability_are_installed(self) -> None:
        self.assertEqual(entrypoint.experiment.LOCAL_EVERY, 250)
        self.assertEqual(entrypoint.experiment.EVAL_EVERY, 250)
        self.assertEqual(entrypoint.experiment.REMOTE_EVERY, 250)
        self.assertGreater(entrypoint.experiment.MAX_STEPS_PER_SESSION, 3_052)
        self.assertEqual(entrypoint.experiment.WANDB_RUN_ID, "20m-100m-data-004")
        self.assertEqual(
            os.environ["PYTORCH_CUDA_ALLOC_CONF"],
            "expandable_segments:True",
        )


if __name__ == "__main__":
    unittest.main()
