"""Regression tests for pre-checkpoint production resume semantics."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from dataset import config
from dataset.production.cli import _builder_resume_mode


class ProductionCliBootstrapResumeTests(unittest.TestCase):
    def test_work_plan_only_state_restarts_builder_from_empty_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / config.WORK_PLAN_FILENAME).write_text("{}", encoding="utf-8")
            self.assertFalse(_builder_resume_mode(root))

    def test_progress_state_uses_true_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / config.WORK_PLAN_FILENAME).write_text("{}", encoding="utf-8")
            (root / config.PROGRESS_FILENAME).write_text("{}", encoding="utf-8")
            self.assertTrue(_builder_resume_mode(root))

    def test_uncheckpointed_artifacts_without_progress_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / config.WORK_PLAN_FILENAME).write_text("{}", encoding="utf-8")
            (root / "train").mkdir()
            with self.assertRaisesRegex(RuntimeError, "unexpected pre-checkpoint artifacts"):
                _builder_resume_mode(root)


if __name__ == "__main__":
    unittest.main()
