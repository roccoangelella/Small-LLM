"""Regression tests for qualification report stdout behavior."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from dataset import qualification


class QualificationReportOutputTests(unittest.TestCase):
    def _dataset_dir(self, root: Path) -> Path:
        dataset_dir = root / "dataset"
        dataset_dir.mkdir()
        (dataset_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
        return dataset_dir

    def test_output_file_suppresses_full_plan_stdout(self) -> None:
        plan = {
            "train": {"block_ids": list(range(10_000))},
            "trainer": {"steps": 10_000},
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = self._dataset_dir(root)
            output = root / "qualification-plan.json"
            stdout = io.StringIO()
            with patch("dataset.qualification.derive_plan", return_value=plan):
                with redirect_stdout(stdout):
                    code = qualification._run_report(
                        [
                            "--profile",
                            "modal-2b-b64",
                            "--dataset-dir",
                            str(dataset_dir),
                            "--output",
                            str(output),
                        ]
                    )

            self.assertEqual(code, 0)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), plan)
            rendered = stdout.getvalue()
            self.assertNotIn('"block_ids"', rendered)
            status = json.loads(rendered)
            self.assertEqual(status["qualification_report"], "written")
            self.assertEqual(status["profile"], "modal-2b-b64")
            self.assertEqual(status["output"], str(output))

    def test_without_output_keeps_interactive_full_json(self) -> None:
        plan = {"train": {"block_ids": [0, 1, 2]}, "trainer": {"steps": 3}}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = self._dataset_dir(root)
            stdout = io.StringIO()
            with patch("dataset.qualification.derive_plan", return_value=plan):
                with redirect_stdout(stdout):
                    code = qualification._run_report(
                        [
                            "--profile",
                            "modal-2b-b64",
                            "--dataset-dir",
                            str(dataset_dir),
                        ]
                    )

            self.assertEqual(code, 0)
            self.assertEqual(json.loads(stdout.getvalue()), plan)


if __name__ == "__main__":
    unittest.main()
