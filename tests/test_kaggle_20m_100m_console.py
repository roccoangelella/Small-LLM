"""Tests for concise 100M Kaggle console output."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

KAGGLE_DIR = Path(__file__).resolve().parents[1] / "kaggle"
if str(KAGGLE_DIR) not in sys.path:
    sys.path.insert(0, str(KAGGLE_DIR))

import run_20m_100m_console as console  # noqa: E402


class GateFailure(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Kaggle20M100MConsoleTests(unittest.TestCase):
    def test_training_metric_is_human_readable(self) -> None:
        formatter = console._Formatter(
            "trainer-00000001-00000749",
            ["python", "-m", "trainer", "--steps", "749"],
        )
        line = formatter.render(
            json.dumps(
                {
                    "step": 1,
                    "block_id": 0,
                    "loss": 8.123456,
                    "tokens_per_second": 18_543.2,
                    "learning_rate": 3e-5,
                    "gradient_norm": 1.234,
                    "gradient_clipped": True,
                    "peak_reserved_memory_bytes": 8 * 1024**3,
                    "overflow_events_total": 0,
                }
            )
        )
        assert line is not None
        self.assertIn("[train] 1/749", line)
        self.assertIn("loss 8.1235", line)
        self.assertIn("18.5k tok/s", line)
        self.assertIn("VRAM 8.0GiB", line)

    def test_console_formatting_preserves_raw_evidence_log(self) -> None:
        common = types.SimpleNamespace(
            check_environment=lambda: {"gpu": "Tesla T4"},
            sha256=_sha256,
            GateFailure=GateFailure,
        )
        console.install_common_console(common)
        payload = {
            "step": 1,
            "block_id": 0,
            "loss": 7.5,
            "tokens_per_second": 10_000,
        }
        code = f"import json; print(json.dumps({payload!r}))"
        with tempfile.TemporaryDirectory() as temporary:
            evidence = Path(temporary)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                result = common.run(
                    [sys.executable, "-c", code, "--steps", "1"],
                    name="trainer-00000001-00000001",
                    evidence=evidence,
                )
            self.assertEqual(result["exit_code"], 0)
            self.assertIn("[train] 1/1", output.getvalue())
            raw = (evidence / "trainer-00000001-00000001.log").read_text()
            self.assertEqual(json.loads(raw), payload)


if __name__ == "__main__":
    unittest.main()
