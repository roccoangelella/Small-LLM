"""Regression guards for Beam distributed-volume startup handling."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCH = ROOT / "beam" / "launch.py"


class BeamVolumePreflightTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = LAUNCH.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source, filename=str(LAUNCH))

    def _function_segment(self, name: str) -> str:
        node = next(
            node
            for node in self.tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
        )
        return ast.get_source_segment(self.source, node) or ""

    def test_volume_retry_is_bounded_and_eagain_only(self) -> None:
        segment = self._function_segment("_retry_transient_volume_io")
        self.assertIn("timeout_seconds: float = 60.0", segment)
        self.assertIn("error.errno != errno.EAGAIN", segment)
        self.assertIn("time.monotonic() >= deadline", segment)
        self.assertIn("time.sleep(1.0)", segment)

    def test_import_preflight_still_proves_run_volume_writeability(self) -> None:
        segment = self._function_segment("remote_import_preflight")
        self.assertIn(
            "_retry_transient_volume_io(lambda: ensure_safe_directory(RUN_ROOT))",
            segment,
        )
        self.assertIn(
            "_retry_transient_volume_io(lambda: ensure_safe_directory(probe))",
            segment,
        )
        self.assertIn("_retry_transient_volume_io(probe.rmdir)", segment)

    def test_direct_remote_calls_fail_closed_on_missing_results(self) -> None:
        helper = self._function_segment("_require_remote_mapping")
        main = self._function_segment("main")
        worker = self._function_segment("_start_remote_thread")
        self.assertIn("if result is None", helper)
        self.assertIn("beam task list --filter status=error", helper)
        self.assertIn("remote_import_preflight.remote()", main)
        self.assertIn("label=\"import preflight\"", main)
        self.assertIn("label=\"dataset visibility\"", main)
        self.assertIn("label=\"training\"", main)
        self.assertIn("_require_remote_mapping(call(*args), label=label)", worker)


if __name__ == "__main__":
    unittest.main()
