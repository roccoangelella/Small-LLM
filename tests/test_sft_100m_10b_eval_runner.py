from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "kaggle" / "src" / "sft_100m_10b_eval_runner.py"


class SFT100M10BEvalRunnerTests(unittest.TestCase):
    def test_standalone_runner_bootstraps_repository_root(self) -> None:
        original_path = list(sys.path)
        try:
            normalized_root = ROOT.resolve()
            sys.path[:] = [
                entry
                for entry in sys.path
                if Path(entry or ".").resolve() != normalized_root
            ]
            spec = importlib.util.spec_from_file_location(
                "small_llm_sft_100m_10b_eval_runner_test",
                RUNNER,
            )
            self.assertIsNotNone(spec)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            self.assertEqual(Path(module.REPO).resolve(), normalized_root)
            self.assertIn(str(normalized_root), sys.path)

            from post_training.sft import eval_suite  # noqa: F401
        finally:
            sys.path[:] = original_path


if __name__ == "__main__":
    unittest.main()
