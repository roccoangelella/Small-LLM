"""Network-free regression test for the Kaggle W&B startup timeout."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path


ENTRYPOINT = Path(__file__).resolve().parents[1] / "kaggle" / "run_20m_100m.py"


class KaggleWandbTimeoutTests(unittest.TestCase):
    def test_entrypoint_sets_five_minute_wandb_init_timeout_before_launcher_imports(self) -> None:
        source = ENTRYPOINT.read_text(encoding="utf-8")
        tree = ast.parse(source)
        assignments = {
            node.targets[0].id: ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        }
        self.assertEqual(assignments["WANDB_INIT_TIMEOUT_SECONDS"], "300")
        timeout_position = source.index('os.environ.setdefault("WANDB_INIT_TIMEOUT"')
        launcher_import_position = source.index("import run_20m_one_click as common")
        self.assertLess(timeout_position, launcher_import_position)


if __name__ == "__main__":
    unittest.main()
