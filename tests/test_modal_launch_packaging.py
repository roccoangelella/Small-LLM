"""Static regression guards for Modal launcher source packaging and GPU dispatch."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCH_PATH = REPO_ROOT / "modal" / "launch.py"


class ModalLaunchPackagingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = LAUNCH_PATH.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source, filename=str(LAUNCH_PATH))

    def test_profiles_is_explicitly_packaged_for_remote_import(self) -> None:
        found = False
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != "add_local_python_source" or not node.args:
                continue
            first = node.args[0]
            if isinstance(first, ast.Constant) and first.value == "profiles":
                found = True
                break
        self.assertTrue(found, "modal/launch.py must explicitly package profiles.py")

    def test_cpu_import_preflight_precedes_h100_spawn(self) -> None:
        main = next(
            node
            for node in self.tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "main"
        )
        segment = ast.get_source_segment(self.source, main) or ""
        preflight = segment.find("remote_import_preflight.remote()")
        gpu_spawn = segment.find("train_remote.with_options(gpu=gpu).spawn(")
        self.assertGreaterEqual(preflight, 0, "main must execute the CPU remote import preflight")
        self.assertGreater(gpu_spawn, preflight, "H100 spawn must happen only after CPU import preflight")

    def test_local_source_mounts_are_last_image_mutations(self) -> None:
        image_assignment = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "IMAGE" for target in node.targets)
        )
        segment = ast.get_source_segment(self.source, image_assignment) or ""
        env_pos = segment.find(".env(")
        python_source_pos = segment.find(".add_local_python_source(")
        local_dir_pos = segment.find(".add_local_dir(")
        self.assertGreater(python_source_pos, env_pos)
        self.assertGreater(local_dir_pos, python_source_pos)


if __name__ == "__main__":
    unittest.main()
