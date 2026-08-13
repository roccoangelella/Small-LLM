"""Static regression guards for the Beam training adapter."""

from __future__ import annotations

import ast
import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BEAM = ROOT / "beam"
LAUNCH = BEAM / "launch.py"


def _load_profiles():
    spec = importlib.util.spec_from_file_location("small_llm_beam_profiles", BEAM / "profiles.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class BeamLaunchTest(unittest.TestCase):
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

    def test_beam_adapter_files_exist(self) -> None:
        for name in (
            "launch.py",
            "profiles.py",
            "runtime.py",
            "model_repo_checkpoint.py",
            "rolling_dataset.py",
            "rolling_producer.py",
            "README.md",
        ):
            self.assertTrue((BEAM / name).is_file(), name)

    def test_current_beam_gpu_set_and_serverless_default(self) -> None:
        profiles = _load_profiles()
        self.assertEqual(profiles.DEFAULT_GPU, "RTX5090")
        self.assertEqual(profiles.SUPPORTED_GPUS, frozenset({"RTX5090", "RTX4090", "A10G"}))
        self.assertNotIn("H100", profiles.SUPPORTED_GPUS)
        self.assertEqual(profiles.SEQUENCES_PER_BLOCK, 64)
        self.assertEqual(profiles.MICROBATCH_CANDIDATES, (8, 12, 16))

    def test_gpu_functions_are_explicitly_pinned(self) -> None:
        for function_name, gpu in (
            ("train_rtx5090_remote", "RTX5090"),
            ("train_rtx4090_remote", "RTX4090"),
            ("train_a10g_remote", "A10G"),
        ):
            node = next(
                node
                for node in self.tree.body
                if isinstance(node, ast.FunctionDef) and node.name == function_name
            )
            decorator = ast.get_source_segment(self.source, node.decorator_list[0]) or ""
            self.assertIn(f'gpu="{gpu}"', decorator)

    def test_incremental_producer_is_headless_cpu_work(self) -> None:
        node = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "produce_rolling_dataset_remote"
        )
        decorator = ast.get_source_segment(self.source, node.decorator_list[0]) or ""
        self.assertIn("headless=True", decorator)
        self.assertNotIn("gpu=", decorator)

    def test_cpu_gates_precede_gpu_dispatch(self) -> None:
        segment = self._function_segment("main")
        preflight = segment.find("remote_import_preflight.remote()")
        stage = segment.find("_stage_with_incremental_producer(")
        visibility = segment.find("verify_staged_dataset_visible_remote.remote(")
        dispatch = segment.find("gpu_function.remote(")
        self.assertGreaterEqual(preflight, 0)
        self.assertGreater(stage, preflight)
        self.assertGreater(visibility, stage)
        self.assertGreater(dispatch, visibility)

    def test_beam_images_split_blackwell_from_older_serverless_gpus(self) -> None:
        self.assertIn("nvidia/cuda:12.8.1-devel-ubuntu24.04", self.source)
        self.assertIn('torch_index="cu128"', self.source)
        self.assertIn("nvidia/cuda:12.4.1-devel-ubuntu22.04", self.source)
        self.assertIn('torch_index="cu126"', self.source)
        self.assertIn("torch==2.10.0", self.source)
        self.assertIn("fla-core==0.5.2", self.source)
        decorators: dict[str, str] = {}
        for function_name in ("train_rtx5090_remote", "train_rtx4090_remote", "train_a10g_remote"):
            node = next(
                node
                for node in self.tree.body
                if isinstance(node, ast.FunctionDef) and node.name == function_name
            )
            decorators[function_name] = ast.get_source_segment(self.source, node.decorator_list[0]) or ""
        self.assertIn("BLACKWELL_IMAGE", decorators["train_rtx5090_remote"])
        self.assertIn("LEGACY_SERVERLESS_IMAGE", decorators["train_rtx4090_remote"])
        self.assertIn("LEGACY_SERVERLESS_IMAGE", decorators["train_a10g_remote"])

    def test_wandb_provider_tag_is_beam(self) -> None:
        runtime = (BEAM / "runtime.py").read_text(encoding="utf-8")
        self.assertIn('f"{tokens.label.lower()}-tokens",\n        "beam",', runtime)

    def test_hf_checkpoint_schema_remains_cross_provider_compatible(self) -> None:
        adapter = (BEAM / "model_repo_checkpoint.py").read_text(encoding="utf-8")
        self.assertIn('expected_transport="modal-hf-checkpoint-v1"', adapter)
        self.assertIn('payload["transport"] = "modal-hf-checkpoint-v1"', adapter)


if __name__ == "__main__":
    unittest.main()
