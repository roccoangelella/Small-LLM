"""Static regression guards for the Beam training adapter."""

from __future__ import annotations

import ast
import importlib.util
import os
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BEAM = ROOT / "beam"
LAUNCH = BEAM / "launch.py"
BEAMIGNORE = ROOT / ".beamignore"


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

    def test_beam_sync_excludes_local_credentials(self) -> None:
        from beta9.vendor.pathspec import PathSpec

        patterns = [
            line.strip()
            for line in BEAMIGNORE.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        spec = PathSpec.from_lines("gitwildmatch", patterns)
        for credential_path in (
            ".env",
            ".env.local",
            "training.env",
            ".secrets/google-drive-authorized-user.json",
            ".secrets/google-drive-oauth-client.json",
            "client_secret_example.apps.googleusercontent.com.json",
            "authorized_user.json",
            "credentials.json",
            "token.json",
            "private-key.pem",
        ):
            self.assertTrue(spec.match_file(credential_path), credential_path)

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

    def test_hf_checkpoint_transport_defaults_to_bucket_latest_and_dedicated_best(self) -> None:
        adapter = (BEAM / "model_repo_checkpoint.py").read_text(encoding="utf-8")
        runtime = (BEAM / "runtime.py").read_text(encoding="utf-8")
        self.assertIn('"--remote-checkpoint-bucket"', runtime)
        self.assertIn('"modal-hf-bucket-checkpoint-v1"', runtime)
        self.assertIn('"--best-model-repo"', adapter)
        self.assertIn('"--best-model-recreate"', adapter)
        self.assertNotIn('rewritten.append("--remote-checkpoint-repo")', adapter)

    def test_beam_adapter_keeps_bucket_flags_in_the_actual_command(self) -> None:
        code = """
import os
os.environ["SMALL_LLM_HF_REPO_ID"] = "owner/base"
import model_repo_checkpoint as transport
transport._ORIGINAL_TRAINER_COMMAND = lambda *args, **kwargs: [
    "python", "-m", "trainer",
    "--remote-checkpoint-bucket", "owner/base-checkpoints",
    "--remote-create-bucket", "--remote-rolling-latest-only",
]
command = transport._trainer_command_split_store(
    online=True,
    wandb_run_id="beam-run-001",
)
print("\\n".join(command))
"""
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join((str(BEAM), str(ROOT)))
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        command = result.stdout.splitlines()
        self.assertIn("--remote-checkpoint-bucket", command)
        self.assertNotIn("--remote-checkpoint-repo", command)
        self.assertEqual(
            command[command.index("--best-model-repo") + 1],
            "owner/base-best-beam-run-001",
        )
        self.assertIn("--best-model-recreate", command)

    def test_deep_decay_cpu_gate_migrates_legacy_latest_before_gpu(self) -> None:
        source = (BEAM / "deep_decay_10b_from_15500.py").read_text(encoding="utf-8")
        self.assertIn("runtime_base._hf_bucket_store()", source)
        self.assertIn("runtime_base._hf_model_repo_store()", source)
        self.assertIn('source="legacy_hf_model_repo"', source)
        self.assertIn("publisher.publish(", source)
        self.assertIn("bucket_store.prune_run_checkpoints(", source)
        self.assertIn("durable_pointer = bucket_store.read_json", source)


if __name__ == "__main__":
    unittest.main()
