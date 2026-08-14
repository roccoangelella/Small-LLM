"""Regression guards for the Beam startup microbatch qualification."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BEAM = ROOT / "beam"


def _load_profiles():
    spec = importlib.util.spec_from_file_location("small_llm_beam_probe_profiles", BEAM / "profiles.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class BeamProbeTest(unittest.TestCase):
    def test_probe_candidates_are_8_12_16(self) -> None:
        profiles = _load_profiles()
        self.assertEqual(profiles.MICROBATCH_CANDIDATES, (8, 12, 16))
        self.assertEqual(profiles.SEQUENCES_PER_BLOCK, 64)

    def test_probe_measures_throughput_and_memory(self) -> None:
        source = (BEAM / "runtime.py").read_text(encoding="utf-8")
        self.assertIn('"tokens_per_second"', source)
        self.assertIn("median_tokens_per_second", source)
        self.assertIn("peak_reserved_memory_bytes", source)
        self.assertIn("fastest_safe_measured_candidate", source)

    def test_triton_cache_uses_container_local_scratch(self) -> None:
        launch_source = (BEAM / "launch.py").read_text(encoding="utf-8")
        runtime_source = (BEAM / "runtime.py").read_text(encoding="utf-8")
        self.assertIn('"TRITON_CACHE_DIR": "/tmp/small-llm-triton-cache"', launch_source)
        self.assertIn(
            'configured_triton_cache = os.environ.get("TRITON_CACHE_DIR", "").strip()',
            runtime_source,
        )
        self.assertIn("Path(configured_triton_cache)", runtime_source)

    def test_beam_checkpointing_skips_posix_fsync(self) -> None:
        launch_source = (BEAM / "launch.py").read_text(encoding="utf-8")
        runtime_source = (BEAM / "runtime.py").read_text(encoding="utf-8")
        transport_source = (BEAM / "model_repo_checkpoint.py").read_text(encoding="utf-8")
        self.assertIn('"SMALL_LLM_CHECKPOINT_FSYNC": "0"', launch_source)
        self.assertIn(
            '"42b0376511ba1fc7ceecfbbafbeae2027530fc2d"',
            launch_source,
        )
        self.assertIn("resume_parent_source_commit = existing_source_commit", runtime_source)
        self.assertIn("actual_source == migration_parent", transport_source)
        self.assertIn("compatible[\"source_commit\"] = actual_source", transport_source)

    def test_vps_preseed_guard_can_import_before_working_directory_is_added(self) -> None:
        environment = os.environ.copy()
        environment.update(
            PYTHONPATH=str(BEAM / "vps_site"),
            SMALL_LLM_DATASET_REQUIRE_PRESEEDED="1",
        )
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import dataset.incremental_frontier as frontier; "
                    "print(frontier._download_verified.__name__)"
                ),
            ],
            cwd="/tmp",
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "_preseeded_only")


if __name__ == "__main__":
    unittest.main()
