"""Offline tests for the 20M-model/1B-token Kaggle profile overlay."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

KAGGLE_DIR = Path(__file__).resolve().parents[1] / "kaggle"
if str(KAGGLE_DIR) not in sys.path:
    sys.path.insert(0, str(KAGGLE_DIR))

import run_20m_100m_data_scaling as launcher_100m  # noqa: E402
import run_20m_1b_data_scaling as launcher  # noqa: E402


def _plan() -> dict[str, object]:
    return {
        "trainer": {
            "warmup_tokens": 10,
            "stable_tokens": 20,
            "decay_tokens": 30,
            "validation_blocks": 4,
        }
    }


class Kaggle20M1BProfileTests(unittest.TestCase):
    def test_profile_does_not_mutate_100m_module(self) -> None:
        self.assertEqual(launcher_100m.DATASET_RUN_ID, "20m-100m-dataset-001")
        self.assertEqual(launcher_100m.PROFILE, "20m-100m-data-scaling-v1")
        self.assertEqual(launcher.DATASET_RUN_ID, "20m-1b-dataset-001")
        self.assertEqual(launcher.PROFILE, "20m-1b-data-scaling-v1")

    def test_1b_manifest_matches_and_500m_manifest_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "train").mkdir()
            (root / "validation").mkdir()
            (root / "drive_manifest.json").write_text("{}", encoding="utf-8")
            manifest = {
                "schema_version": 2,
                "sequence_format": "context_plus_one",
                "context_length": 2048,
                "stored_tokens_per_sequence": 2049,
                "sequences_per_block": 16,
                "target_shard_bytes": 8 * 1024 * 1024,
                "production": {
                    "run_id": "20m-1b-dataset-001",
                    "target_source_tokens": 1_000_000_000,
                    "minimum_source_tokens": 900_000_000,
                    "maximum_source_tokens": 1_100_000_000,
                    "checkpoint_source_tokens": 40_000_000,
                    "target_reached": True,
                    "remote_required": True,
                },
            }
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            matched, _ = launcher.profile_match(root)
            self.assertTrue(matched)

            manifest["production"]["run_id"] = "20m-500m-dataset-001"
            manifest["production"]["target_source_tokens"] = 500_000_000
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            matched, _ = launcher.profile_match(root)
            self.assertFalse(matched)

    def test_trainer_command_uses_distinct_1b_telemetry_identity(self) -> None:
        command = launcher.trainer_command(
            "uv",
            Path("/data"),
            _plan(),
            Path("/checkpoints"),
            additional_steps=12,
            microbatch=4,
            online=True,
        )
        self.assertEqual(command[command.index("--microbatch-size") + 1], "4")
        self.assertEqual(
            command[command.index("--wandb-run-id") + 1],
            "20m-1b-data-001",
        )
        self.assertEqual(
            command[command.index("--wandb-run-name") + 1],
            "20M model on 1B tokens",
        )
        tags = command[command.index("--wandb-tags") + 1 :]
        self.assertIn("1b-tokens", tags)
        self.assertNotIn("100m-tokens", tags)

    def test_fresh_1b_run_skips_probes_and_selects_microbatch_four(self) -> None:
        verdict = launcher.qualify_microbatch("uv", Path("/data"), _plan(), {})
        self.assertIs(launcher.base.qualify_microbatch, launcher.qualify_microbatch)
        self.assertEqual(verdict["status"], "skipped_by_experiment_decision")
        self.assertEqual(verdict["selected_microbatch"], 4)
        self.assertEqual(verdict["probe_steps_executed"], 0)

    def test_preflight_uses_1b_run_name_and_id(self) -> None:
        command, _, _ = launcher.wandb_preflight_command(
            "uv", Path("/evidence"), "entity"
        )
        self.assertEqual(command[command.index("--run-id") + 1], "20m-1b-data-001")
        self.assertEqual(
            command[command.index("--run-name") + 1],
            "20M model on 1B tokens",
        )

    def test_runtime_configuration_propagates_to_private_base(self) -> None:
        launcher.configure_runtime(
            durability_every=250,
            max_steps_per_session=sys.maxsize,
            wandb_run_id="20m-1b-data-001",
        )
        self.assertEqual(launcher.base.LOCAL_EVERY, 250)
        self.assertEqual(launcher.base.EVAL_EVERY, 250)
        self.assertEqual(launcher.base.REMOTE_EVERY, 250)
        self.assertEqual(launcher.base.MAX_STEPS_PER_SESSION, sys.maxsize)


if __name__ == "__main__":
    unittest.main()
