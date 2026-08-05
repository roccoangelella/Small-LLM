"""Offline tests for the 20M-model/100M-token Kaggle launcher."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

KAGGLE_DIR = Path(__file__).resolve().parents[1] / "kaggle"
if str(KAGGLE_DIR) not in sys.path:
    sys.path.insert(0, str(KAGGLE_DIR))

import run_20m_100m_data_scaling as launcher  # noqa: E402


def _plan() -> dict[str, object]:
    return {
        "trainer": {
            "warmup_tokens": 10,
            "stable_tokens": 20,
            "decay_tokens": 30,
            "validation_blocks": 4,
        }
    }


def _rows(
    tokens_per_second: float,
    *,
    loss_shift: float = 0.0,
    gradient_scale: float = 1.0,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for step in range(1, 9):
        rows.append(
            {
                "step": step,
                "block_id": step - 1,
                "loss": 10.0 - step * 0.1 + loss_shift,
                "gradient_norm": (1.0 + step * 0.01) * gradient_scale,
                "tokens_per_second": tokens_per_second,
                "grad_scaler_scale": 65_536.0,
                "learning_rate": step * 1e-5,
                "sequences": 16,
                "target_tokens": 32_768,
                "consumed_tokens": step * 32_768,
                "overflow_retries": 0,
                "overflow_events_total": 0,
                "peak_memory_bytes": 100,
                "peak_reserved_memory_bytes": 200,
                "gradient_clipped": True,
            }
        )
    return rows


class Kaggle20M100MLauncherTests(unittest.TestCase):
    def test_session_plan_is_bounded_and_avoids_periodic_boundary(self) -> None:
        self.assertEqual(
            launcher.segment_plan(0, 3_052, 749)["expected_final_step"],
            749,
        )
        self.assertEqual(
            launcher.segment_plan(751, 3_052, 749)["expected_final_step"],
            1_499,
        )
        self.assertTrue(
            launcher.segment_plan(3_000, 3_052, 749)["complete_after_session"]
        )

    def test_resume_command_keeps_microbatch_and_wandb_identity(self) -> None:
        command = launcher.trainer_command(
            "uv",
            Path("/data"),
            _plan(),
            Path("/checkpoints"),
            additional_steps=12,
            microbatch=4,
            online=True,
            resume="step-00000749",
        )
        self.assertEqual(command[command.index("--steps") + 1], "12")
        self.assertEqual(command[command.index("--microbatch-size") + 1], "4")
        self.assertEqual(command[command.index("--resume") + 1], "step-00000749")
        self.assertEqual(command[command.index("--wandb-resume") + 1], "must")
        self.assertEqual(
            command[command.index("--wandb-run-id") + 1],
            launcher.WANDB_RUN_ID,
        )

    def test_microbatch_gate_accepts_safe_speedup(self) -> None:
        verdict = launcher.compare_probes(
            _rows(100.0),
            _rows(120.0, loss_shift=0.001, gradient_scale=1.001),
            10_000,
        )
        self.assertEqual(verdict["status"], "passed")

    def test_microbatch_gate_rejects_insufficient_speedup(self) -> None:
        with self.assertRaises(launcher.LaunchFailure):
            launcher.compare_probes(_rows(100.0), _rows(101.0), 10_000)

    def test_segment_log_requires_explicit_final_remote_publication(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trainer.log"
            path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "remote_publication": {
                                    "checkpoint_id": "step-00000749",
                                    "final": True,
                                }
                            }
                        ),
                        json.dumps({"checkpoint_id": "step-00000749"}),
                    ]
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                launcher.verify_segment_log(path, 749),
                "step-00000749",
            )

    def test_dataset_profile_requires_20m_producer_checkpoint_cadence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "train").mkdir()
            (root / "validation").mkdir()
            (root / "drive_manifest.json").write_text("{}", encoding="utf-8")
            manifest = {
                "schema_version": 2,
                "sequence_format": "context_plus_one",
                "context_length": 2_048,
                "stored_tokens_per_sequence": 2_049,
                "sequences_per_block": 16,
                "target_shard_bytes": 8 * 1024 * 1024,
                "production": {
                    "run_id": launcher.DATASET_RUN_ID,
                    "target_source_tokens": 100_000_000,
                    "minimum_source_tokens": 90_000_000,
                    "maximum_source_tokens": 110_000_000,
                    "checkpoint_source_tokens": 20_000_000,
                    "target_reached": True,
                    "remote_required": True,
                },
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertTrue(launcher.profile_match(root)[0])

            manifest["production"]["checkpoint_source_tokens"] = 2_000_000
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertFalse(launcher.profile_match(root)[0])


if __name__ == "__main__":
    unittest.main()
