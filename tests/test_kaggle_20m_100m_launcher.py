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

    def test_fresh_command_can_recover_a_run_created_before_init_timeout(self) -> None:
        command = launcher.trainer_command(
            "uv",
            Path("/data"),
            _plan(),
            Path("/checkpoints"),
            additional_steps=12,
            microbatch=4,
            online=True,
        )
        self.assertNotIn("--resume", command)
        self.assertEqual(command[command.index("--wandb-resume") + 1], "allow")
        self.assertEqual(
            command[command.index("--wandb-run-id") + 1],
            launcher.WANDB_RUN_ID,
        )

    def test_preflight_pins_exact_runtime_and_clean_production_run_id(self) -> None:
        command, root, result = launcher.wandb_preflight_command(
            "uv", Path("/evidence"), "explicit-entity"
        )
        self.assertEqual(launcher.WANDB_RUN_ID, "20m-100m-data-003")
        self.assertEqual(command[command.index("--python") + 1], "3.13")
        self.assertEqual(command[command.index("--with") + 1], "wandb==0.26.1")
        self.assertEqual(command[command.index("--run-id") + 1], launcher.WANDB_RUN_ID)
        self.assertEqual(command[command.index("--init-timeout") + 1], "30")
        self.assertEqual(command[command.index("--entity") + 1], "explicit-entity")
        self.assertEqual(root, Path("/evidence/wandb-preflight"))
        self.assertEqual(result, root / "result.json")

    def test_preflight_result_requires_all_phases_and_preserved_logs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            preserved = root / "preserved"
            preserved.mkdir()
            debug_logs: dict[str, object] = {}
            for name in ("debug.log", "debug-internal.log", "debug-core.log"):
                path = preserved / name
                path.write_text(name, encoding="utf-8")
                debug_logs[name] = {"path": str(path)}
            phases = [
                {"name": name, "status": "passed", "elapsed_seconds": 0.1}
                for name in (
                    "secret_propagation",
                    "dns",
                    "tls",
                    "api_key_authentication",
                    "local_wandb_core",
                    "project_run_resume",
                )
            ]
            result_path = root / "result.json"
            result_path.write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "run_id": launcher.WANDB_RUN_ID,
                        "init_timeout_seconds": 30,
                        "phases": phases,
                        "debug_logs": debug_logs,
                    }
                ),
                encoding="utf-8",
            )
            validated = launcher.validate_wandb_preflight_result(result_path)
            self.assertEqual(validated["status"], "passed")

            phases[-1]["elapsed_seconds"] = 30.1
            result_path.write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "run_id": launcher.WANDB_RUN_ID,
                        "init_timeout_seconds": 30,
                        "phases": phases,
                        "debug_logs": debug_logs,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(launcher.LaunchFailure, "not healthy"):
                launcher.validate_wandb_preflight_result(result_path)

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
