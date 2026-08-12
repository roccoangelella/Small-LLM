"""CPU-only contract tests for the Kaggle dual-T4 qualification harness."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "kaggle" / "qualify_dual_t4.py"
SPEC = importlib.util.spec_from_file_location("qualify_dual_t4", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
qualification = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = qualification
SPEC.loader.exec_module(qualification)


class DualT4QualificationContractTests(unittest.TestCase):
    def test_dataset_match_requires_exact_2b_schema_v2_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "train").mkdir()
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "sequence_format": "context_plus_one",
                        "context_length": 2048,
                        "sequences_per_block": 16,
                        "production": {"run_id": "20m-2b-dataset-001"},
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(qualification._dataset_matches(root))
            payload = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            payload["production"]["run_id"] = "wrong"
            (root / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
            self.assertFalse(qualification._dataset_matches(root))

    def test_row_comparison_detects_loss_and_gradient_deltas(self) -> None:
        single = [
            {"block_id": 0, "target_tokens": 32768, "warmup": False, "loss": 3.0, "gradient_norm": 2.0},
            {"block_id": 1, "target_tokens": 32768, "warmup": False, "loss": 2.9, "gradient_norm": 1.0},
        ]
        ddp = [
            {"block_id": 0, "target_tokens": 32768, "warmup": False, "loss": 3.0005, "gradient_norm": 2.002},
            {"block_id": 1, "target_tokens": 32768, "warmup": False, "loss": 2.8998, "gradient_norm": 0.999},
        ]
        result = qualification._compare_rows(single, ddp)
        self.assertAlmostEqual(result["maximum_loss_delta"], 0.0005)
        self.assertAlmostEqual(result["maximum_gradient_relative_delta"], 0.001)

    def test_throughput_excludes_warmup(self) -> None:
        rows = [
            {"warmup": True, "tokens_per_second": 1.0},
            {"warmup": False, "tokens_per_second": 10.0},
            {"warmup": False, "tokens_per_second": 20.0},
        ]
        summary = qualification._throughput(rows)
        self.assertEqual(summary["median_tokens_per_second"], 15.0)
        self.assertEqual(summary["mean_tokens_per_second"], 15.0)

    def test_ddp_worker_command_uses_two_processes(self) -> None:
        args = qualification.build_parser().parse_args([])
        with mock.patch.object(qualification, "resolve_dataset", return_value=Path("/tmp/data")):
            command = qualification._worker_command(args, "ddp", Path("/tmp/out.pt"))
        self.assertIn("torch.distributed.run", command)
        self.assertIn("--nproc-per-node=2", command)
        self.assertIn("--worker", command)
        self.assertIn("ddp", command)

    def test_canonical_launcher_exposes_qualification_dry_run(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "kaggle/launch.py",
                "qualify-dual-t4",
                "--model",
                "20M",
                "--tokens",
                "2B",
                "--measure-blocks",
                "3",
                "--minimum-speedup",
                "1.5",
                "--dry-run",
            ],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["action"], "qualify-dual-t4")
        self.assertEqual(payload["runtime"], "kaggle/qualify_dual_t4.py")
        self.assertEqual(payload["dataset_run_id"], "20m-2b-dataset-001")
        self.assertEqual(payload["resume"], "not_applicable")
        self.assertEqual(payload["arguments"]["measure_blocks"], 3)
        self.assertEqual(payload["arguments"]["minimum_speedup"], 1.5)


if __name__ == "__main__":
    unittest.main()
