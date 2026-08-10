"""Tests for the single Kaggle launch front door."""
from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
KAGGLE = ROOT / "kaggle"
if str(KAGGLE) not in sys.path:
    sys.path.insert(0, str(KAGGLE))

import launch  # noqa: E402
import launch_sft  # noqa: E402
import runtime  # noqa: E402


class UnifiedKaggleLauncherTests(unittest.TestCase):
    def test_quantity_aliases_resolve_the_same_profile(self) -> None:
        self.assertEqual(launch.parse_quantity("20M"), 20_000_000)
        self.assertEqual(launch.parse_quantity("500m"), 500_000_000)
        self.assertEqual(launch.parse_quantity("2B"), 2_000_000_000)
        self.assertEqual(launch.parse_quantity("2000M"), 2_000_000_000)
        profile = runtime.resolve_profile(20_000_000, launch.parse_quantity("2000M"))
        self.assertEqual(profile.dataset_run_id, "20m-2b-dataset-001")
        self.assertEqual(
            profile.launch_commit,
            "3c920a7b682382181d4dc7557e217e6509d0dabe",
        )

    def test_sft_train_launch_log_matches_pretraining_exactly(self) -> None:
        argv = ["train", "--model", "20M", "--tokens", "500M"]

        pretraining_stdout = io.StringIO()
        with mock.patch.object(runtime, "train", return_value=0), redirect_stdout(
            pretraining_stdout
        ):
            self.assertEqual(launch.main(argv), 0)

        sft_stdout = io.StringIO()
        with mock.patch.object(launch_sft.sft_runtime, "train", return_value=0), redirect_stdout(
            sft_stdout
        ):
            self.assertEqual(launch_sft.main(argv), 0)

        expected = "[launch] action=train model=20M tokens=500M resume=automatic_verified\n"
        self.assertEqual(pretraining_stdout.getvalue(), expected)
        self.assertEqual(sft_stdout.getvalue(), expected)

    def test_train_dry_run_exposes_profile_contract(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "kaggle/launch.py",
                "train",
                "--model",
                "20M",
                "--tokens",
                "2B",
                "--max-steps-this-session",
                "250",
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
        self.assertEqual(payload["runtime"], "kaggle/runtime.py")
        self.assertEqual(payload["profile"], "20m-2b-data-scaling-v1")
        self.assertEqual(payload["dataset_run_id"], "20m-2b-dataset-001")
        self.assertEqual(payload["wandb_run_id"], "20m-2b-data-001")
        self.assertEqual(payload["arguments"], {"max_steps_this_session": 250})
        self.assertEqual(payload["resume"], "automatic_verified")

    def test_publish_dry_run_does_not_bootstrap_or_import_profile_overlay(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "kaggle/launch.py",
                "publish",
                "--model",
                "20M",
                "--tokens",
                "500M",
                "--dataset-dir",
                "/tmp/dataset",
                "--ops-dir",
                "/tmp/ops",
                "--force-upload",
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
        self.assertEqual(payload["runtime"], "kaggle/runtime.py")
        self.assertEqual(payload["profile"], "20m-500m-data-scaling-v1")
        self.assertEqual(
            payload["arguments"],
            {
                "dataset_dir": "/tmp/dataset",
                "force_upload": True,
                "ops_dir": "/tmp/ops",
            },
        )

    def test_resume_flag_is_rejected(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "kaggle/launch.py",
                "train",
                "--model",
                "20M",
                "--tokens",
                "2B",
                "--resume",
            ],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("Resume is fail-closed and automatic", result.stderr)

    def test_unsupported_profile_fails_before_runtime_execution(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "kaggle/launch.py",
                "train",
                "--model",
                "100M",
                "--tokens",
                "2B",
                "--dry-run",
            ],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("unsupported model/token profile", result.stderr)


if __name__ == "__main__":
    unittest.main()
