"""Tests for the unified Kaggle launch front door."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KAGGLE = ROOT / "kaggle"
if str(KAGGLE) not in sys.path:
    sys.path.insert(0, str(KAGGLE))

import launch  # noqa: E402


def test_quantity_aliases_and_profile_resolution() -> None:
    assert launch.parse_quantity("20M") == 20_000_000
    assert launch.parse_quantity("500m") == 500_000_000
    assert launch.parse_quantity("2B") == 2_000_000_000
    assert launch.parse_quantity("2000M") == 2_000_000_000
    profile = launch.resolve_profile(
        launch.parse_quantity("20M"),
        launch.parse_quantity("2000M"),
    )
    assert profile.train_module == "run_20m_2b"
    assert profile.publish_module == "build_and_push_2b"


def test_train_dry_run_resolves_without_importing_backend() -> None:
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
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload == {
        "action": "train",
        "backend_argv": ["--max-steps-this-session", "250"],
        "backend_module": "run_20m_2b",
        "model": "20M",
        "resume": "automatic_verified",
        "tokens": "2B",
    }


def test_publish_dry_run_forwards_publication_options() -> None:
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
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["backend_module"] == "build_and_push_500m"
    assert payload["backend_argv"] == [
        "--dataset-dir",
        "/tmp/dataset",
        "--ops-dir",
        "/tmp/ops",
        "--force-upload",
    ]
    assert payload["resume"] == "automatic_verified"


def test_resume_flag_is_rejected_because_resume_is_automatic() -> None:
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
    assert result.returncode == 2
    assert "Resume is fail-closed and automatic" in result.stderr


def test_unsupported_profile_fails_before_backend_import() -> None:
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
    assert result.returncode == 2
    assert "unsupported model/token profile" in result.stderr
