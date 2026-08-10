"""Regression tests for the 2B one-click profile dispatch and pinning."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path


def test_2b_entry_rewrites_only_the_qualification_report_module() -> None:
    root = Path(__file__).resolve().parents[1]
    kaggle = root / "kaggle"
    script = r'''
import json
import sys
sys.path.insert(0, sys.argv[1])
import run_20m_2b as entry
commands = [
    ["python", "-m", "dataset.qualification_100m_report", "--dataset-dir", "/data"],
    ["python", "-m", "dataset.main", "verify", "--output-dir", "/data"],
]
print(json.dumps([entry._rewrite_profile_command(command) for command in commands]))
'''
    output = subprocess.check_output(
        [sys.executable, "-c", script, str(kaggle)],
        cwd=root,
        text=True,
    )
    rewritten, untouched = json.loads(output.strip().splitlines()[-1])
    assert rewritten[2] == "dataset.qualification_2b_report"
    assert "dataset.qualification_100m_report" not in rewritten
    assert untouched == [
        "python",
        "-m",
        "dataset.main",
        "verify",
        "--output-dir",
        "/data",
    ]


def test_2b_entry_has_an_immutable_launch_commit() -> None:
    root = Path(__file__).resolve().parents[1]
    kaggle = root / "kaggle"
    script = r'''
import json
import sys
sys.path.insert(0, sys.argv[1])
import run_20m_2b as entry
print(json.dumps({"commit": entry.PINNED_LAUNCH_COMMIT}))
'''
    output = subprocess.check_output(
        [sys.executable, "-c", script, str(kaggle)],
        cwd=root,
        text=True,
    )
    payload = json.loads(output.strip().splitlines()[-1])
    assert re.fullmatch(r"[0-9a-f]{40}", payload["commit"])
