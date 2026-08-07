"""Regression test for the 500M one-click qualification-report dispatch."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_500m_entry_rewrites_only_the_qualification_report_module() -> None:
    root = Path(__file__).resolve().parents[1]
    kaggle = root / "kaggle"
    script = r'''
import json
import sys
sys.path.insert(0, sys.argv[1])
import run_20m_500m as entry
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
    assert rewritten[2] == "dataset.qualification_500m_report"
    assert "dataset.qualification_100m_report" not in rewritten
    assert untouched == [
        "python",
        "-m",
        "dataset.main",
        "verify",
        "--output-dir",
        "/data",
    ]
