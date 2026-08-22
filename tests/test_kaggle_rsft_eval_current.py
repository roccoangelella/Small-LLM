from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
KAGGLE = REPO / "kaggle"
if str(KAGGLE) not in sys.path:
    sys.path.insert(0, str(KAGGLE))

import rsft_cli


def test_canonical_eval_targets_current_three_epoch_rsft(
    tmp_path: Path,
    capsys,
) -> None:
    eval_dir = tmp_path / "eval_core_v1"
    eval_dir.mkdir()

    result = rsft_cli.main(
        [
            "eval",
            "--model",
            "100M",
            "--tokens",
            "2B",
            "--eval-dir",
            str(eval_dir),
            "--dry-run",
        ]
    )

    assert result == 0
    output = capsys.readouterr().out
    assert rsft_cli.CURRENT_EVAL_EPOCHS == 3
    assert rsft_cli.CURRENT_EVAL_RUN_ID == "100m-2b-rsft-r0-16716-e3-001"
    assert "100m-2b-sft-s0-001->100m-2b-rsft-r0-16716-e3-001" in output
