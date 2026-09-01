"""Static contracts for the disposable Kaggle Probe A LR-reset launcher."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "kaggle" / "probe_a_lr_reset_10b.py"


def test_probe_a_script_compiles() -> None:
    compile(SCRIPT.read_text(encoding="utf-8"), str(SCRIPT), "exec")


def test_probe_a_has_two_wandb_only_branches() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "reset-low" in text
    assert "reset-mid" in text
    assert "100m-10b-probe-a-{branch.slug}-from-step{source_step}" in text
    assert "--wandb-run-id" in text
    assert "--wandb-mode" in text
    assert '"online"' in text


def test_probe_a_disables_hf_publication() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "_publish_latest_to_bucket" not in text
    assert "_prepare(runtime_base)" not in text
    assert '"--remote-publish-every-steps", "0"' in text
    assert '"--best-model-repo"' in text
    assert "HF publication flags are forbidden for Probe A" in text
