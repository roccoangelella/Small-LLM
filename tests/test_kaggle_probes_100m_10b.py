"""Static contracts for the canonical 100M/10B Kaggle probe launcher."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = ROOT / "kaggle" / "probes_100m_10b.py"
PROBES = ROOT / "kaggle" / "src" / "probes_100m_10b.py"


def test_100m_10b_probe_files_compile() -> None:
    compile(ENTRYPOINT.read_text(encoding="utf-8"), str(ENTRYPOINT), "exec")
    compile(PROBES.read_text(encoding="utf-8"), str(PROBES), "exec")


def test_public_probe_entrypoint_normalizes_moved_runtime_paths() -> None:
    text = ENTRYPOINT.read_text(encoding="utf-8")
    assert 'ROOT = KAGGLE.parent' in text
    assert 'SRC = KAGGLE / "src"' in text
    assert 'BEAM = ROOT / "beam"' in text
    assert 'deep_impl.ROOT = ROOT' in text
    assert 'deep_impl.KAGGLE = SRC' in text
    assert 'deep_impl.BEAM = BEAM' in text
    assert 'sys.modules.pop("runtime", None)' in text
    assert 'BEAM / "runtime.py"' in text


def test_100m_10b_probes_are_consolidated() -> None:
    text = PROBES.read_text(encoding="utf-8")
    assert "single home for short, W&B-visible 100M/10B pretraining probes" in text
    assert 'PROBE_NAME = "100m-10b-probes"' in text
    assert "hold-1e-5" in text
    assert "hold-2e-5" in text
    assert "legacy-reset-1e-4" in text


def test_100m_10b_probes_prefer_71750_then_current_best() -> None:
    text = PROBES.read_text(encoding="utf-8")
    assert 'PREFERRED_SOURCE_CHECKPOINT_ID = "step-00071750"' in text
    assert "best_model.json" in text
    assert "current_best_fallback" in text
    assert 'rolling_latest_fallback": False' in text
    assert "_restore_source_checkpoint" in text


def test_100m_10b_probes_disable_hf_publication() -> None:
    text = PROBES.read_text(encoding="utf-8")
    assert '"--remote-publish-every-steps", "0"' in text
    assert "HF publication flags are forbidden" in text
    assert "_publish_latest_to_bucket" not in text


def test_100m_10b_active_probes_are_low_lr_constant_holds() -> None:
    text = PROBES.read_text(encoding="utf-8")
    assert 'ProbeBranch("hold-1e-5", "Hold 1e-5", 1e-5, "hold-1e-5")' in text
    assert 'ProbeBranch("hold-2e-5", "Hold 2e-5", 2e-5, "hold-2e-5")' in text
    assert '"schedule": "constant"' in text
    assert "DEFAULT_PROBE_STEPS = 3_000" in text


def test_100m_10b_probe_run_ids_encode_actual_source() -> None:
    text = PROBES.read_text(encoding="utf-8")
    assert "from-step{source_step}" in text
    assert "100m-10b-probe-a-reset-low-from-step71750" in text
    assert "wandb_run_id_template" in text
