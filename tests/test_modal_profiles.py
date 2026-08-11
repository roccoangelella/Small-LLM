"""Regression tests for the pure Modal training profile surface."""
from __future__ import annotations

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILES = runpy.run_path(str(ROOT / "modal" / "profiles.py"))


def test_modal_100m_2b_profile_uses_substantive_geometry_and_block64_dataset() -> None:
    model, tokens = PROFILES["resolve_presets"]("100M", "2B")

    assert model.trainer_size == "substantive"
    assert tokens.dataset_profile == "modal-2b-b64"
    assert PROFILES["canonical_run_id"](model, tokens) == "100m-2b-data-001"


def test_modal_microbatch_candidates_cover_h100_capacity_range() -> None:
    assert PROFILES["SEQUENCES_PER_BLOCK"] == 64
    assert PROFILES["MICROBATCH_CANDIDATES"] == (16, 32, 48, 64)
    assert max(PROFILES["MICROBATCH_CANDIDATES"]) == PROFILES["SEQUENCES_PER_BLOCK"]


def test_modal_default_gpu_is_h100() -> None:
    assert PROFILES["DEFAULT_GPU"] == "H100"
