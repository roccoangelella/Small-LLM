"""Regression guard for the local Beam launcher dependency."""

from __future__ import annotations

import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_beam_client_is_declared() -> None:
    payload = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = payload["project"]["optional-dependencies"]
    assert extras["beam"] == ["beam-client==0.2.201"]
