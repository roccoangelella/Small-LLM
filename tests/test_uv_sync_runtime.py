"""Regression guards for the canonical plain-uv-sync runtime."""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_plain_uv_sync_runtime_contract() -> None:
    payload = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = set(payload["project"]["dependencies"])
    assert "beam-client==0.2.201" in dependencies
    assert "fla-core==0.5.2" in dependencies
    assert "torch>=2.7.0" in dependencies
    assert payload["tool"]["uv"]["default-groups"] == ["runtime"]
    assert set(payload["dependency-groups"]["runtime"]) == {
        "huggingface-hub>=1.5,<2",
        "tiktoken>=0.9.0",
        "wandb==0.26.1",
    }
