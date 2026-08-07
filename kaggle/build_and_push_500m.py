#!/usr/bin/env python3
"""Build, verify, and privately publish the fixed 500M-token Kaggle dataset.

This is a fail-closed profile overlay on the proven 100M publication suite.  It
reuses the same verification and round-trip machinery while binding a distinct
500M production identity and distinct environment namespace.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import re
import sys
from pathlib import Path
from typing import Mapping, Sequence

IMPLEMENTATION = Path(__file__).with_name("build_and_push_100m.py")
SPEC = importlib.util.spec_from_file_location("small_llm_build_and_push_500m_base", IMPLEMENTATION)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load {IMPLEMENTATION}")
base = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = base
SPEC.loader.exec_module(base)

PROFILE = "20m-500m-data-scaling-v1"
RUN_ID = "20m-500m-dataset-001"
SLUG = "small-llm-20m-500m-dataset-001"
TARGET_SOURCE_TOKENS = 500_000_000
MINIMUM_SOURCE_TOKENS = 450_000_000
MAXIMUM_SOURCE_TOKENS = 550_000_000
CHECKPOINT_SOURCE_TOKENS = 20_000_000
DEFAULT_WEIGHTS = "/data/climbmix-mixture-calibration/climbmix_code_free_weights.json"
DEFAULT_DATASET = "/data/small-llm/20m-500m-dataset-001"
DEFAULT_OPS = "/data/small-llm/20m-500m-ops"
HANDLE_ENV = "SMALL_LLM_500M_KAGGLE_DATASET_HANDLE"
KAGGLE_TRANSPORT_ARCHIVE = re.compile(r"^[0-9]+\.archive$")

# The base suite is intentionally reused rather than forked.  Fail closed if
# its identity changes in a way that would make this overlay ambiguous.
_EXPECTED_BASE = {
    "PROFILE": "20m-100m-data-scaling-v1",
    "RUN_ID": "20m-100m-dataset-001",
    "SLUG": "small-llm-20m-100m-dataset-001",
}
for _name, _expected in _EXPECTED_BASE.items():
    if getattr(base, _name, None) != _expected:
        raise RuntimeError(
            f"500M publisher base contract changed: {_name}="
            f"{getattr(base, _name, None)!r}, expected {_expected!r}"
        )

base.PROFILE = PROFILE
base.RUN_ID = RUN_ID
base.SLUG = SLUG
base.DEFAULT_WEIGHTS = DEFAULT_WEIGHTS
base.DEFAULT_DATASET = DEFAULT_DATASET
base.DEFAULT_OPS = DEFAULT_OPS


def production_identity() -> dict[str, object]:
    return {
        "run_id": RUN_ID,
        "target_source_tokens": TARGET_SOURCE_TOKENS,
        "minimum_source_tokens": MINIMUM_SOURCE_TOKENS,
        "maximum_source_tokens": MAXIMUM_SOURCE_TOKENS,
        "checkpoint_source_tokens": CHECKPOINT_SOURCE_TOKENS,
        "target_reached": True,
        "remote_required": True,
    }


def resolve_handle(explicit: str | None, env: Mapping[str, str]) -> str:
    handle = explicit or env.get(HANDLE_ENV, "")
    if not handle and env.get("KAGGLE_USERNAME"):
        handle = f"{env['KAGGLE_USERNAME']}/{SLUG}"
    if not base.HANDLE_RE.fullmatch(handle):
        raise base.SuiteFailure(
            f"Set KAGGLE_USERNAME or {HANDLE_ENV}=owner/dataset in .env"
        )
    return handle


def producer_command(config: base.Config, resume: bool) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "dataset.qualification_500m",
        "--weights-file",
        str(config.weights),
        "--output-dir",
        str(config.dataset),
    ]
    return command + (["--resume"] if resume else [])


def derive_plan(root: Path, prefix: str, config: base.Config) -> None:
    base.run(
        [
            sys.executable,
            "-m",
            "dataset.qualification_500m_report",
            "--dataset-dir",
            str(root),
            "--drive-manifest",
            str(root / "drive_manifest.json"),
            "--output",
            str(root / "qualification_plan.json"),
        ],
        f"{prefix}-qualification-plan",
        config,
    )


def dataset_tree_identity(root: Path) -> dict[str, object]:
    """Hash payload files while excluding kagglehub's local transport archive."""

    digest = hashlib.sha256()
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and not (
            path.parent == root
            and KAGGLE_TRANSPORT_ARCHIVE.fullmatch(path.name)
        )
    )
    total = 0
    for path in files:
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        file_hash = base.sha256(path)
        total += size
        digest.update(f"{relative}\0{size}\0{file_hash}\n".encode())
    return {
        "tree_sha256": digest.hexdigest(),
        "file_count": len(files),
        "total_bytes": total,
    }


def _install_environment_aliases() -> None:
    """Keep old 100M-only parser internals from leaking into this profile."""

    aliases = {
        "SMALL_LLM_100M_WEIGHTS_FILE": "SMALL_LLM_500M_WEIGHTS_FILE",
        "SMALL_LLM_100M_DATASET_DIR": "SMALL_LLM_500M_DATASET_DIR",
        "SMALL_LLM_100M_OPS_DIR": "SMALL_LLM_500M_OPS_DIR",
    }
    for legacy, current in aliases.items():
        value = os.environ.get(current)
        if value:
            os.environ[legacy] = value
        else:
            os.environ.pop(legacy, None)


def _install_overlay() -> None:
    base.production_identity = production_identity
    base.resolve_handle = resolve_handle
    base.producer_command = producer_command
    base.derive_plan = derive_plan
    base.tree_identity = dataset_tree_identity


def main(argv: Sequence[str] | None = None) -> int:
    _install_environment_aliases()
    _install_overlay()
    return base.main(argv)


SuiteFailure = base.SuiteFailure
Config = base.Config
sha256 = base.sha256


if __name__ == "__main__":
    raise SystemExit(main())
