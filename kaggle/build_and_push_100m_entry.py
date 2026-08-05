#!/usr/bin/env python3
"""Official entry point for the idempotent 100M Kaggle publisher."""
from __future__ import annotations

import hashlib
import importlib.util
import re
import sys
from pathlib import Path

IMPLEMENTATION = Path(__file__).with_name("build_and_push_100m.py")
SPEC = importlib.util.spec_from_file_location("small_llm_build_and_push_100m", IMPLEMENTATION)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load {IMPLEMENTATION}")
suite = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = suite
SPEC.loader.exec_module(suite)

KAGGLE_TRANSPORT_ARCHIVE = re.compile(r"^[0-9]+\.archive$")


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
        file_hash = suite.sha256(path)
        total += size
        digest.update(f"{relative}\0{size}\0{file_hash}\n".encode())
    return {
        "tree_sha256": digest.hexdigest(),
        "file_count": len(files),
        "total_bytes": total,
    }


suite.tree_identity = dataset_tree_identity


if __name__ == "__main__":
    raise SystemExit(suite.main())
