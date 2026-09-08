#!/usr/bin/env python3
"""Compatibility wrapper for :mod:`kaggle.src.wandb_preflight`.

The stable root path remains both executable and import-safe for commit-pinned
Kaggle launch worktrees.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent / "src"
_TARGET = _SRC_DIR / "wandb_preflight.py"

if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

_IMPL = runpy.run_path(str(_TARGET), run_name="_small_llm_kaggle_wandb_preflight_impl")
for _name, _value in _IMPL.items():
    if not _name.startswith("__"):
        globals()[_name] = _value

if __name__ == "__main__":
    raise SystemExit(main())
