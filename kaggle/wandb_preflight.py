#!/usr/bin/env python3
"""Compatibility wrapper for kaggle/src/wandb_preflight.py.

Keep this file thin so commit-pinned Kaggle launch worktrees that still call
`python kaggle/wandb_preflight.py` continue to work after the Kaggle workspace
reorganization moved implementation files under `kaggle/src/`.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

_KAGGLE_DIR = Path(__file__).resolve().parent
_SRC_DIR = _KAGGLE_DIR / "src"
_TARGET = _SRC_DIR / "wandb_preflight.py"

if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

runpy.run_path(str(_TARGET), run_name="__main__")
