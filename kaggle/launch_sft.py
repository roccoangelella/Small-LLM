#!/usr/bin/env python3
"""Compatibility wrapper for :mod:`kaggle.src.launch_sft`.

The stable root path remains both executable and import-safe after the Kaggle
workspace reorganization.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent / "src"
_TARGET = _SRC_DIR / "launch_sft.py"

if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

_IMPL = runpy.run_path(str(_TARGET), run_name="_small_llm_kaggle_launch_sft_impl")
for _name, _value in _IMPL.items():
    if not _name.startswith("__"):
        globals()[_name] = _value

if __name__ == "__main__":
    raise SystemExit(main())
