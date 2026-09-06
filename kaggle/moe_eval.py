#!/usr/bin/env python3
"""Run canonical pretrained evaluation v2 on a Top-1 MoE checkpoint from Kaggle."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from MOE_model.evaluation import main


if __name__ == "__main__":
    raise SystemExit(main())
