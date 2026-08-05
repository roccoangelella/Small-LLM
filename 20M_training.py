#!/usr/bin/env python3
"""Simple entrypoint for the authorized Kaggle 20M full training run."""

import sys
from pathlib import Path
from runpy import run_path

launcher_dir = Path(__file__).resolve().parent / "kaggle"
sys.path.insert(0, str(launcher_dir))
run_path(str(launcher_dir / "run_20m_full_training.py"), run_name="__main__")
