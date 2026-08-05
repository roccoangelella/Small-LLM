#!/usr/bin/env python3
"""Simple entrypoint for the authorized Kaggle 20M full training run."""

from pathlib import Path
from runpy import run_path

run_path(
    str(Path(__file__).resolve().parent / "kaggle" / "run_20m_full_training.py"),
    run_name="__main__",
)
