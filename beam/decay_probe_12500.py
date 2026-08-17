#!/usr/bin/env python3
"""Compatibility redirect for the superseded step-12,500 decay probe.

The exact step-12,500 checkpoint is no longer retained. Use
``beam/decay_probe_15500.py``; this wrapper forwards there so an old command
cannot accidentally launch the obsolete experiment.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beam.decay_probe_15500 import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
