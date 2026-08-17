#!/usr/bin/env python3
"""Compatibility redirect for the superseded step-12,500 decay probe.

The exact step-12,500 checkpoint is no longer retained. Use
``beam/decay_probe_15500.py``; this wrapper forwards there so an old command
cannot accidentally launch the obsolete experiment.
"""
from __future__ import annotations

from beam.decay_probe_15500 import main


if __name__ == "__main__":
    raise SystemExit(main())
