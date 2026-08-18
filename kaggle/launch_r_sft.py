#!/usr/bin/env python3
"""Single human entry point for Small-LLM reasoning SFT on Kaggle.

The first 630-example pilot is committed to the repository. Kaggle builds the
matched native bundle automatically from that corpus plus the completed S0
instruction bundle, then trains the selected arm on the qualified 2xT4 path.

Examples:
  python kaggle/launch_r_sft.py train --model 100M --tokens 2B --delimiter-format atomic
  python kaggle/launch_r_sft.py train --model 100M --tokens 2B --delimiter-format textual
  python kaggle/launch_r_sft.py train --model 100M --tokens 2B --delimiter-format atomic --dry-run

Pass --dataset-dir, --s0-bundle, --run-id, or --token-spec only as explicit
overrides; they are not needed for the canonical pilot.
"""
from __future__ import annotations

from rsft_cli import build_parser, main, parse_quantity


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main", "parse_quantity"]
