#!/usr/bin/env python3
"""Single human entry point for Small-LLM reasoning SFT on Kaggle.

Examples:
  python kaggle/launch_r_sft.py train --model 100M --tokens 2B \
    --dataset-dir /kaggle/input/rsft-bundle --run-id 100m-2b-rsft-r0-atomic-pilot-001 \
    --delimiter-format atomic --token-spec /kaggle/input/rsft-bundle/reasoning-tokens.json
  python kaggle/launch_r_sft.py train --model 100M --tokens 2B \
    --dataset-dir /kaggle/input/rsft-bundle-text --run-id 100m-2b-rsft-r0-text-pilot-001 \
    --delimiter-format textual --token-spec /kaggle/input/rsft-bundle-text/reasoning-tokens.json
"""
from __future__ import annotations

from rsft_cli import build_parser, main, parse_quantity


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main", "parse_quantity"]
