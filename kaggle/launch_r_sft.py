#!/usr/bin/env python3
"""Single human entry point for Small-LLM reasoning SFT on Kaggle.

Canonical production R-SFT is atomic-only. By default it materializes the
committed 12,306-row Superior-instruction checkpoint corpus with the completed S0 retention
bundle, verifies the resulting ``atomic-production-v1`` bundle, then trains:

  python kaggle/launch_r_sft.py train --model 100M --tokens 2B

A prebuilt verified bundle can still be supplied explicitly with ``--dataset-dir``.

The completed 630-example delimiter experiment remains reproducible separately:

  python kaggle/launch_r_sft.py ablation --model 100M --tokens 2B \
    --delimiter-format atomic
  python kaggle/launch_r_sft.py ablation --model 100M --tokens 2B \
    --delimiter-format textual

Production ``train`` has no textual mode. It validates the bundle and frozen
<think>, </think>, <answer> token contract before dual-T4 dispatch.
"""
from __future__ import annotations

from rsft_cli import build_parser, main, parse_quantity


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main", "parse_quantity"]
