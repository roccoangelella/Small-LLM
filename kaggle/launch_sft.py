#!/usr/bin/env python3
"""Single human entry point for Small-LLM supervised fine-tuning."""
from __future__ import annotations

import sft_runtime
from sft_cli import build_parser, main, parse_quantity


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main", "parse_quantity", "sft_runtime"]
