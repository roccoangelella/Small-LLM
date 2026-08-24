#!/usr/bin/env python3
"""Single human entry point for Small-LLM supervised fine-tuning."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys

_MINIMUM_PYTHON = (3, 12)
_BOOTSTRAP_ENV = "SMALL_LLM_SFT_PYTHON_BOOTSTRAPPED"


def _ensure_supported_python() -> None:
    if sys.version_info >= _MINIMUM_PYTHON:
        return
    current = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if os.environ.get(_BOOTSTRAP_ENV) == "1":
        raise SystemExit(
            f"Small-LLM requires Python >=3.12; bootstrap still resolved Python {current}."
        )
    uv = shutil.which("uv")
    if uv is None:
        raise SystemExit(
            f"Small-LLM requires Python >=3.12, but this host's python3 is {current}. "
            "Install uv or run this launcher with Python 3.12+."
        )
    env = dict(os.environ)
    env[_BOOTSTRAP_ENV] = "1"
    command = [
        uv,
        "run",
        "--no-project",
        "--python",
        "3.13",
        "python",
        str(Path(__file__).resolve()),
        *sys.argv[1:],
    ]
    print(
        f"[launch] host_python={current} unsupported; re-exec with uv Python 3.13",
        flush=True,
    )
    raise SystemExit(subprocess.call(command, env=env))


_ensure_supported_python()

import sft_runtime  # noqa: E402
from sft_cli import build_parser, main, parse_quantity  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main", "parse_quantity", "sft_runtime"]
