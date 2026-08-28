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
_HF_UPLOAD_ENV = {
    # Kaggle has already shown that Xet-backed checkpoint publication can stall
    # or kill a training process while uploading the large trainer state. The
    # two-phase publisher is resume-safe, but publication must not terminate the
    # training session. Match the already-qualified R-SFT Kaggle hardening.
    "HF_HUB_DISABLE_XET": "1",
    # Notebook progress rendering can itself block on IOStream flushes during a
    # large upload. Keep publication logs textual and bounded.
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
}


def _apply_hf_upload_hardening() -> None:
    """Force the stable HTTP upload path for every canonical SFT subprocess."""

    for key, value in _HF_UPLOAD_ENV.items():
        os.environ[key] = value


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


# Apply this before the optional uv re-exec so the hardened transport survives
# both the bootstrap process and the later detached-worktree/DDP subprocesses.
_apply_hf_upload_hardening()
_ensure_supported_python()

import sft_runtime  # noqa: E402
from sft_cli import build_parser, main, parse_quantity  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "_HF_UPLOAD_ENV",
    "_apply_hf_upload_hardening",
    "build_parser",
    "main",
    "parse_quantity",
    "sft_runtime",
]
