"""Self-provisioning entry point for the complete Small-LLM evaluation suite.

The underlying evaluator intentionally keeps eval_core verification explicit.
This wrapper makes the normal user-facing command operationally complete:
if the frozen eval corpus is absent, build it; always verify it; then run the
existing evaluator against the selected checkpoint.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Sequence

from dataset.eval_core import build_eval_core, verify_eval_core
from trainer import eval_suite


def default_eval_dir() -> Path:
    """Return the persistent eval_core_v1 cache location for this environment."""

    configured = os.environ.get("SMALL_LLM_EVAL_DIR")
    if configured:
        return Path(configured).expanduser()
    kaggle_working = Path("/kaggle/working")
    if kaggle_working.is_dir():
        return kaggle_working / "eval_core_v1"
    return Path("artifacts") / "eval_core_v1"


def _eval_dir_from_argv(argv: Sequence[str]) -> Path | None:
    for index, argument in enumerate(argv):
        if argument == "--eval-dir":
            if index + 1 >= len(argv):
                return None
            return Path(argv[index + 1]).expanduser()
        if argument.startswith("--eval-dir="):
            return Path(argument.split("=", 1)[1]).expanduser()
    return None


def ensure_eval_core(eval_dir: Path) -> Path:
    """Build the frozen eval corpus when absent and always verify it."""

    resolved = eval_dir.expanduser().resolve()
    if not resolved.exists():
        print(
            f"eval_core_v1 not found at {resolved}; building the frozen evaluation corpus...",
            flush=True,
        )
        resolved.parent.mkdir(parents=True, exist_ok=True)
        build_eval_core(resolved)
    else:
        print(f"Using existing eval_core_v1 at {resolved}", flush=True)

    verify_eval_core(resolved)
    print(f"Verified eval_core_v1 at {resolved}", flush=True)
    return resolved


def _with_eval_dir(argv: Sequence[str]) -> tuple[list[str], Path]:
    forwarded = list(argv)
    selected = _eval_dir_from_argv(forwarded)
    if selected is None:
        selected = default_eval_dir()
        forwarded.extend(("--eval-dir", str(selected)))
    return forwarded, selected


def main(argv: Sequence[str] | None = None) -> int:
    forwarded = list(sys.argv[1:] if argv is None else argv)
    if "-h" in forwarded or "--help" in forwarded:
        return eval_suite.main(forwarded)

    forwarded, selected = _with_eval_dir(forwarded)

    # Parse once before any potentially expensive corpus build so malformed
    # commands fail immediately. The underlying evaluator parses again when run.
    eval_suite._arguments(forwarded)
    ensure_eval_core(selected)
    return eval_suite.main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
