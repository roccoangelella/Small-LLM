"""Self-provisioning entry point for the complete Small-LLM evaluation suite.

The underlying evaluator intentionally keeps eval_core verification explicit.
This wrapper makes the normal user-facing command operationally complete:
if the frozen eval corpus is absent, build it with the deterministic accelerated
scanner; always verify it; then run the existing evaluator against the selected
checkpoint.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import threading
from typing import Sequence

from dataset.eval_core import verify_eval_core
from dataset.eval_core_accelerated import build_eval_core_accelerated
from trainer import eval_suite

_BUILD_HEARTBEAT_SECONDS = 15.0


def _attached_eval_dir() -> Path | None:
    """Return the unique attached Kaggle eval_core_v1 root, when present."""

    kaggle_input = Path("/kaggle/input")
    if not kaggle_input.is_dir():
        return None
    matches: list[Path] = []
    for manifest_path in kaggle_input.rglob("manifest.json"):
        if not manifest_path.is_file():
            continue
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        if isinstance(payload, dict) and payload.get("name") == "eval_core_v1":
            matches.append(manifest_path.parent.resolve())
    unique = sorted(set(matches))
    if len(unique) > 1:
        raise RuntimeError(
            f"expected at most one attached eval_core_v1 corpus; found {len(unique)}: {unique}"
        )
    return unique[0] if unique else None


def default_eval_dir() -> Path:
    """Return an attached eval corpus or the writable local cache location."""

    configured = os.environ.get("SMALL_LLM_EVAL_DIR")
    if configured:
        return Path(configured).expanduser()
    attached = _attached_eval_dir()
    if attached is not None:
        print(f"Discovered attached eval_core_v1 at {attached}", flush=True)
        return attached
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


def _partial_build_bytes(eval_dir: Path) -> int:
    """Return bytes currently written by this process's atomic eval build."""

    parent = eval_dir.parent
    prefix = f".{eval_dir.name}.partial-"
    total = 0
    try:
        candidates = tuple(parent.iterdir())
    except OSError:
        return 0
    for candidate in candidates:
        if not candidate.is_dir() or not candidate.name.startswith(prefix):
            continue
        try:
            total += sum(
                path.stat().st_size
                for path in candidate.rglob("*")
                if path.is_file()
            )
        except OSError:
            continue
    return total


def _build_with_heartbeat(eval_dir: Path) -> None:
    """Build eval_core while making long remote scans visibly alive."""

    stop = threading.Event()

    def heartbeat() -> None:
        while not stop.wait(_BUILD_HEARTBEAT_SECONDS):
            written = _partial_build_bytes(eval_dir)
            print(
                "eval_core_v1 build active: accelerated pinned ClimbMix scan "
                f"| partial_output={written / (1024 * 1024):.1f} MiB",
                flush=True,
            )

    thread = threading.Thread(
        target=heartbeat,
        name="eval-core-build-heartbeat",
        daemon=True,
    )
    thread.start()
    try:
        build_eval_core_accelerated(eval_dir)
    finally:
        stop.set()
        thread.join(timeout=1.0)


def ensure_eval_core(eval_dir: Path) -> Path:
    """Build the frozen eval corpus when absent and always verify it."""

    resolved = eval_dir.expanduser().resolve()
    if not resolved.exists():
        print(
            f"eval_core_v1 not found at {resolved}; building the frozen evaluation corpus...",
            flush=True,
        )
        print(
            "Using the deterministic accelerated source scanner: validation identity "
            "is checked before JSON parsing and source regions are fetched concurrently.",
            flush=True,
        )
        resolved.parent.mkdir(parents=True, exist_ok=True)
        _build_with_heartbeat(resolved)
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
