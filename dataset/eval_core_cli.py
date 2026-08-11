"""Standalone CPU-friendly CLI for building and verifying ``eval_core_v1``.

The permanent evaluation corpus is model-independent. Build it once with the
accelerated deterministic source scanner, verify it immediately, then persist
that immutable directory for reuse by all compatible model checkpoints.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from dataset.eval_core import verify_eval_core
from dataset.eval_core_accelerated import build_eval_core_accelerated


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build or verify the permanent frozen eval_core_v1 corpus"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser(
        "build",
        help="build with the accelerated CPU/network scanner, then verify",
    )
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument(
        "--max-work-items",
        type=int,
        help="diagnostic scan bound only; leave unset for the production corpus",
    )

    verify = sub.add_parser(
        "verify",
        help="verify hashes, frozen geometry, per-cluster quotas, and nesting",
    )
    verify.add_argument("--eval-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _summary(status: str, eval_dir: Path, manifest: dict[str, object]) -> None:
    print(
        json.dumps(
            {
                "status": status,
                "eval_dir": str(eval_dir.expanduser().resolve()),
                "manifest_sha256": manifest["manifest_sha256"],
                "suites": manifest["suites"],
            },
            indent=2,
            sort_keys=True,
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    if args.command == "build":
        output_dir = args.output_dir.expanduser().resolve()
        build_eval_core_accelerated(
            output_dir,
            max_work_items=args.max_work_items,
        )
        manifest = verify_eval_core(output_dir)
        _summary("completed_and_verified", output_dir, manifest)
        return 0

    eval_dir = args.eval_dir.expanduser().resolve()
    manifest = verify_eval_core(eval_dir)
    _summary("verified", eval_dir, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
