"""Fixed finite-dataset entry point for the 20M-model / 100M-token run."""

from __future__ import annotations

import sys
from collections.abc import Sequence

from dataset.production.cli import main as production_main

TARGET_SOURCE_TOKENS = 100_000_000
MINIMUM_SOURCE_TOKENS = 90_000_000
MAXIMUM_SOURCE_TOKENS = 110_000_000
CONTEXT_LENGTH = 2_048
SEQUENCES_PER_BLOCK = 16
TARGET_SHARD_BYTES = 8 * 1024 * 1024
CHECKPOINT_SOURCE_TOKENS = 20_000_000

_LOCKED_FLAGS = frozenset(
    {
        "--target-tokens",
        "--minimum-tokens",
        "--maximum-tokens",
        "--context-length",
        "--sequences-per-block",
        "--target-shard-bytes",
        "--checkpoint-source-tokens",
        "--allow-local-only",
    }
)


def qualification_arguments(argv: Sequence[str]) -> list[str]:
    """Append the frozen 100M data-scaling identity to production arguments."""

    supplied = {
        argument.split("=", 1)[0]
        for argument in argv
        if argument.startswith("--")
    }
    conflicts = sorted(supplied & _LOCKED_FLAGS)
    if conflicts:
        raise SystemExit(
            "the 20M-model / 100M-token dataset fixes these arguments: "
            + ", ".join(conflicts)
        )
    return [
        *argv,
        "--target-tokens",
        str(TARGET_SOURCE_TOKENS),
        "--minimum-tokens",
        str(MINIMUM_SOURCE_TOKENS),
        "--maximum-tokens",
        str(MAXIMUM_SOURCE_TOKENS),
        "--checkpoint-source-tokens",
        str(CHECKPOINT_SOURCE_TOKENS),
        "--context-length",
        str(CONTEXT_LENGTH),
        "--sequences-per-block",
        str(SEQUENCES_PER_BLOCK),
        "--target-shard-bytes",
        str(TARGET_SHARD_BYTES),
    ]


def main(argv: Sequence[str] | None = None) -> int:
    supplied = list(sys.argv[1:] if argv is None else argv)
    return production_main(qualification_arguments(supplied))


if __name__ == "__main__":
    raise SystemExit(main())
