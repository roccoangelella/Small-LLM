"""Fixed finite-dataset entry point for the 20M-model/500M-token scaling run."""

from __future__ import annotations

import sys
from collections.abc import Sequence

from dataset.production.cli import main as production_main

TARGET_SOURCE_TOKENS = 500_000_000
MINIMUM_SOURCE_TOKENS = 450_000_000
MAXIMUM_SOURCE_TOKENS = 550_000_000
CONTEXT_LENGTH = 2_048
SEQUENCES_PER_BLOCK = 16
TARGET_SHARD_BYTES = 8 * 1024 * 1024
CHECKPOINT_SOURCE_TOKENS = 20_000_000
RUN_ID = "20m-500m-dataset-001"

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
        "--run-id",
    }
)


def qualification_arguments(argv: Sequence[str]) -> list[str]:
    """Append the fixed 500M profile while retaining safe producer tuning."""

    supplied = {
        argument.split("=", 1)[0]
        for argument in argv
        if argument.startswith("--")
    }
    conflicts = sorted(supplied & _LOCKED_FLAGS)
    if conflicts:
        raise SystemExit(
            "the 20M-model/500M-token dataset fixes these arguments: "
            + ", ".join(conflicts)
        )
    return [
        *argv,
        "--run-id",
        RUN_ID,
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
