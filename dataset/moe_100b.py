"""Frozen 100B SuperBPE dataset entrypoint for the accepted MoE line.

This is a thin identity layer over ``dataset.production``.  Source selection,
cluster stratification, crash-safe checkpoints, immutable schema-v2 shards,
incremental READY publication and Hugging Face durability remain implemented by
the existing production pipeline.
"""

from __future__ import annotations

from collections.abc import Sequence
import sys
from typing import Callable

from dataset.production.cli import main as production_main

RUN_ID = "moe-100b-superbpe-b64-dataset-001"
TARGET_SOURCE_TOKENS = 100_000_000_000
MINIMUM_SOURCE_TOKENS = 90_000_000_000
MAXIMUM_SOURCE_TOKENS = 110_000_000_000
CHECKPOINT_SOURCE_TOKENS = 500_000_000
CONTEXT_LENGTH = 2_048
SEQUENCES_PER_BLOCK = 64
TARGET_SHARD_BYTES = 1024**3
NOMINAL_TRAINING_TOKENS = 100_000_000_000
TRAINING_VALIDATION_BLOCKS = 16

_LOCKED_FLAGS = frozenset(
    {
        "--run-id",
        "--tokenizer-contract",
        "--target-tokens",
        "--minimum-tokens",
        "--maximum-tokens",
        "--checkpoint-source-tokens",
        "--context-length",
        "--sequences-per-block",
        "--target-shard-bytes",
        "--public-hf-bucket",
        "--evict-remote-shards",
        "--incremental-frontier",
        "--nominal-training-tokens",
        "--training-validation-blocks",
        "--allow-local-only",
    }
)


def production_arguments(argv: Sequence[str]) -> list[str]:
    supplied = {
        argument.split("=", 1)[0]
        for argument in argv
        if argument.startswith("--")
    }
    conflicts = sorted(supplied & _LOCKED_FLAGS)
    if conflicts:
        raise SystemExit(
            "the accepted 100B SuperBPE corpus fixes these arguments: "
            + ", ".join(conflicts)
        )
    return [
        *argv,
        "--run-id", RUN_ID,
        "--tokenizer-contract", "superbpe_8000",
        "--target-tokens", str(TARGET_SOURCE_TOKENS),
        "--minimum-tokens", str(MINIMUM_SOURCE_TOKENS),
        "--maximum-tokens", str(MAXIMUM_SOURCE_TOKENS),
        "--checkpoint-source-tokens", str(CHECKPOINT_SOURCE_TOKENS),
        "--context-length", str(CONTEXT_LENGTH),
        "--sequences-per-block", str(SEQUENCES_PER_BLOCK),
        "--target-shard-bytes", str(TARGET_SHARD_BYTES),
        "--public-hf-bucket",
        "--evict-remote-shards",
        "--incremental-frontier",
        "--nominal-training-tokens", str(NOMINAL_TRAINING_TOKENS),
        "--training-validation-blocks", str(TRAINING_VALIDATION_BLOCKS),
    ]


def main(
    argv: Sequence[str] | None = None,
    *,
    durable_progress_hook: Callable[[], object] | None = None,
) -> int:
    supplied = list(sys.argv[1:] if argv is None else argv)
    return production_main(
        production_arguments(supplied),
        durable_progress_hook=durable_progress_hook,
    )


if __name__ == "__main__":
    raise SystemExit(main())
