#!/usr/bin/env python3
"""Evaluate 100M/10B same-data SFT with the bucket-latest parent."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

EXPECTED_PARENT_RUN_ID = "100m-10b-deep-decay-from-step15500"
EXPECTED_PARENT_POINTER = "latest"


def _argument_value(argv: Sequence[str], flag: str) -> str:
    values = list(argv)
    try:
        index = values.index(flag)
    except ValueError as error:
        raise RuntimeError(f"required 10B SFT eval argument is missing: {flag}") from error
    if index + 1 >= len(values):
        raise RuntimeError(f"10B SFT eval argument has no value: {flag}")
    return str(values[index + 1])


def _expect_argument(argv: Sequence[str], flag: str, expected: str) -> None:
    actual = _argument_value(argv, flag)
    if actual != expected:
        raise RuntimeError(
            f"10B SFT eval is frozen to {flag} {expected}; received {actual}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    supplied = list(sys.argv[1:] if argv is None else argv)
    if "--parent-checkpoint-dir" not in supplied:
        _expect_argument(supplied, "--parent-run-id", EXPECTED_PARENT_RUN_ID)
        _expect_argument(supplied, "--parent-pointer", EXPECTED_PARENT_POINTER)

    from post_training.sft import eval_suite
    from post_training.sft.checkpoints import download_parent_checkpoint as original_download

    def bucket_latest_parent_or_default(
        *,
        repo_id: str,
        run_id: str,
        pointer: str = "best",
        token: str | None = None,
        revision: str | None = None,
        destination: Path | str | None = None,
    ):
        if run_id == EXPECTED_PARENT_RUN_ID:
            if pointer != EXPECTED_PARENT_POINTER:
                raise RuntimeError(
                    f"100M/10B SFT eval parent must use {EXPECTED_PARENT_POINTER}, got {pointer}"
                )
            return original_download(
                repo_id=repo_id,
                run_id=run_id,
                pointer=EXPECTED_PARENT_POINTER,
                transport="hf_storage_bucket",
                token=token,
                revision=revision,
                destination=destination,
            )
        return original_download(
            repo_id=repo_id,
            run_id=run_id,
            pointer=pointer,
            token=token,
            revision=revision,
            destination=destination,
        )

    eval_suite.download_parent_checkpoint = bucket_latest_parent_or_default
    return int(eval_suite.main(supplied))


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["EXPECTED_PARENT_POINTER", "EXPECTED_PARENT_RUN_ID", "REPO", "main"]
