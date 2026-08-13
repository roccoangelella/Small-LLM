"""Pure helpers for the opt-in live Modal/HF incremental smoke test."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

SMOKE_CONTEXT_LENGTH = 2048
SMOKE_SEQUENCES_PER_BLOCK = 64
SMOKE_BLOCK_TARGET_TOKENS = SMOKE_CONTEXT_LENGTH * SMOKE_SEQUENCES_PER_BLOCK
SMOKE_BLOCK_BYTES = (SMOKE_CONTEXT_LENGTH + 1) * SMOKE_SEQUENCES_PER_BLOCK * 2
SMOKE_SHARD_BLOCKS = 16
SMOKE_TARGET_SHARD_BYTES = SMOKE_BLOCK_BYTES * SMOKE_SHARD_BLOCKS
SMOKE_TRAIN_BLOCKS = 64
SMOKE_NOMINAL_TRAINING_TOKENS = SMOKE_TRAIN_BLOCKS * SMOKE_BLOCK_TARGET_TOKENS
SMOKE_VALIDATION_BLOCKS = 1
SMOKE_VALIDATION_PROBABILITY = 0.10
SMOKE_SOURCE_TARGET_TOKENS = 1_000_000_000
SMOKE_SOURCE_MINIMUM_TOKENS = 900_000_000
SMOKE_SOURCE_MAXIMUM_TOKENS = 1_100_000_000
SMOKE_CHECKPOINT_SOURCE_TOKENS = 5_000_000
SMOKE_FIRST_SEGMENT_STEPS = SMOKE_SHARD_BLOCKS
SMOKE_SECOND_SEGMENT_STEPS = 4
SMOKE_TOTAL_EXERCISED_STEPS = SMOKE_FIRST_SEGMENT_STEPS + SMOKE_SECOND_SEGMENT_STEPS
SMOKE_MICROBATCH = 16
SMOKE_REMOTE_PUBLISH_EVERY = 10_000
APPROVED_WEIGHTS_SHA256 = "76e82e22760adcac59c7294fe9bac11358f5a8b7a26035aae64c3f2e6fa1acb7"

_NONCE = re.compile(r"^[0-9a-f]{12}$")
_RUN_ID = re.compile(r"^smoke-incremental-(?:dataset|train)-[0-9a-f]{12}$")


@dataclass(frozen=True, slots=True)
class SmokeIdentity:
    nonce: str
    dataset_run_id: str
    training_run_id: str
    checkpoint_repo_id: str


def validate_nonce(nonce: str) -> str:
    value = nonce.strip().lower()
    if _NONCE.fullmatch(value) is None:
        raise ValueError("smoke nonce must be exactly 12 lowercase hexadecimal characters")
    return value


def validate_smoke_run_id(run_id: str) -> str:
    if _RUN_ID.fullmatch(run_id) is None:
        raise ValueError(f"cleanup is restricted to smoke run IDs: {run_id!r}")
    return run_id


def smoke_identity(base_repo_id: str, nonce: str) -> SmokeIdentity:
    nonce = validate_nonce(nonce)
    if base_repo_id.count("/") != 1:
        raise ValueError("SMALL_LLM_HF_REPO_ID must be in owner/name form")
    owner, name = base_repo_id.split("/", 1)
    if not owner or not name:
        raise ValueError("SMALL_LLM_HF_REPO_ID must be in owner/name form")
    return SmokeIdentity(
        nonce=nonce,
        dataset_run_id=f"smoke-incremental-dataset-{nonce}",
        training_run_id=f"smoke-incremental-train-{nonce}",
        checkpoint_repo_id=f"{owner}/{name}-incremental-smoke-{nonce}",
    )


def producer_arguments(*, weights_file: Path, output_dir: Path, dataset_run_id: str, dataset_bucket_id: str) -> list[str]:
    validate_smoke_run_id(dataset_run_id)
    return [
        "--weights-file", str(weights_file), "--output-dir", str(output_dir),
        "--run-id", dataset_run_id, "--evict-remote-shards", "--incremental-frontier",
        "--nominal-training-tokens", str(SMOKE_NOMINAL_TRAINING_TOKENS),
        "--training-validation-blocks", str(SMOKE_VALIDATION_BLOCKS),
        "--hf-bucket-id", dataset_bucket_id,
        "--target-tokens", str(SMOKE_SOURCE_TARGET_TOKENS),
        "--minimum-tokens", str(SMOKE_SOURCE_MINIMUM_TOKENS),
        "--maximum-tokens", str(SMOKE_SOURCE_MAXIMUM_TOKENS),
        "--checkpoint-source-tokens", str(SMOKE_CHECKPOINT_SOURCE_TOKENS),
        "--context-length", str(SMOKE_CONTEXT_LENGTH),
        "--sequences-per-block", str(SMOKE_SEQUENCES_PER_BLOCK),
        "--target-shard-bytes", str(SMOKE_TARGET_SHARD_BYTES),
        "--reader-workers", "4", "--max-in-flight-work-items", "16",
    ]


def _replace_option(command: list[str], option: str, value: str) -> None:
    try:
        index = command.index(option)
    except ValueError as error:
        raise RuntimeError(f"trainer command is missing required option {option}") from error
    if index + 1 >= len(command):
        raise RuntimeError(f"trainer command option has no value: {option}")
    command[index + 1] = value


def wire_live_smoke_trainer_command(command: list[str], *, dataset_bucket_id: str, dataset_run_id: str, remote_manifest: Path, checkpoint_repo_id: str) -> list[str]:
    validate_smoke_run_id(dataset_run_id)
    result = list(command)
    _replace_option(result, "--remote-publish-every-steps", str(SMOKE_REMOTE_PUBLISH_EVERY))
    _replace_option(result, "--wandb-mode", "disabled")
    result.extend([
        "--dataset-shard-bucket", dataset_bucket_id,
        "--dataset-shard-run-id", dataset_run_id,
        "--dataset-shard-prefetch", "1",
        "--remote-drive-manifest", str(remote_manifest),
        "--remote-checkpoint-repo", checkpoint_repo_id,
        "--remote-token-env", "HF_TOKEN",
        "--remote-create-repo", "--remote-rolling-latest-only",
    ])
    return result
