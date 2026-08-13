"""Verification helpers for CPU-staged incremental dataset windows."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from pathlib import Path

from dataset.incremental_frontier import (
    RUN_CONTRACT_FILENAME,
    SHARD_FRONTIER_FILENAME,
    build_consumer_manifest,
    stage_incremental_window,
)
from dataset.src.remote import sha256_path
from dataset.src.storage import write_json_atomic

CPU_STAGE_MAX_SECONDS = 24 * 60 * 60


def _load(path: Path, *, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as error:
        raise RuntimeError(f"cannot read {label}: {path}") from error
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"{label} must contain an object")
    return dict(payload)


def _stable_consumer_manifest(
    contract: Mapping[str, object],
    frontier: Mapping[str, object],
) -> dict[str, object]:
    """Return trainer-facing metadata that does not change when production completes."""

    manifest = build_consumer_manifest(contract=contract, frontier=frontier)
    production = manifest.get("production")
    if not isinstance(production, dict):
        raise RuntimeError("incremental consumer manifest has no production identity")
    # Completion is mutable control-plane state and therefore cannot participate
    # in the trainer/checkpoint identity. The live value remains in shard_frontier.json.
    production["incremental_producer_complete"] = False
    return manifest


def stage_incremental_window_when_ready(
    *,
    store: object,
    run_id: str,
    destination: Path,
    start_block_id: int,
    timeout_seconds: float = CPU_STAGE_MAX_SECONDS,
    poll_seconds: float = 5.0,
) -> dict[str, object]:
    """Wait on producer bootstrap metadata entirely on CPU, then stage the lead window.

    Only expected not-yet-published conditions are retried. Identity,
    monotonicity, checksum, and all other integrity failures remain fail-closed.
    """

    if timeout_seconds <= 0 or poll_seconds <= 0:
        raise ValueError("incremental CPU-stage wait intervals must be positive")
    # The stager and producer are intentionally concurrent. If the stager wins
    # even the bucket-creation race, make that idempotently ready here rather
    # than treating normal startup ordering as a dataset failure.
    ensure_bucket = getattr(store, "ensure_bucket", None)
    if callable(ensure_bucket):
        ensure_bucket()
    deadline = time.monotonic() + timeout_seconds
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("timed out waiting for incremental producer bootstrap metadata")
        try:
            result = stage_incremental_window(
                store=store,
                run_id=run_id,
                destination=destination,
                start_block_id=start_block_id,
                timeout_seconds=remaining,
                poll_seconds=poll_seconds,
            )
            if result.get("status") == "ready":
                contract = _load(destination / RUN_CONTRACT_FILENAME, label="incremental run contract")
                frontier = _load(destination / SHARD_FRONTIER_FILENAME, label="incremental shard frontier")
                write_json_atomic(
                    destination / "manifest.json",
                    _stable_consumer_manifest(contract, frontier),
                )
            return result
        except RuntimeError as error:
            message = str(error)
            retryable = (
                "incremental dataset run contract is not ready" in message
                or "incremental shard frontier is not ready" in message
            )
            if not retryable:
                raise
            time.sleep(min(poll_seconds, max(0.0, remaining)))


def _shards(frontier: Mapping[str, object], field: str) -> list[dict[str, object]]:
    raw = frontier.get(field)
    if not isinstance(raw, list):
        raise RuntimeError(f"incremental frontier has invalid {field}")
    result: list[dict[str, object]] = []
    for row in raw:
        if not isinstance(row, Mapping):
            raise RuntimeError(f"incremental frontier {field} contains a malformed shard")
        result.append(dict(row))
    return result


def _matches(root: Path, row: Mapping[str, object]) -> bool:
    filename = row.get("filename")
    byte_size = row.get("byte_size")
    checksum = row.get("checksum")
    if not isinstance(filename, str) or not isinstance(byte_size, int) or not isinstance(checksum, str):
        return False
    path = root / filename
    if path.is_symlink() or not path.is_file():
        return False
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return path.stat().st_size == byte_size and sha256_path(path) == checksum


def verify_incremental_stage(
    *,
    destination: Path,
    bucket_id: str,
    run_id: str,
    required_train_block: int,
) -> dict[str, object]:
    """Re-hash the checkpoint-aligned lead window before H100 work begins."""

    marker = _load(destination / "rolling_cache_stage.json", label="incremental stage marker")
    contract = _load(destination / RUN_CONTRACT_FILENAME, label="incremental run contract")
    frontier = _load(destination / SHARD_FRONTIER_FILENAME, label="incremental shard frontier")
    manifest = _load(destination / "manifest.json", label="incremental consumer manifest")
    if marker.get("version") != 2 or marker.get("transport") != "hf-bucket-incremental-frontier-v1":
        raise RuntimeError("incremental stage marker has the wrong version or transport")
    if marker.get("bucket_id") != bucket_id or marker.get("run_id") != run_id:
        raise RuntimeError("incremental stage marker identity mismatch")
    if marker.get("contract_sha256") != contract.get("contract_sha256"):
        raise RuntimeError("incremental stage marker/contract identity mismatch")
    if frontier.get("contract_sha256") != contract.get("contract_sha256"):
        raise RuntimeError("incremental frontier/contract identity mismatch")
    expected_manifest = _stable_consumer_manifest(contract, frontier)
    if manifest != expected_manifest:
        raise RuntimeError("incremental consumer manifest changed after CPU staging")

    train = _shards(frontier, "ready_train_shards")
    validation = _shards(frontier, "frozen_validation_shards")
    current_index: int | None = None
    for index, row in enumerate(train):
        first, last = row.get("first_block_id"), row.get("last_block_id")
        if isinstance(first, int) and isinstance(last, int) and first <= required_train_block <= last:
            current_index = index
            break
    if current_index is None:
        raise RuntimeError("CPU-staged frontier does not contain the checkpoint-aligned train block")
    planned = int(contract["planned_train_blocks"])
    selected = [train[current_index]]
    if int(train[current_index]["last_block_id"]) + 1 < planned:
        if current_index + 1 >= len(train):
            raise RuntimeError("CPU-staged incremental frontier has no successor lead shard")
        selected.append(train[current_index + 1])
    for row in selected:
        if not _matches(destination, row):
            raise RuntimeError(f"CPU-staged incremental train shard is missing or corrupt: {row.get('filename')}")
    if not validation:
        raise RuntimeError("CPU-staged incremental validation set is not frozen")
    for row in validation:
        if not _matches(destination, row):
            raise RuntimeError(
                f"CPU-staged incremental validation shard is missing or corrupt: {row.get('filename')}"
            )
    return {
        "status": "verified",
        "contract_sha256": contract["contract_sha256"],
        "required_train_block": required_train_block,
        "staged_train_shards": [row["filename"] for row in selected],
        "validation_shards": len(validation),
        "planned_train_blocks": planned,
        "producer_complete": bool(frontier.get("producer_complete")),
    }


__all__ = [
    "CPU_STAGE_MAX_SECONDS",
    "_stable_consumer_manifest",
    "stage_incremental_window_when_ready",
    "verify_incremental_stage",
]
