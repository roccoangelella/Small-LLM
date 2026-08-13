"""Incremental producer/consumer contract for remotely durable dataset shards.

This module owns the provider-neutral semantics needed to let a trainer consume
an immutable prefix while the deterministic producer is still extending the
corpus.  Hugging Face is only the durable object store; Modal-specific staging
and GPU dispatch remain outside :mod:`dataset`.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol

from dataset.src.remote import ensure_safe_directory, sha256_path
from dataset.src.storage import write_json_atomic

RUN_CONTRACT_FILENAME = "run_contract.json"
SHARD_FRONTIER_FILENAME = "shard_frontier.json"
INCREMENTAL_CONSUMER_MANIFEST_FILENAME = "manifest.json"
FRONTIER_VERSION = 1
CONTRACT_VERSION = 1
DEFAULT_TRAINING_VALIDATION_BLOCKS = 16


class FrontierStore(Protocol):
    bucket_id: str

    @staticmethod
    def object_key(run_id: str, logical_name: str) -> str: ...

    def download_shard(
        self,
        *,
        run_id: str,
        logical_name: str,
        file_id: str,
        destination: Path,
        byte_size: int,
        sha256: str,
    ) -> None: ...

    def _write_json(self, key: str, payload: Mapping[str, object]) -> None: ...
    def _read_json(self, key: str) -> dict[str, object] | None: ...


@dataclass(frozen=True, slots=True)
class FrontierShard:
    filename: str
    split: str
    byte_size: int
    checksum: str
    first_block_id: int
    last_block_id: int
    sequence_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "filename": self.filename,
            "split": self.split,
            "byte_size": self.byte_size,
            "checksum": self.checksum,
            "first_block_id": self.first_block_id,
            "last_block_id": self.last_block_id,
            "sequence_count": self.sequence_count,
        }


def _canonical_hash(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def planned_train_blocks(
    nominal_target_tokens: int,
    *,
    context_length: int,
    sequences_per_block: int,
) -> int:
    if min(nominal_target_tokens, context_length, sequences_per_block) <= 0:
        raise ValueError("incremental training horizon inputs must be positive")
    per_block = context_length * sequences_per_block
    return math.ceil(nominal_target_tokens / per_block)


def standard_wsd_plan(
    blocks: int,
    *,
    context_length: int,
    sequences_per_block: int,
    validation_blocks: int,
) -> dict[str, object]:
    if blocks <= 0 or validation_blocks <= 0:
        raise ValueError("incremental trainer plan requires positive train/validation blocks")
    full_block_target_tokens = context_length * sequences_per_block
    warmup_updates = max(16, math.ceil(blocks * 0.05))
    decay_updates = math.ceil(blocks * 0.20)
    if warmup_updates + decay_updates >= blocks:
        raise ValueError("incremental horizon is too short for the standard WSD phases")
    stable_updates = blocks - warmup_updates - decay_updates
    return {
        "steps": blocks,
        "passes": 1,
        "full_block_target_tokens": full_block_target_tokens,
        "schedule": "wsd",
        "warmup_updates": warmup_updates,
        "stable_updates": stable_updates,
        "decay_updates": decay_updates,
        "warmup_tokens": warmup_updates * full_block_target_tokens,
        "stable_tokens": stable_updates * full_block_target_tokens,
        "decay_tokens": decay_updates * full_block_target_tokens,
        "minimum_lr_ratio": 0.1,
        "validation_blocks": validation_blocks,
        "planned_target_tokens": blocks * full_block_target_tokens,
    }


def build_run_contract(
    *,
    run_id: str,
    nominal_training_tokens: int,
    target_source_tokens: int,
    minimum_source_tokens: int,
    maximum_source_tokens: int,
    checkpoint_source_tokens: int,
    context_length: int,
    sequences_per_block: int,
    target_shard_bytes: int,
    configuration_hash: str,
    schema_hash: str,
    work_plan_hash: str,
    validation_blocks: int = DEFAULT_TRAINING_VALIDATION_BLOCKS,
) -> dict[str, object]:
    blocks = planned_train_blocks(
        nominal_training_tokens,
        context_length=context_length,
        sequences_per_block=sequences_per_block,
    )
    trainer = standard_wsd_plan(
        blocks,
        context_length=context_length,
        sequences_per_block=sequences_per_block,
        validation_blocks=validation_blocks,
    )
    contract: dict[str, object] = {
        "version": CONTRACT_VERSION,
        "kind": "incremental-schema-v2-prefix",
        "run_id": run_id,
        "nominal_training_tokens": nominal_training_tokens,
        "planned_train_blocks": blocks,
        "planned_train_target_tokens": trainer["planned_target_tokens"],
        "schema_version": 2,
        "sequence_format": "context_plus_one",
        "context_length": context_length,
        "stored_tokens_per_sequence": context_length + 1,
        "sequences_per_block": sequences_per_block,
        "target_shard_bytes": target_shard_bytes,
        "configuration_hash": configuration_hash,
        "schema_hash": schema_hash,
        "work_plan_hash": work_plan_hash,
        "source_policy": {
            "target_source_tokens": target_source_tokens,
            "minimum_source_tokens": minimum_source_tokens,
            "maximum_source_tokens": maximum_source_tokens,
            "checkpoint_source_tokens": checkpoint_source_tokens,
        },
        "trainer": trainer,
        "frontier_policy": {
            "minimum_ready_train_shards_before_gpu": 2,
            "training_validation_blocks": validation_blocks,
            "ready_entries_are_immutable": True,
            "consumer_blocks_on_missing_future_ready_shard": True,
        },
    }
    contract["contract_sha256"] = _canonical_hash(contract)
    return contract


def _require_shard(row: Mapping[str, object]) -> FrontierShard:
    filename = row.get("filename")
    split = row.get("split")
    byte_size = row.get("byte_size")
    checksum = row.get("checksum", row.get("local_sha256"))
    first = row.get("first_block_id")
    last = row.get("last_block_id")
    sequences = row.get("sequence_count")
    if not isinstance(filename, str) or not filename:
        raise RuntimeError("frontier shard has an invalid filename")
    if split not in {"train", "validation"}:
        raise RuntimeError(f"frontier shard has an invalid split: {filename}")
    if isinstance(byte_size, bool) or not isinstance(byte_size, int) or byte_size <= 0:
        raise RuntimeError(f"frontier shard has an invalid byte size: {filename}")
    if not isinstance(checksum, str) or len(checksum) != 64:
        raise RuntimeError(f"frontier shard has an invalid checksum: {filename}")
    if (
        isinstance(first, bool) or not isinstance(first, int) or first < 0
        or isinstance(last, bool) or not isinstance(last, int) or last < first
        or isinstance(sequences, bool) or not isinstance(sequences, int) or sequences <= 0
    ):
        raise RuntimeError(f"frontier shard has invalid block geometry: {filename}")
    return FrontierShard(filename, str(split), byte_size, checksum, first, last, sequences)


def _sorted_contiguous(rows: list[FrontierShard], *, split: str) -> list[FrontierShard]:
    selected = sorted((row for row in rows if row.split == split), key=lambda row: row.first_block_id)
    expected = 0
    names: set[str] = set()
    for row in selected:
        if row.filename in names or row.first_block_id != expected:
            raise RuntimeError(f"{split} frontier is not a unique contiguous block prefix")
        names.add(row.filename)
        expected = row.last_block_id + 1
    return selected


def _verified_rows(durability_manifest: Mapping[str, object]) -> list[FrontierShard]:
    raw = durability_manifest.get("shards")
    if not isinstance(raw, list):
        raise RuntimeError("durability manifest has no shard list")
    rows: list[FrontierShard] = []
    for item in raw:
        if not isinstance(item, Mapping) or item.get("remote_durable") is not True:
            raise RuntimeError("durability manifest contains a non-durable shard entry")
        rows.append(_require_shard(item))
    return rows


def _prefix_for_blocks(rows: list[FrontierShard], blocks: int) -> list[FrontierShard]:
    if blocks <= 0:
        return []
    result: list[FrontierShard] = []
    covered = 0
    for row in rows:
        result.append(row)
        covered = row.last_block_id + 1
        if covered >= blocks:
            return result
    return []


def _object(store: FrontierStore, run_id: str, name: str) -> str:
    return store.object_key(run_id, name)


def publish_run_contract(
    store: FrontierStore,
    *,
    run_id: str,
    contract: Mapping[str, object],
) -> dict[str, object]:
    payload = dict(contract)
    if payload.get("run_id") != run_id or payload.get("version") != CONTRACT_VERSION:
        raise RuntimeError("incremental run contract identity is invalid")
    key = _object(store, run_id, RUN_CONTRACT_FILENAME)
    existing = store._read_json(key)
    if existing is not None and existing != payload:
        raise RuntimeError("refusing to mutate an existing incremental dataset run contract")
    if existing is None:
        store._write_json(key, payload)
    observed = store._read_json(key)
    if observed != payload:
        raise RuntimeError("incremental dataset run contract read-back mismatch")
    return payload


def read_run_contract(store: FrontierStore, *, run_id: str) -> dict[str, object]:
    payload = store._read_json(_object(store, run_id, RUN_CONTRACT_FILENAME))
    if payload is None or payload.get("run_id") != run_id:
        raise RuntimeError(f"incremental dataset run contract is not ready: {run_id}")
    expected_hash = payload.get("contract_sha256")
    without_hash = dict(payload)
    without_hash.pop("contract_sha256", None)
    if expected_hash != _canonical_hash(without_hash):
        raise RuntimeError("incremental dataset run contract hash is invalid")
    return payload


def publish_frontier(
    store: FrontierStore,
    *,
    run_id: str,
    contract: Mapping[str, object],
    durability_manifest: Mapping[str, object],
    producer_complete: bool = False,
    final_manifest_sha256: str | None = None,
) -> dict[str, object]:
    if contract.get("run_id") != run_id:
        raise RuntimeError("frontier contract/run ID mismatch")
    rows = _verified_rows(durability_manifest)
    train = _sorted_contiguous(rows, split="train")
    validation = _sorted_contiguous(rows, split="validation")
    planned_blocks = int(contract["planned_train_blocks"])
    validation_blocks = int(dict(contract["trainer"])["validation_blocks"])
    frozen_validation = _prefix_for_blocks(validation, validation_blocks)

    key = _object(store, run_id, SHARD_FRONTIER_FILENAME)
    previous = store._read_json(key)
    previous_train: list[Mapping[str, object]] = []
    previous_validation: list[Mapping[str, object]] = []
    if previous is not None:
        if previous.get("run_id") != run_id or previous.get("contract_sha256") != contract.get("contract_sha256"):
            raise RuntimeError("existing shard frontier belongs to a different run contract")
        raw_previous_train = previous.get("ready_train_shards", [])
        raw_previous_validation = previous.get("frozen_validation_shards", [])
        if not isinstance(raw_previous_train, list) or not isinstance(raw_previous_validation, list):
            raise RuntimeError("existing shard frontier has invalid shard lists")
        previous_train = [item for item in raw_previous_train if isinstance(item, Mapping)]
        previous_validation = [item for item in raw_previous_validation if isinstance(item, Mapping)]

    train_payload = [row.as_dict() for row in train]
    if train_payload[: len(previous_train)] != [dict(row) for row in previous_train]:
        raise RuntimeError("READY training shard metadata is not an immutable monotonic prefix")

    if previous_validation:
        validation_payload = [dict(row) for row in previous_validation]
        current_by_name = {row.filename: row.as_dict() for row in validation}
        for row in validation_payload:
            if current_by_name.get(str(row.get("filename"))) != row:
                raise RuntimeError("frozen validation shard changed after publication")
    else:
        validation_payload = [row.as_dict() for row in frozen_validation]

    last_ready = train[-1].last_block_id if train else -1
    frontier = {
        "version": FRONTIER_VERSION,
        "run_id": run_id,
        "contract_sha256": contract.get("contract_sha256"),
        "ready_train_shards": train_payload,
        "last_ready_train_block_id": last_ready,
        "frozen_validation_shards": validation_payload,
        "validation_ready": bool(validation_payload),
        "planned_train_blocks": planned_blocks,
        "producer_complete": bool(producer_complete),
        "final_manifest_sha256": final_manifest_sha256 if producer_complete else None,
    }
    if previous is not None:
        if previous.get("producer_complete") is True and frontier != previous:
            raise RuntimeError("completed shard frontier is immutable")
        if previous.get("producer_complete") is True and not producer_complete:
            raise RuntimeError("completed shard frontier cannot return to active state")
    if producer_complete:
        if last_ready + 1 < planned_blocks:
            raise RuntimeError("producer completed before the frozen training horizon became durable")
        if not isinstance(final_manifest_sha256, str) or len(final_manifest_sha256) != 64:
            raise RuntimeError("completed frontier requires the final manifest SHA-256")
        if not validation_payload:
            raise RuntimeError("producer completed without a frozen training-validation set")

    store._write_json(key, frontier)
    observed = store._read_json(key)
    if observed != frontier:
        raise RuntimeError("shard frontier read-back mismatch")
    return frontier


def read_frontier(store: FrontierStore, *, run_id: str, contract: Mapping[str, object]) -> dict[str, object]:
    payload = store._read_json(_object(store, run_id, SHARD_FRONTIER_FILENAME))
    if payload is None:
        raise RuntimeError(f"incremental shard frontier is not ready: {run_id}")
    if payload.get("run_id") != run_id or payload.get("contract_sha256") != contract.get("contract_sha256"):
        raise RuntimeError("incremental shard frontier identity mismatch")
    return payload


def _frontier_shards(frontier: Mapping[str, object], field: str) -> list[FrontierShard]:
    raw = frontier.get(field)
    if not isinstance(raw, list):
        raise RuntimeError(f"incremental frontier has invalid {field}")
    return [_require_shard(item) for item in raw if isinstance(item, Mapping)]


def build_consumer_manifest(
    *,
    contract: Mapping[str, object],
    frontier: Mapping[str, object],
) -> dict[str, object]:
    """Build the stable trainer identity manifest from frozen bootstrap data.

    The first two training shards and the frozen validation inventory never
    change.  The online train reader uses the mutable frontier for later blocks,
    while checkpoint identity remains bound to this immutable contract manifest.
    """

    train = _frontier_shards(frontier, "ready_train_shards")
    validation = _frontier_shards(frontier, "frozen_validation_shards")
    minimum = int(dict(contract["frontier_policy"])["minimum_ready_train_shards_before_gpu"])
    if len(train) < minimum or not validation:
        raise RuntimeError("incremental consumer manifest requested before the GPU lead buffer is ready")
    source_policy = dict(contract["source_policy"])
    bootstrap = train[:minimum]
    return {
        "schema_version": 2,
        "sequence_format": "context_plus_one",
        "context_length": contract["context_length"],
        "stored_tokens_per_sequence": contract["stored_tokens_per_sequence"],
        "sequences_per_block": contract["sequences_per_block"],
        "target_shard_bytes": contract["target_shard_bytes"],
        "work_plan_hash": contract["work_plan_hash"],
        "incremental_frontier": {
            "version": FRONTIER_VERSION,
            "run_id": contract["run_id"],
            "contract_sha256": contract["contract_sha256"],
            "planned_train_blocks": contract["planned_train_blocks"],
        },
        "production": {
            "version": 1,
            "run_id": contract["run_id"],
            "configuration_hash": contract["configuration_hash"],
            "schema_hash": contract["schema_hash"],
            "target_source_tokens": source_policy["target_source_tokens"],
            "minimum_source_tokens": source_policy["minimum_source_tokens"],
            "maximum_source_tokens": source_policy["maximum_source_tokens"],
            "checkpoint_source_tokens": source_policy["checkpoint_source_tokens"],
            # This manifest represents the frozen training contract, not producer completion.
            "target_reached": True,
            "remote_required": True,
            "incremental_producer_complete": bool(frontier.get("producer_complete")),
        },
        "shards": [row.as_dict() for row in bootstrap + validation],
    }


def write_consumer_snapshot(
    *,
    destination: Path,
    contract: Mapping[str, object],
    frontier: Mapping[str, object],
) -> Path:
    destination = ensure_safe_directory(destination)
    write_json_atomic(destination / RUN_CONTRACT_FILENAME, dict(contract))
    write_json_atomic(destination / SHARD_FRONTIER_FILENAME, dict(frontier))
    write_json_atomic(
        destination / INCREMENTAL_CONSUMER_MANIFEST_FILENAME,
        build_consumer_manifest(contract=contract, frontier=frontier),
    )
    return destination / INCREMENTAL_CONSUMER_MANIFEST_FILENAME


def _file_matches(root: Path, shard: FrontierShard) -> bool:
    path = root / shard.filename
    if path.is_symlink() or not path.is_file():
        return False
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return path.stat().st_size == shard.byte_size and sha256_path(path) == shard.checksum


def _download_verified(
    store: FrontierStore,
    *,
    run_id: str,
    root: Path,
    shard: FrontierShard,
) -> Path:
    destination = root / shard.filename
    if _file_matches(root, shard):
        return destination
    if destination.exists() or destination.is_symlink():
        if destination.is_dir() and not destination.is_symlink():
            raise RuntimeError(f"dataset shard destination is unexpectedly a directory: {destination}")
        destination.unlink(missing_ok=True)
    ensure_safe_directory(destination.parent)
    store.download_shard(
        run_id=run_id,
        logical_name=shard.filename,
        file_id=store.object_key(run_id, shard.filename),
        destination=destination,
        byte_size=shard.byte_size,
        sha256=shard.checksum,
    )
    if not _file_matches(root, shard):
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"downloaded incremental dataset shard failed verification: {shard.filename}")
    return destination


def _train_index_for_block(train: list[FrontierShard], block_id: int) -> int | None:
    for index, shard in enumerate(train):
        if shard.first_block_id <= block_id <= shard.last_block_id:
            return index
    return None


def stage_incremental_window(
    *,
    store: FrontierStore,
    run_id: str,
    destination: Path,
    start_block_id: int,
    timeout_seconds: float = 2 * 60 * 60,
    poll_seconds: float = 5.0,
) -> dict[str, object]:
    """CPU-stage current+next READY shards and frozen validation before GPU dispatch."""

    destination = ensure_safe_directory(destination)
    contract = read_run_contract(store, run_id=run_id)
    planned = int(contract["planned_train_blocks"])
    if start_block_id >= planned:
        return {
            "status": "training_complete",
            "dataset_dir": str(destination),
            "run_id": run_id,
            "start_block_id": start_block_id,
            "training_complete": True,
        }
    deadline = time.monotonic() + timeout_seconds
    minimum = int(dict(contract["frontier_policy"])["minimum_ready_train_shards_before_gpu"])
    while True:
        frontier = read_frontier(store, run_id=run_id, contract=contract)
        train = _frontier_shards(frontier, "ready_train_shards")
        validation = _frontier_shards(frontier, "frozen_validation_shards")
        index = _train_index_for_block(train, start_block_id)
        enough = index is not None and len(validation) > 0
        if enough:
            assert index is not None
            successors_expected = train[index].last_block_id + 1 < planned
            enough = (not successors_expected) or index + 1 < len(train) or frontier.get("producer_complete") is True
        if enough:
            break
        if frontier.get("producer_complete") is True:
            raise RuntimeError("completed producer frontier does not cover the checkpoint-aligned GPU lead window")
        if time.monotonic() >= deadline:
            raise TimeoutError("timed out waiting for the incremental dataset lead buffer on CPU")
        time.sleep(poll_seconds)

    assert index is not None
    selected = train[index : min(len(train), index + minimum)]
    for shard in validation:
        _download_verified(store, run_id=run_id, root=destination, shard=shard)
    for shard in selected:
        _download_verified(store, run_id=run_id, root=destination, shard=shard)
    write_consumer_snapshot(destination=destination, contract=contract, frontier=frontier)
    marker = {
        "version": 2,
        "transport": "hf-bucket-incremental-frontier-v1",
        "bucket_id": store.bucket_id,
        "run_id": run_id,
        "contract_sha256": contract["contract_sha256"],
        "start_block_id": start_block_id,
        "training_complete": False,
        "staged_train_shards": [row.filename for row in selected],
        "validation_shards": [row.filename for row in validation],
        "planned_train_blocks": planned,
    }
    write_json_atomic(destination / "rolling_cache_stage.json", marker)
    return {"status": "ready", "dataset_dir": str(destination), **marker}


class IncrementalRollingShardCache:
    """Dynamic current+next cache backed by a monotonic remote READY frontier."""

    def __init__(
        self,
        *,
        root: Path,
        run_id: str,
        contract: Mapping[str, object],
        store: FrontierStore,
        prefetch_shards: int = 1,
        poll_seconds: float = 5.0,
    ) -> None:
        if prefetch_shards < 1:
            raise ValueError("incremental prefetch_shards must be at least one")
        self.root = ensure_safe_directory(root)
        self.run_id = run_id
        self.contract = dict(contract)
        self.store = store
        self.prefetch_shards = prefetch_shards
        self.poll_seconds = poll_seconds
        self.planned_block_count = int(contract["planned_train_blocks"])
        self._lock = __import__("threading").Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dataset-frontier-prefetch")
        self._futures: dict[int, Future[Path]] = {}
        self._last_frontier: dict[str, object] | None = None

    def _frontier(self) -> dict[str, object]:
        frontier = read_frontier(self.store, run_id=self.run_id, contract=self.contract)
        previous = self._last_frontier
        if previous is not None:
            old = previous.get("ready_train_shards", [])
            new = frontier.get("ready_train_shards", [])
            if not isinstance(old, list) or not isinstance(new, list) or new[: len(old)] != old:
                raise RuntimeError("remote training frontier regressed or mutated")
        self._last_frontier = frontier
        return frontier

    def _wait_for_shard(self, block_id: int) -> FrontierShard:
        while True:
            frontier = self._frontier()
            train = _frontier_shards(frontier, "ready_train_shards")
            index = _train_index_for_block(train, block_id)
            if index is not None:
                return train[index]
            if frontier.get("producer_complete") is True:
                raise RuntimeError(f"producer completed without required train block {block_id}")
            time.sleep(self.poll_seconds)

    def shard_for_block(self, block_id: int) -> FrontierShard:
        if block_id < 0 or block_id >= self.planned_block_count:
            raise RuntimeError(f"incremental train block {block_id} is outside the frozen horizon")
        return self._wait_for_shard(block_id)

    def _download_block_shard(self, block_id: int) -> Path:
        shard = self._wait_for_shard(block_id)
        return _download_verified(self.store, run_id=self.run_id, root=self.root, shard=shard)

    def _future(self, block_id: int) -> Future[Path]:
        with self._lock:
            future = self._futures.get(block_id)
            if future is None:
                future = self._executor.submit(self._download_block_shard, block_id)
                self._futures[block_id] = future
            return future

    def ensure_block(self, block_id: int) -> None:
        self._future(block_id).result()
        shard = self.shard_for_block(block_id)
        next_block = shard.last_block_id + 1
        if next_block < self.planned_block_count:
            self._future(next_block)

    def acknowledge(self, block_id: int) -> None:
        shard = self.shard_for_block(block_id)
        if block_id != shard.last_block_id:
            return
        path = self.root / shard.filename
        if path.is_file() and not path.is_symlink():
            path.unlink()
        next_block = block_id + 1
        if next_block < self.planned_block_count:
            self._future(next_block)

    def restore_after_acknowledged(self, block_id: int) -> None:
        next_block = block_id + 1
        if next_block < self.planned_block_count:
            self._future(next_block).result()

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


__all__ = [
    "CONTRACT_VERSION",
    "DEFAULT_TRAINING_VALIDATION_BLOCKS",
    "FRONTIER_VERSION",
    "FrontierShard",
    "IncrementalRollingShardCache",
    "RUN_CONTRACT_FILENAME",
    "SHARD_FRONTIER_FILENAME",
    "build_consumer_manifest",
    "build_run_contract",
    "planned_train_blocks",
    "publish_frontier",
    "publish_run_contract",
    "read_frontier",
    "read_run_contract",
    "stage_incremental_window",
    "standard_wsd_plan",
    "write_consumer_snapshot",
]
