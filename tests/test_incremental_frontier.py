"""Network-free tests for concurrent 10B shard production and consumption."""

from __future__ import annotations

import copy
import hashlib
import json
import threading
import time
from pathlib import Path

import pytest

from dataset.incremental_frontier import (
    build_consumer_manifest,
    build_run_contract,
    publish_frontier,
    publish_run_contract,
    stage_incremental_window,
)
from dataset.incremental_stage import (
    stage_incremental_window_when_ready,
    verify_incremental_stage,
)
from dataset.production.incremental_builder import _durability_for_committed_state


class FakeStore:
    def __init__(self) -> None:
        self.bucket_id = "fake-datasets"
        self.json: dict[str, dict[str, object]] = {}
        self.blobs: dict[str, bytes] = {}
        self.downloaded: list[str] = []

    @staticmethod
    def object_key(run_id: str, logical_name: str) -> str:
        return f"run/{run_id}/{logical_name}"

    def _write_json(self, key: str, payload: dict[str, object]) -> None:
        self.json[key] = json.loads(json.dumps(payload))

    def _read_json(self, key: str) -> dict[str, object] | None:
        value = self.json.get(key)
        return None if value is None else copy.deepcopy(value)

    def download_shard(
        self,
        *,
        run_id: str,
        logical_name: str,
        file_id: str,
        destination: Path,
        byte_size: int,
        sha256: str,
    ) -> None:
        assert file_id == self.object_key(run_id, logical_name)
        payload = self.blobs[file_id]
        assert len(payload) == byte_size
        assert hashlib.sha256(payload).hexdigest() == sha256
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        self.downloaded.append(logical_name)


def _contract() -> dict[str, object]:
    return build_run_contract(
        run_id="modal-10b-b64-dataset-001",
        nominal_training_tokens=10_000_000_000,
        target_source_tokens=10_000_000_000,
        minimum_source_tokens=9_000_000_000,
        maximum_source_tokens=11_000_000_000,
        checkpoint_source_tokens=500_000_000,
        context_length=2048,
        sequences_per_block=64,
        target_shard_bytes=1024**3,
        configuration_hash="a" * 64,
        schema_hash="b" * 64,
        work_plan_hash="c" * 64,
        validation_blocks=16,
    )


def _entry(filename: str, split: str, first: int, last: int, payload: bytes) -> dict[str, object]:
    checksum = hashlib.sha256(payload).hexdigest()
    blocks = last - first + 1
    sequences = blocks * 64
    return {
        "filename": filename,
        "split": split,
        "byte_size": len(payload),
        "token_count": sequences * 2049,
        "sequence_count": sequences,
        "checksum": checksum,
        "local_sha256": checksum,
        "first_block_id": first,
        "last_block_id": last,
        "context_length": 2048,
        "int_type": "uint16",
        "byte_order": "little",
        "cumulative_cluster_source_tokens": {},
        "shard_cluster_source_tokens": {},
        "remote_durable": True,
        "drive_file_id": f"object:{filename}",
        "configuration_hash": "a" * 64,
        "schema_hash": "b" * 64,
    }


def _durability(entries: list[dict[str, object]]) -> dict[str, object]:
    return {
        "version": 1,
        "run_id": "modal-10b-b64-dataset-001",
        "configuration_hash": "a" * 64,
        "schema_hash": "b" * 64,
        "shards": entries,
    }


def _ready_store() -> tuple[FakeStore, dict[str, object]]:
    store = FakeStore()
    contract = _contract()
    publish_run_contract(store, run_id=str(contract["run_id"]), contract=contract)
    entries: list[dict[str, object]] = []
    for index, first in enumerate((0, 4094, 8188)):
        payload = bytes([index + 1]) * 64
        entry = _entry(
            f"train/train-{index:06d}.bin",
            "train",
            first,
            first + 4093,
            payload,
        )
        entries.append(entry)
        store.blobs[store.object_key(str(contract["run_id"]), str(entry["filename"]))] = payload
    validation_payload = b"v" * 64
    validation = _entry("validation/validation-000000.bin", "validation", 0, 15, validation_payload)
    entries.append(validation)
    store.blobs[store.object_key(str(contract["run_id"]), str(validation["filename"]))] = validation_payload
    publish_frontier(
        store,
        run_id=str(contract["run_id"]),
        contract=contract,
        durability_manifest=_durability(entries),
    )
    return store, contract


def test_10b_contract_freezes_exact_horizon_and_standard_wsd() -> None:
    contract = _contract()
    trainer = contract["trainer"]
    assert isinstance(trainer, dict)
    assert contract["planned_train_blocks"] == 76_294
    assert contract["planned_train_target_tokens"] == 10_000_007_168
    assert trainer["steps"] == 76_294
    assert trainer["warmup_updates"] == 3_815
    assert trainer["stable_updates"] == 57_220
    assert trainer["decay_updates"] == 15_259
    assert trainer["warmup_tokens"] == 500_039_680
    assert trainer["stable_tokens"] == 7_499_939_840
    assert trainer["decay_tokens"] == 2_000_027_648
    assert trainer["validation_blocks"] == 16
    assert contract["frontier_policy"] == {
        "minimum_ready_train_shards_before_gpu": 2,
        "training_validation_blocks": 16,
        "ready_entries_are_immutable": True,
        "consumer_blocks_on_missing_future_ready_shard": True,
    }


def test_run_contract_is_immutable_and_read_back_verified() -> None:
    store = FakeStore()
    contract = _contract()
    assert publish_run_contract(store, run_id=str(contract["run_id"]), contract=contract) == contract
    assert publish_run_contract(store, run_id=str(contract["run_id"]), contract=contract) == contract
    mutated = copy.deepcopy(contract)
    mutated["planned_train_blocks"] = 76_295
    with pytest.raises(RuntimeError, match="refusing to mutate"):
        publish_run_contract(store, run_id=str(contract["run_id"]), contract=mutated)


def test_ready_frontier_only_appends_and_validation_freezes() -> None:
    store = FakeStore()
    contract = _contract()
    publish_run_contract(store, run_id=str(contract["run_id"]), contract=contract)
    train0 = _entry("train/train-000000.bin", "train", 0, 4093, b"0" * 64)
    train1 = _entry("train/train-000001.bin", "train", 4094, 8187, b"1" * 64)
    val0 = _entry("validation/validation-000000.bin", "validation", 0, 15, b"v" * 64)
    first = publish_frontier(
        store,
        run_id=str(contract["run_id"]),
        contract=contract,
        durability_manifest=_durability([train0, train1, val0]),
    )
    assert len(first["ready_train_shards"]) == 2
    assert len(first["frozen_validation_shards"]) == 1

    train2 = _entry("train/train-000002.bin", "train", 8188, 12281, b"2" * 64)
    val1 = _entry("validation/validation-000001.bin", "validation", 16, 31, b"w" * 64)
    second = publish_frontier(
        store,
        run_id=str(contract["run_id"]),
        contract=contract,
        durability_manifest=_durability([train0, train1, train2, val0, val1]),
    )
    assert len(second["ready_train_shards"]) == 3
    assert second["frozen_validation_shards"] == first["frozen_validation_shards"]

    changed = copy.deepcopy(train0)
    changed["checksum"] = changed["local_sha256"] = "f" * 64
    with pytest.raises(RuntimeError, match="immutable monotonic prefix"):
        publish_frontier(
            store,
            run_id=str(contract["run_id"]),
            contract=contract,
            durability_manifest=_durability([changed, train1, train2, val0, val1]),
        )


def test_consumer_manifest_identity_does_not_grow_with_frontier() -> None:
    store = FakeStore()
    contract = _contract()
    train0 = _entry("train/train-000000.bin", "train", 0, 4093, b"0" * 64)
    train1 = _entry("train/train-000001.bin", "train", 4094, 8187, b"1" * 64)
    train2 = _entry("train/train-000002.bin", "train", 8188, 12281, b"2" * 64)
    val0 = _entry("validation/validation-000000.bin", "validation", 0, 15, b"v" * 64)
    first = publish_frontier(
        store,
        run_id=str(contract["run_id"]),
        contract=contract,
        durability_manifest=_durability([train0, train1, val0]),
    )
    before = build_consumer_manifest(contract=contract, frontier=first)
    second = publish_frontier(
        store,
        run_id=str(contract["run_id"]),
        contract=contract,
        durability_manifest=_durability([train0, train1, train2, val0]),
    )
    after = build_consumer_manifest(contract=contract, frontier=second)
    assert before == after
    assert [row["filename"] for row in before["shards"][:2]] == [
        "train/train-000000.bin",
        "train/train-000001.bin",
    ]


def test_cpu_stage_downloads_current_plus_successor_not_complete_frontier(tmp_path: Path) -> None:
    store, contract = _ready_store()
    staged = stage_incremental_window(
        store=store,
        run_id=str(contract["run_id"]),
        destination=tmp_path,
        start_block_id=0,
        timeout_seconds=0.1,
        poll_seconds=0.001,
    )
    assert staged["status"] == "ready"
    assert staged["staged_train_shards"] == [
        "train/train-000000.bin",
        "train/train-000001.bin",
    ]
    assert "train/train-000002.bin" not in store.downloaded
    verification = verify_incremental_stage(
        destination=tmp_path,
        bucket_id=store.bucket_id,
        run_id=str(contract["run_id"]),
        required_train_block=0,
    )
    assert verification["status"] == "verified"
    assert verification["staged_train_shards"] == [
        "train/train-000000.bin",
        "train/train-000001.bin",
    ]


def test_cpu_stage_waits_for_producer_bootstrap_without_hiding_integrity_errors(tmp_path: Path) -> None:
    ready, contract = _ready_store()
    store = FakeStore()

    def publish_later() -> None:
        time.sleep(0.01)
        store.json = copy.deepcopy(ready.json)
        store.blobs = copy.deepcopy(ready.blobs)

    thread = threading.Thread(target=publish_later)
    thread.start()
    try:
        staged = stage_incremental_window_when_ready(
            store=store,
            run_id=str(contract["run_id"]),
            destination=tmp_path,
            start_block_id=0,
            timeout_seconds=1.0,
            poll_seconds=0.002,
        )
    finally:
        thread.join()
    assert staged["status"] == "ready"

    corrupt = FakeStore()
    bad_contract = copy.deepcopy(contract)
    bad_contract["contract_sha256"] = "0" * 64
    corrupt._write_json(corrupt.object_key(str(contract["run_id"]), "run_contract.json"), bad_contract)
    with pytest.raises(RuntimeError, match="hash is invalid"):
        stage_incremental_window_when_ready(
            store=corrupt,
            run_id=str(contract["run_id"]),
            destination=tmp_path / "bad",
            start_block_id=0,
            timeout_seconds=0.1,
            poll_seconds=0.001,
        )


def test_uncommitted_remote_shards_are_not_frontier_visible() -> None:
    train0 = _entry("train/train-000000.bin", "train", 0, 4093, b"0" * 64)
    train1 = _entry("train/train-000001.bin", "train", 4094, 8187, b"1" * 64)
    validation = _entry("validation/validation-000000.bin", "validation", 0, 15, b"v" * 64)
    durability = _durability([train0, train1, validation])
    committed_state = {"finalized_shards": [train0, validation]}

    filtered = _durability_for_committed_state(durability, committed_state)
    assert [row["filename"] for row in filtered["shards"]] == [
        "train/train-000000.bin",
        "validation/validation-000000.bin",
    ]
