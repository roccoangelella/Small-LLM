"""Regression coverage for byte-preserving block-64 Modal dataset derivation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from dataset.reblock import reblock_dataset


def _write_shard(path: Path, *, sequences: int) -> tuple[int, int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    token_count = sequences * 2049
    payload = b"\x00\x00" * token_count
    path.write_bytes(payload)
    return len(payload), token_count, hashlib.sha256(payload).hexdigest()


def _source_dataset(root: Path) -> None:
    train_sequences = 96 * 16
    validation_sequences = 4 * 16
    train_size, train_tokens, train_sha = _write_shard(
        root / "train" / "train-000000.bin",
        sequences=train_sequences,
    )
    validation_size, validation_tokens, validation_sha = _write_shard(
        root / "validation" / "validation-000000.bin",
        sequences=validation_sequences,
    )
    train_source_tokens = 1_999_000_000
    validation_source_tokens = 1_000_000
    manifest = {
        "schema_version": 2,
        "sequence_format": "context_plus_one",
        "context_length": 2048,
        "stored_tokens_per_sequence": 2049,
        "sequences_per_block": 16,
        "target_shard_bytes": 8 * 1024 * 1024,
        "accepted_source_tokens": 2_000_000_000,
        "validation_source_tokens": validation_source_tokens,
        "scheduler": {"total_emitted_source_tokens": train_source_tokens},
        "work_plan_hash": "d" * 64,
        "production": {
            "version": 1,
            "run_id": "20m-2b-dataset-001",
            "configuration_hash": "b" * 64,
            "schema_hash": "c" * 64,
            "target_source_tokens": 2_000_000_000,
            "minimum_source_tokens": 1_800_000_000,
            "maximum_source_tokens": 2_200_000_000,
            "checkpoint_source_tokens": 80_000_000,
            "target_reached": True,
            "completion_reason": "target_reached",
            "remote_required": True,
        },
        "shards": [
            {
                "filename": "train/train-000000.bin",
                "split": "train",
                "byte_size": train_size,
                "token_count": train_tokens,
                "sequence_count": train_sequences,
                "checksum": train_sha,
                "first_block_id": 0,
                "last_block_id": 95,
                "context_length": 2048,
                "int_type": "uint16",
                "byte_order": "little",
                "cumulative_cluster_source_tokens": {"1": train_source_tokens},
                "shard_cluster_source_tokens": {"1": train_source_tokens},
            },
            {
                "filename": "validation/validation-000000.bin",
                "split": "validation",
                "byte_size": validation_size,
                "token_count": validation_tokens,
                "sequence_count": validation_sequences,
                "checksum": validation_sha,
                "first_block_id": 0,
                "last_block_id": 3,
                "context_length": 2048,
                "int_type": "uint16",
                "byte_order": "little",
                "cumulative_cluster_source_tokens": {"1": validation_source_tokens},
                "shard_cluster_source_tokens": {"1": validation_source_tokens},
            },
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_reblock_preserves_bytes_and_uses_64_sequence_optimizer_blocks(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    _source_dataset(source)

    result = reblock_dataset(source, target)

    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    drive = json.loads((target / "drive_manifest.json").read_text(encoding="utf-8"))
    assert result["run_id"] == "modal-2b-b64-dataset-001"
    assert result["sequences_per_block"] == 64
    assert result["train_blocks"] == 24
    assert result["validation_blocks"] == 1
    assert manifest["sequences_per_block"] == 64
    assert manifest["target_shard_bytes"] == 32 * 1024 * 1024
    assert manifest["production"]["run_id"] == "modal-2b-b64-dataset-001"
    assert manifest["reblock"]["byte_preserving"] is True
    assert drive["transport"] == "modal_volume"
    assert drive["volume_name"] == "small-llm-data"
    assert (source / "train" / "train-000000.bin").read_bytes() == (
        target / "train" / "train-000000.bin"
    ).read_bytes()
    assert (source / "validation" / "validation-000000.bin").read_bytes() == (
        target / "validation" / "validation-000000.bin"
    ).read_bytes()
