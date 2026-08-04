"""Tests for exact schedule derivation from the completed 20M manifest."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dataset.qualification_20m_report import derive_plan


def _shard(
    split: str,
    first: int,
    last: int,
    sequences: int,
) -> dict[str, object]:
    tokens = sequences * 2049
    return {
        "filename": f"{split}/{split}-{first:06d}.bin",
        "split": split,
        "byte_size": tokens * 2,
        "token_count": tokens,
        "sequence_count": sequences,
        "checksum": "a" * 64,
        "first_block_id": first,
        "last_block_id": last,
    }


def _manifest(*, final_train_sequences: int = 16) -> dict[str, object]:
    # 305 train blocks: two 100-block shards and a final 105-block shard.
    final_shard_sequences = 104 * 16 + final_train_sequences
    return {
        "schema_version": 2,
        "sequence_format": "context_plus_one",
        "context_length": 2048,
        "stored_tokens_per_sequence": 2049,
        "sequences_per_block": 16,
        "target_shard_bytes": 8_388_608,
        "accepted_source_tokens": 10_000_100,
        "validation_source_tokens": 10_000,
        "production": {
            "run_id": "20m-qualification-dataset-001",
            "configuration_hash": "b" * 64,
            "schema_hash": "c" * 64,
            "target_source_tokens": 10_000_000,
            "minimum_source_tokens": 9_000_000,
            "maximum_source_tokens": 11_000_000,
            "checkpoint_source_tokens": 2_000_000,
            "target_reached": True,
            "remote_required": True,
        },
        "shards": [
            _shard("train", 0, 99, 100 * 16),
            _shard("train", 100, 199, 100 * 16),
            _shard("train", 200, 304, final_shard_sequences),
            _shard("validation", 0, 0, 4),
        ],
    }


def _drive_manifest(manifest: dict[str, object]) -> dict[str, object]:
    production = manifest["production"]
    return {
        "version": 1,
        "run_id": production["run_id"],
        "configuration_hash": production["configuration_hash"],
        "schema_hash": production["schema_hash"],
        "shards": [
            {
                "filename": shard["filename"],
                "drive_file_id": f"drive-{index}",
                "byte_size": shard["byte_size"],
                "local_sha256": shard["checksum"],
                "remote_durable": True,
                "configuration_hash": production["configuration_hash"],
                "schema_hash": production["schema_hash"],
            }
            for index, shard in enumerate(manifest["shards"])
        ],
    }


class Qualification20MReportTests(unittest.TestCase):
    def test_full_final_block_derives_exact_schedule(self) -> None:
        plan = derive_plan(_manifest())
        trainer = plan["trainer"]
        self.assertEqual(trainer["steps"], 305)
        self.assertEqual(trainer["warmup_updates"], 16)
        self.assertEqual(trainer["stable_updates"], 228)
        self.assertEqual(trainer["decay_updates"], 61)
        self.assertEqual(trainer["warmup_tokens"], 524_288)
        self.assertEqual(trainer["stable_tokens"], 7_471_104)
        self.assertEqual(trainer["decay_tokens"], 1_998_848)
        self.assertEqual(trainer["validation_blocks"], 1)
        self.assertEqual(plan["train"]["block_ids"], list(range(305)))

    def test_partial_final_block_changes_only_exact_tail_tokens(self) -> None:
        plan = derive_plan(_manifest(final_train_sequences=7))
        trainer = plan["trainer"]
        self.assertEqual(trainer["steps"], 305)
        self.assertEqual(trainer["warmup_tokens"], 524_288)
        self.assertEqual(trainer["stable_tokens"], 7_471_104)
        self.assertEqual(trainer["decay_tokens"], 1_980_416)
        self.assertEqual(plan["train"]["sequence_count"], 4_871)

    def test_matching_drive_manifest_is_bound_into_identity(self) -> None:
        manifest = _manifest()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "drive_manifest.json"
            path.write_text(json.dumps(_drive_manifest(manifest)), encoding="utf-8")
            plan = derive_plan(manifest, drive_manifest_path=path)
        self.assertEqual(plan["identity"]["drive_shard_count"], 4)
        self.assertEqual(
            plan["identity"]["drive_run_id"],
            "20m-qualification-dataset-001",
        )

    def test_drive_checksum_mismatch_fails_closed(self) -> None:
        manifest = _manifest()
        drive = _drive_manifest(manifest)
        drive["shards"][0]["local_sha256"] = "d" * 64
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "drive_manifest.json"
            path.write_text(json.dumps(drive), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                derive_plan(manifest, drive_manifest_path=path)

    def test_wrong_block_geometry_fails_closed(self) -> None:
        manifest = _manifest()
        manifest["sequences_per_block"] = 32
        with self.assertRaisesRegex(ValueError, "sequences_per_block mismatch"):
            derive_plan(manifest)

    def test_missing_validation_block_fails_closed(self) -> None:
        manifest = _manifest()
        manifest["shards"] = [
            item for item in manifest["shards"] if item["split"] == "train"
        ]
        with self.assertRaisesRegex(ValueError, "no validation block"):
            derive_plan(manifest)


if __name__ == "__main__":
    unittest.main()
