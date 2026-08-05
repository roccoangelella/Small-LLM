"""Tests for the fixed 20M-model/100M-token qualification profile."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dataset.qualification_100m import RUN_ID, qualification_arguments
from dataset.qualification_100m_report import derive_plan


def _shard(split: str, first: int, last: int, sequences: int) -> dict[str, object]:
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
    final_shard_sequences = 1_051 * 16 + final_train_sequences
    return {
        "schema_version": 2,
        "sequence_format": "context_plus_one",
        "context_length": 2048,
        "stored_tokens_per_sequence": 2049,
        "sequences_per_block": 16,
        "target_shard_bytes": 8_388_608,
        "accepted_source_tokens": 100_000_100,
        "validation_source_tokens": 100_000,
        "production": {
            "run_id": RUN_ID,
            "configuration_hash": "b" * 64,
            "schema_hash": "c" * 64,
            "target_source_tokens": 100_000_000,
            "minimum_source_tokens": 90_000_000,
            "maximum_source_tokens": 110_000_000,
            "checkpoint_source_tokens": 2_000_000,
            "target_reached": True,
            "remote_required": True,
        },
        "shards": [
            _shard("train", 0, 999, 1_000 * 16),
            _shard("train", 1_000, 1_999, 1_000 * 16),
            _shard("train", 2_000, 3_051, final_shard_sequences),
            _shard("validation", 0, 3, 64),
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


class Qualification100MTests(unittest.TestCase):
    def test_entry_point_locks_exact_10x_source_envelope(self) -> None:
        arguments = qualification_arguments(
            ["--weights-file", "weights.json", "--output-dir", "out"]
        )
        self.assertEqual(
            arguments[-16:],
            [
                "--run-id", RUN_ID,
                "--target-tokens", "100000000",
                "--minimum-tokens", "90000000",
                "--maximum-tokens", "110000000",
                "--checkpoint-source-tokens", "2000000",
                "--context-length", "2048",
                "--sequences-per-block", "16",
                "--target-shard-bytes", "8388608",
            ],
        )

    def test_full_profile_derives_exact_wsd_schedule(self) -> None:
        plan = derive_plan(_manifest())
        trainer = plan["trainer"]
        self.assertEqual(plan["qualification_profile"], "20m-100m-data-scaling-v1")
        self.assertEqual(trainer["steps"], 3_052)
        self.assertEqual(trainer["warmup_updates"], 153)
        self.assertEqual(trainer["stable_updates"], 2_288)
        self.assertEqual(trainer["decay_updates"], 611)
        self.assertEqual(trainer["warmup_tokens"], 5_013_504)
        self.assertEqual(trainer["stable_tokens"], 74_973_184)
        self.assertEqual(trainer["decay_tokens"], 20_021_248)
        self.assertEqual(trainer["validation_blocks"], 4)
        self.assertEqual(plan["train"]["target_tokens"], 100_007_936)
        self.assertEqual(plan["train"]["block_ids"], list(range(3_052)))

    def test_partial_final_block_changes_only_decay_tail(self) -> None:
        plan = derive_plan(_manifest(final_train_sequences=7))
        trainer = plan["trainer"]
        self.assertEqual(trainer["steps"], 3_052)
        self.assertEqual(trainer["warmup_tokens"], 5_013_504)
        self.assertEqual(trainer["stable_tokens"], 74_973_184)
        self.assertEqual(trainer["decay_tokens"], 20_002_816)

    def test_drive_manifest_is_bound_into_identity(self) -> None:
        manifest = _manifest()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "drive_manifest.json"
            path.write_text(json.dumps(_drive_manifest(manifest)), encoding="utf-8")
            plan = derive_plan(manifest, drive_manifest_path=path)
        self.assertEqual(plan["identity"]["drive_run_id"], RUN_ID)
        self.assertEqual(plan["identity"]["drive_shard_count"], 4)

    def test_wrong_source_envelope_fails_closed(self) -> None:
        manifest = _manifest()
        manifest["production"]["target_source_tokens"] = 99_000_000
        with self.assertRaisesRegex(ValueError, "target_source_tokens mismatch"):
            derive_plan(manifest)

    def test_target_not_reached_fails_closed(self) -> None:
        manifest = _manifest()
        manifest["accepted_source_tokens"] = 99_999_999
        manifest["production"]["target_reached"] = False
        with self.assertRaisesRegex(ValueError, "did not reach"):
            derive_plan(manifest)


if __name__ == "__main__":
    unittest.main()
