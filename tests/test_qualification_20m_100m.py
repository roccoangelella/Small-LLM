"""Tests for the fixed 20M-model / 100M-token dataset profile."""

from __future__ import annotations

import unittest

from dataset.qualification_20m_100m import qualification_arguments
from dataset.qualification_20m_100m_report import derive_plan


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


def _manifest() -> dict[str, object]:
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
            "run_id": "20m-100m-dataset-001",
            "configuration_hash": "b" * 64,
            "schema_hash": "c" * 64,
            "target_source_tokens": 100_000_000,
            "minimum_source_tokens": 90_000_000,
            "maximum_source_tokens": 110_000_000,
            "checkpoint_source_tokens": 20_000_000,
            "target_reached": True,
            "remote_required": True,
        },
        "shards": [
            _shard("train", 0, 999, 1000 * 16),
            _shard("train", 1000, 1999, 1000 * 16),
            _shard("train", 2000, 3051, 1052 * 16),
            _shard("validation", 0, 0, 4),
        ],
    }


class Qualification20M100MTests(unittest.TestCase):
    def test_entry_point_appends_exact_profile(self) -> None:
        args = qualification_arguments(
            ["--weights-file", "weights.json", "--output-dir", "out"]
        )
        joined = " ".join(args)
        self.assertIn("--target-tokens 100000000", joined)
        self.assertIn("--minimum-tokens 90000000", joined)
        self.assertIn("--maximum-tokens 110000000", joined)
        self.assertIn("--checkpoint-source-tokens 20000000", joined)
        self.assertIn("--sequences-per-block 16", joined)
        self.assertIn("--target-shard-bytes 8388608", joined)

    def test_entry_point_rejects_scientific_overrides(self) -> None:
        with self.assertRaises(SystemExit):
            qualification_arguments(["--target-tokens", "12"])
        with self.assertRaises(SystemExit):
            qualification_arguments(["--allow-local-only"])

    def test_exact_schedule_and_operational_plan(self) -> None:
        plan = derive_plan(_manifest())
        trainer = plan["trainer"]
        self.assertEqual(plan["qualification_profile"], "20m-100m-one-pass-v1")
        self.assertEqual(trainer["steps"], 3052)
        self.assertEqual(trainer["warmup_updates"], 153)
        self.assertEqual(trainer["stable_updates"], 2288)
        self.assertEqual(trainer["decay_updates"], 611)
        self.assertEqual(trainer["warmup_tokens"], 5_013_504)
        self.assertEqual(trainer["stable_tokens"], 74_973_184)
        self.assertEqual(trainer["decay_tokens"], 20_021_248)
        self.assertEqual(trainer["checkpoint_every_steps"], 250)
        self.assertEqual(trainer["evaluation_every_steps"], 500)
        self.assertEqual(trainer["remote_publish_every_steps"], 500)
        self.assertEqual(trainer["microbatch_size"], 4)
        self.assertEqual(plan["train"]["target_tokens"], 100_007_936)
        self.assertEqual(plan["train"]["block_ids"], list(range(3052)))

    def test_10m_profile_is_rejected(self) -> None:
        manifest = _manifest()
        manifest["production"]["target_source_tokens"] = 10_000_000
        with self.assertRaisesRegex(ValueError, "target_source_tokens mismatch"):
            derive_plan(manifest)

    def test_wrong_optimizer_block_geometry_is_rejected(self) -> None:
        manifest = _manifest()
        manifest["sequences_per_block"] = 32
        with self.assertRaisesRegex(ValueError, "sequences_per_block mismatch"):
            derive_plan(manifest)


if __name__ == "__main__":
    unittest.main()
