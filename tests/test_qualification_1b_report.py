"""Tests for the fixed 20M-model/1B-token qualification profile."""

from __future__ import annotations

import unittest

from dataset.qualification_1b import RUN_ID, qualification_arguments
from dataset.qualification_1b_report import derive_plan


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
        "accepted_source_tokens": 1_000_000_100,
        "validation_source_tokens": 1_000_000,
        "production": {
            "run_id": RUN_ID,
            "configuration_hash": "b" * 64,
            "schema_hash": "c" * 64,
            "target_source_tokens": 1_000_000_000,
            "minimum_source_tokens": 900_000_000,
            "maximum_source_tokens": 1_100_000_000,
            "checkpoint_source_tokens": 40_000_000,
            "target_reached": True,
            "remote_required": True,
        },
        "shards": [
            _shard("train", 0, 30_499, 30_500 * 16),
            _shard("validation", 0, 7, 128),
        ],
    }


class Qualification1BTests(unittest.TestCase):
    def test_entry_point_locks_exact_1b_source_envelope(self) -> None:
        arguments = qualification_arguments(
            ["--weights-file", "weights.json", "--output-dir", "out"]
        )
        self.assertEqual(
            arguments[-16:],
            [
                "--run-id", RUN_ID,
                "--target-tokens", "1000000000",
                "--minimum-tokens", "900000000",
                "--maximum-tokens", "1100000000",
                "--checkpoint-source-tokens", "40000000",
                "--context-length", "2048",
                "--sequences-per-block", "16",
                "--target-shard-bytes", "8388608",
            ],
        )

    def test_entry_point_rejects_identity_overrides(self) -> None:
        for flag in ("--run-id", "--target-tokens", "--checkpoint-source-tokens"):
            with self.subTest(flag=flag), self.assertRaises(SystemExit):
                qualification_arguments([flag, "wrong"])

    def test_full_profile_derives_1b_wsd_schedule(self) -> None:
        plan = derive_plan(_manifest())
        trainer = plan["trainer"]
        self.assertEqual(plan["qualification_profile"], "20m-1b-data-scaling-v1")
        self.assertEqual(trainer["steps"], 30_500)
        self.assertEqual(trainer["warmup_updates"], 1_525)
        self.assertEqual(trainer["stable_updates"], 22_875)
        self.assertEqual(trainer["decay_updates"], 6_100)
        self.assertEqual(trainer["warmup_tokens"], 49_971_200)
        self.assertEqual(trainer["stable_tokens"], 749_568_000)
        self.assertEqual(trainer["decay_tokens"], 199_884_800)
        self.assertEqual(trainer["validation_blocks"], 8)
        self.assertEqual(plan["train"]["target_tokens"], 999_424_000)

    def test_wrong_500m_identity_fails_closed(self) -> None:
        manifest = _manifest()
        manifest["production"]["run_id"] = "20m-500m-dataset-001"
        manifest["production"]["target_source_tokens"] = 500_000_000
        with self.assertRaisesRegex(ValueError, "target_source_tokens mismatch"):
            derive_plan(manifest)


if __name__ == "__main__":
    unittest.main()
