"""Tests for the fixed 20M-model/2B-token qualification profile."""

from __future__ import annotations

import unittest

from dataset.qualification_2b import RUN_ID, qualification_arguments
from dataset.qualification_2b_report import derive_plan


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
        "accepted_source_tokens": 2_000_000_100,
        "validation_source_tokens": 2_000_000,
        "production": {
            "run_id": RUN_ID,
            "configuration_hash": "b" * 64,
            "schema_hash": "c" * 64,
            "target_source_tokens": 2_000_000_000,
            "minimum_source_tokens": 1_800_000_000,
            "maximum_source_tokens": 2_200_000_000,
            "checkpoint_source_tokens": 80_000_000,
            "target_reached": True,
            "remote_required": True,
        },
        "shards": [
            _shard("train", 0, 61_034, 61_035 * 16),
            _shard("validation", 0, 7, 128),
        ],
    }


class Qualification2BTests(unittest.TestCase):
    def test_entry_point_locks_exact_2b_source_envelope(self) -> None:
        arguments = qualification_arguments(
            ["--weights-file", "weights.json", "--output-dir", "out"]
        )
        self.assertEqual(
            arguments[-16:],
            [
                "--run-id", RUN_ID,
                "--target-tokens", "2000000000",
                "--minimum-tokens", "1800000000",
                "--maximum-tokens", "2200000000",
                "--checkpoint-source-tokens", "80000000",
                "--context-length", "2048",
                "--sequences-per-block", "16",
                "--target-shard-bytes", "8388608",
            ],
        )

    def test_entry_point_rejects_identity_overrides(self) -> None:
        for flag in ("--run-id", "--target-tokens", "--checkpoint-source-tokens"):
            with self.subTest(flag=flag), self.assertRaises(SystemExit):
                qualification_arguments([flag, "wrong"])

    def test_full_profile_derives_2b_wsd_schedule(self) -> None:
        plan = derive_plan(_manifest())
        trainer = plan["trainer"]
        self.assertEqual(plan["qualification_profile"], "20m-2b-data-scaling-v1")
        self.assertEqual(trainer["steps"], 61_035)
        self.assertEqual(trainer["warmup_updates"], 3_052)
        self.assertEqual(trainer["stable_updates"], 45_776)
        self.assertEqual(trainer["decay_updates"], 12_207)
        self.assertEqual(trainer["warmup_tokens"], 100_007_936)
        self.assertEqual(trainer["stable_tokens"], 1_499_987_968)
        self.assertEqual(trainer["decay_tokens"], 399_998_976)
        self.assertEqual(trainer["validation_blocks"], 8)
        self.assertEqual(plan["train"]["target_tokens"], 1_999_994_880)

    def test_wrong_1b_identity_fails_closed(self) -> None:
        manifest = _manifest()
        manifest["production"]["run_id"] = "20m-1b-dataset-001"
        manifest["production"]["target_source_tokens"] = 1_000_000_000
        with self.assertRaisesRegex(ValueError, "target_source_tokens mismatch"):
            derive_plan(manifest)


if __name__ == "__main__":
    unittest.main()
