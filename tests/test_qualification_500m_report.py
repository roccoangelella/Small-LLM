"""Tests for the fixed 20M-model/500M-token qualification profile."""

from __future__ import annotations

import unittest

from dataset.qualification_500m import RUN_ID, qualification_arguments
from dataset.qualification_500m_report import derive_plan


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
        "accepted_source_tokens": 500_000_100,
        "validation_source_tokens": 500_000,
        "production": {
            "run_id": RUN_ID,
            "configuration_hash": "b" * 64,
            "schema_hash": "c" * 64,
            "target_source_tokens": 500_000_000,
            "minimum_source_tokens": 450_000_000,
            "maximum_source_tokens": 550_000_000,
            "checkpoint_source_tokens": 20_000_000,
            "target_reached": True,
            "remote_required": True,
        },
        "shards": [
            _shard("train", 0, 15_249, 15_250 * 16),
            _shard("validation", 0, 3, 64),
        ],
    }


class Qualification500MTests(unittest.TestCase):
    def test_entry_point_locks_exact_500m_source_envelope(self) -> None:
        arguments = qualification_arguments(
            ["--weights-file", "weights.json", "--output-dir", "out"]
        )
        self.assertEqual(
            arguments[-16:],
            [
                "--run-id", RUN_ID,
                "--target-tokens", "500000000",
                "--minimum-tokens", "450000000",
                "--maximum-tokens", "550000000",
                "--checkpoint-source-tokens", "20000000",
                "--context-length", "2048",
                "--sequences-per-block", "16",
                "--target-shard-bytes", "8388608",
            ],
        )

    def test_entry_point_rejects_identity_overrides(self) -> None:
        for flag in ("--run-id", "--target-tokens", "--checkpoint-source-tokens"):
            with self.subTest(flag=flag), self.assertRaises(SystemExit):
                qualification_arguments([flag, "wrong"])

    def test_full_profile_derives_500m_wsd_schedule(self) -> None:
        plan = derive_plan(_manifest())
        trainer = plan["trainer"]
        self.assertEqual(plan["qualification_profile"], "20m-500m-data-scaling-v1")
        self.assertEqual(trainer["steps"], 15_250)
        self.assertEqual(trainer["warmup_updates"], 763)
        self.assertEqual(trainer["stable_updates"], 11_437)
        self.assertEqual(trainer["decay_updates"], 3_050)
        self.assertEqual(trainer["warmup_tokens"], 25_001_984)
        self.assertEqual(trainer["stable_tokens"], 374_767_616)
        self.assertEqual(trainer["decay_tokens"], 99_942_400)
        self.assertEqual(trainer["validation_blocks"], 4)
        self.assertEqual(plan["train"]["target_tokens"], 499_712_000)

    def test_wrong_100m_identity_fails_closed(self) -> None:
        manifest = _manifest()
        manifest["production"]["run_id"] = "20m-100m-dataset-001"
        manifest["production"]["target_source_tokens"] = 100_000_000
        with self.assertRaisesRegex(ValueError, "target_source_tokens mismatch"):
            derive_plan(manifest)


if __name__ == "__main__":
    unittest.main()
