"""Tests for the fixed 20M qualification token-by-token verifier."""

from __future__ import annotations

import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dataset import qualification_20m_verify as qualification_verify


class Qualification20MVerifyTest(unittest.TestCase):
    def _write_dataset(
        self,
        root: Path,
        *,
        train_tokens: tuple[int, ...] = (1, 2, 3),
        train_sequence_count: int = 1,
    ) -> None:
        train_path = root / "train" / "train-000000.bin"
        validation_path = root / "validation" / "validation-000000.bin"
        train_path.parent.mkdir(parents=True)
        validation_path.parent.mkdir(parents=True)
        train_payload = b"".join(struct.pack("<H", token) for token in train_tokens)
        validation_payload = b"".join(struct.pack("<H", token) for token in (4, 5, 6))
        train_path.write_bytes(train_payload)
        validation_path.write_bytes(validation_payload)

        train_last_block = max(0, train_sequence_count - 1)
        manifest = {
            "schema_version": 2,
            "sequence_format": "context_plus_one",
            "context_length": 2,
            "stored_tokens_per_sequence": 3,
            "sequences_per_block": 1,
            "target_shard_bytes": 8,
            "accepted_source_tokens": 3,
            "production": {
                "target_source_tokens": 3,
                "minimum_source_tokens": 3,
                "maximum_source_tokens": 4,
                "remote_required": True,
                "target_reached": True,
                "completion_reason": "target_reached",
            },
            "scheduler": {"total_emitted_source_tokens": 2},
            "shards": [
                {
                    "filename": "train/train-000000.bin",
                    "split": "train",
                    "byte_size": len(train_payload),
                    "token_count": len(train_tokens),
                    "sequence_count": train_sequence_count,
                    "first_block_id": 0,
                    "last_block_id": train_last_block,
                    "checksum": hashlib.sha256(train_payload).hexdigest(),
                    "shard_cluster_source_tokens": {"1": 2},
                },
                {
                    "filename": "validation/validation-000000.bin",
                    "split": "validation",
                    "byte_size": len(validation_payload),
                    "token_count": 3,
                    "sequence_count": 1,
                    "first_block_id": 0,
                    "last_block_id": 0,
                    "checksum": hashlib.sha256(validation_payload).hexdigest(),
                    "shard_cluster_source_tokens": {"2": 1},
                },
            ],
        }
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def _verify(self, root: Path) -> dict[str, object]:
        with mock.patch.multiple(
            qualification_verify,
            TARGET_SOURCE_TOKENS=3,
            MINIMUM_SOURCE_TOKENS=3,
            MAXIMUM_SOURCE_TOKENS=4,
            CONTEXT_LENGTH=2,
            SEQUENCES_PER_BLOCK=1,
            TARGET_SHARD_BYTES=8,
        ):
            return qualification_verify.verify_qualification_dataset(root)

    def test_valid_dataset_is_fully_scanned_and_reports_cluster_totals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_dataset(root)
            report = self._verify(root)

        self.assertTrue(report["passed"], report["problems"])
        self.assertTrue(report["full_scan"])
        self.assertEqual(report["scanned_tokens"], 6)
        self.assertEqual(report["train_stored_tokens"], 3)
        self.assertEqual(report["validation_stored_tokens"], 3)
        self.assertEqual(report["per_cluster_source_tokens"], {"1": 2, "2": 1})
        self.assertEqual(
            report["unavailable_counters"],
            ["accepted_document_count", "inserted_eod_count"],
        )

    def test_out_of_range_token_fails_even_when_checksum_matches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_dataset(root, train_tokens=(1, 2, 65535))
            report = self._verify(root)

        self.assertFalse(report["passed"])
        self.assertTrue(
            any("outside the vocabulary range" in problem for problem in report["problems"])
        )

    def test_sequence_geometry_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_dataset(root, train_sequence_count=2)
            report = self._verify(root)

        self.assertFalse(report["passed"])
        self.assertTrue(
            any("expected 6 from sequence geometry" in problem for problem in report["problems"])
        )


if __name__ == "__main__":
    unittest.main()
