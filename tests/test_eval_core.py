from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from dataset.eval_core import (
    ACCEPTED_CLUSTERS,
    build_eval_core,
    document_windows,
    verify_eval_core,
)
from dataset.src.records import ParsedRecord


class EvalCoreTests(unittest.TestCase):
    def test_document_windows_append_eod_and_mask_padding(self) -> None:
        rows = document_windows([10, 11, 12], context_length=4, eod_token_id=99)
        self.assertEqual(len(rows), 1)
        sequence, valid_targets = rows[0]
        self.assertEqual(sequence, (10, 11, 12, 99, 99))
        self.assertEqual(valid_targets, 3)

    def test_tiny_injected_build_is_nested_and_verifiable(self) -> None:
        stream: list[tuple[str, ParsedRecord]] = []
        for index, cluster in enumerate(ACCEPTED_CLUSTERS):
            raw = json.dumps(
                {"cluster_id": cluster, "tokens": [index + 1, index + 2]}
            ).encode("utf-8")
            stream.append(
                (
                    "part_0.tokenized.jsonl",
                    ParsedRecord(record_start=index * 100, raw=raw),
                )
            )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "eval_core_v1"
            manifest = build_eval_core(
                output,
                record_stream=stream,
                validation_probability=1.0,
                fast_documents_per_cluster=1,
                fast_targets_per_cluster=1,
                full_documents_per_cluster=1,
                full_targets_per_cluster=1,
            )
            verified = verify_eval_core(output, enforce_frozen_minimums=False)
            self.assertEqual(
                verified["manifest_sha256"], manifest["manifest_sha256"]
            )
            suites = verified["suites"]
            self.assertEqual(
                suites["fast"]["document_count"], len(ACCEPTED_CLUSTERS)
            )
            self.assertEqual(
                suites["full"]["document_count"], len(ACCEPTED_CLUSTERS)
            )
            self.assertEqual(
                suites["fast"]["target_token_count"],
                suites["full"]["target_token_count"],
            )

    def test_verifier_rejects_changed_binary(self) -> None:
        stream: list[tuple[str, ParsedRecord]] = []
        for index, cluster in enumerate(ACCEPTED_CLUSTERS):
            raw = json.dumps(
                {"cluster_id": cluster, "tokens": [1, 2]}
            ).encode("utf-8")
            stream.append(
                (
                    "part_0.tokenized.jsonl",
                    ParsedRecord(record_start=index, raw=raw),
                )
            )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "eval_core_v1"
            build_eval_core(
                output,
                record_stream=stream,
                validation_probability=1.0,
                fast_documents_per_cluster=1,
                fast_targets_per_cluster=1,
                full_documents_per_cluster=1,
                full_targets_per_cluster=1,
            )
            with (output / "fast.bin").open("r+b") as handle:
                first = handle.read(1)
                handle.seek(0)
                handle.write(bytes([first[0] ^ 0xFF]))
            with self.assertRaisesRegex(ValueError, "data hash mismatch"):
                verify_eval_core(output, enforce_frozen_minimums=False)


if __name__ == "__main__":
    unittest.main()
