from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from dataset import config, eval_core
from dataset.eval_core_accelerated import (
    _SourceBatch,
    _ValidationCandidate,
    _ordered_validation_batches,
    build_eval_core_accelerated,
)
from dataset.src.bytesource import SourceFile
from dataset.src.records import ParsedRecord


class AcceleratedEvalCoreTests(unittest.TestCase):
    def test_parallel_batches_are_yielded_in_frozen_workplan_order(self) -> None:
        source = SourceFile(
            path="part_0.tokenized.jsonl",
            size=config.REGION_BYTES * 4,
        )

        def fake_scan_item(*, ordinal, item, source, validation_probability):
            # Deliberately make earlier ordinals slower so completion order differs
            # from the required deterministic consumption order.
            time.sleep((3 - ordinal) * 0.005)
            return _SourceBatch(
                ordinal=ordinal,
                filename=item.filename,
                scanned_records=ordinal + 1,
                candidates=(),
            )

        with (
            patch.dict(os.environ, {"SMALL_LLM_EVAL_SCAN_WORKERS": "4"}),
            patch(
                "dataset.eval_core_accelerated._scan_item",
                side_effect=fake_scan_item,
            ),
        ):
            batches = list(
                _ordered_validation_batches(
                    [source],
                    validation_probability=1.0,
                    max_work_items=4,
                )
            )
        self.assertEqual([batch.ordinal for batch in batches], [0, 1, 2, 3])

    def test_accelerated_builder_matches_legacy_manifest_and_bytes(self) -> None:
        filename = "part_0.tokenized.jsonl"
        stream: list[tuple[str, ParsedRecord]] = []
        for index, cluster in enumerate(eval_core.ACCEPTED_CLUSTERS):
            raw = json.dumps(
                {"cluster_id": cluster, "tokens": [index + 1, index + 2]}
            ).encode("utf-8")
            stream.append(
                (
                    filename,
                    ParsedRecord(record_start=index * 100, raw=raw),
                )
            )

        # The legacy builder checks completion on the next valid validation record
        # after the final quota-filling document. Include that record so the exact
        # scanned/validation counters are part of the equivalence assertion.
        extra_raw = json.dumps(
            {"cluster_id": eval_core.ACCEPTED_CLUSTERS[0], "tokens": [7, 8]}
        ).encode("utf-8")
        stream.append(
            (
                filename,
                ParsedRecord(record_start=len(stream) * 100, raw=extra_raw),
            )
        )

        candidates = tuple(
            _ValidationCandidate(record=record, scanned_through=index + 1)
            for index, (_, record) in enumerate(stream)
        )
        batch = _SourceBatch(
            ordinal=0,
            filename=filename,
            scanned_records=len(stream),
            candidates=candidates,
        )
        source = SourceFile(path=filename, size=1)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            legacy_dir = root / "legacy"
            accelerated_dir = root / "accelerated"

            legacy_manifest = eval_core.build_eval_core(
                legacy_dir,
                record_stream=stream,
                validation_probability=1.0,
                fast_documents_per_cluster=1,
                fast_targets_per_cluster=1,
                full_documents_per_cluster=1,
                full_targets_per_cluster=1,
            )

            with (
                patch(
                    "dataset.eval_core_accelerated.list_source_files",
                    return_value=[source],
                ),
                patch(
                    "dataset.eval_core_accelerated._ordered_validation_batches",
                    return_value=iter((batch,)),
                ),
            ):
                accelerated_manifest = build_eval_core_accelerated(
                    accelerated_dir,
                    validation_probability=1.0,
                    fast_documents_per_cluster=1,
                    fast_targets_per_cluster=1,
                    full_documents_per_cluster=1,
                    full_targets_per_cluster=1,
                )

            self.assertEqual(accelerated_manifest, legacy_manifest)
            for name in (
                "manifest.json",
                "fast.bin",
                "fast.records.jsonl",
                "full.bin",
                "full.records.jsonl",
            ):
                self.assertEqual(
                    (accelerated_dir / name).read_bytes(),
                    (legacy_dir / name).read_bytes(),
                    name,
                )


if __name__ == "__main__":
    unittest.main()
