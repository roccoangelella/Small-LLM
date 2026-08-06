from __future__ import annotations

from array import array
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from torch import nn

from dataset import config
from dataset.eval_core import (
    ACCEPTED_CLUSTERS,
    CONTEXT_LENGTH,
    EVAL_NAME,
    SCHEMA_VERSION,
    STORED_TOKENS,
    canonical_json_bytes,
    sha256_file,
)
from trainer.eval_suite import evaluate_split


class _PerfectNextTokenModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        logits = torch.full(
            (*input_ids.shape, 4),
            -5.0,
            device=input_ids.device,
        )
        expected = (input_ids + 1).remainder(4)
        return logits.scatter(-1, expected.unsqueeze(-1), 5.0)


def _write_tiny_eval(root: Path) -> None:
    rows = []
    payload = array("H")
    for sequence_index, cluster in enumerate(ACCEPTED_CLUSTERS):
        sequence = [0, 1, 2, 3] + [3] * (STORED_TOKENS - 4)
        payload.extend(sequence)
        rows.append(
            {
                "sequence_index": sequence_index,
                "document_id": f"doc-{cluster}",
                "cluster_id": cluster,
                "filename": "synthetic.jsonl",
                "record_start": sequence_index,
                "valid_targets": 3,
            }
        )
    if sys.byteorder != "little":
        payload.byteswap()
    for suite in ("fast", "full"):
        (root / f"{suite}.bin").write_bytes(payload.tobytes())
        (root / f"{suite}.records.jsonl").write_text(
            "".join(
                json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )
    suite_rows = {}
    for suite in ("fast", "full"):
        suite_rows[suite] = {
            "data_file": f"{suite}.bin",
            "records_file": f"{suite}.records.jsonl",
            "data_sha256": sha256_file(root / f"{suite}.bin"),
            "records_sha256": sha256_file(root / f"{suite}.records.jsonl"),
            "sequence_count": len(rows),
            "document_count": len(rows),
            "target_token_count": len(rows) * 3,
            "per_cluster": {},
        }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "name": EVAL_NAME,
        "dataset": config.DATASET_REPOSITORY,
        "revision": config.DATASET_REVISION,
        "tokenizer": "synthetic",
        "semantic_vocab_size": 4,
        "eod_token_id": 3,
        "context_length": CONTEXT_LENGTH,
        "stored_tokens_per_sequence": STORED_TOKENS,
        "dtype": "uint16-le",
        "split": {
            "seed": config.SELECTION_SEED,
            "hash_version": config.SPLIT_HASH_VERSION,
            "validation_probability": 1.0,
        },
        "accepted_clusters": list(ACCEPTED_CLUSTERS),
        "mixture_source_tokens": {},
        "mixture_weights_sha256": "test",
        "minimums": {},
        "suites": suite_rows,
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(manifest)
    ).hexdigest()
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


class EvalSuiteTests(unittest.TestCase):
    def test_streaming_metrics_cover_all_clusters(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_tiny_eval(root)
            metrics = evaluate_split(
                _PerfectNextTokenModel(),
                eval_dir=root,
                suite="fast",
                precision="fp32",
                batch_size=4,
                bootstrap_samples=10,
                token_byte_lengths=[1, 1, 1, 1],
                enforce_frozen_minimums=False,
            )
            self.assertEqual(
                metrics["target_tokens"], 3 * len(ACCEPTED_CLUSTERS)
            )
            self.assertEqual(metrics["documents"], len(ACCEPTED_CLUSTERS))
            self.assertAlmostEqual(metrics["top_k_accuracy"]["1"], 1.0)
            self.assertEqual(
                set(metrics["per_cluster"]),
                {str(cluster) for cluster in ACCEPTED_CLUSTERS},
            )
            self.assertTrue(math.isfinite(metrics["loss"]))
            self.assertGreater(metrics["bits_per_byte"], 0.0)
            self.assertEqual(metrics["bootstrap_95"]["samples"], 10)


if __name__ == "__main__":
    unittest.main()
