"""Tests for teacher-forced held-out confidence diagnostics."""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from trainer.post_pretraining_prompt_suite import _parse_args
from trainer.teacher_forced_diagnostic import (
    _annotated_target_text,
    _stage_incremental_validation_dataset,
    _summary,
    resolve_validation_dataset,
    teacher_forced_token_metrics,
)


class TeacherForcedDiagnosticTests(unittest.TestCase):
    def test_cli_enables_auto_validation_dataset_resolution(self) -> None:
        args = _parse_args(
            [
                "--repo-id",
                "owner/repo",
                "--teacher-forced-validation",
            ]
        )
        self.assertEqual(args.teacher_forced_validation, "auto")

    def test_token_metrics_report_true_rank_and_raw_probabilities(self) -> None:
        logits = torch.tensor(
            [
                [3.0, 1.0, 0.0],
                [0.0, 2.0, 1.0],
            ]
        )
        labels = torch.tensor([0, 2])
        metrics = teacher_forced_token_metrics(logits, labels, top_n=3)

        first_probability = math.exp(3.0) / (
            math.exp(3.0) + math.exp(1.0) + math.exp(0.0)
        )
        second_true_probability = math.exp(1.0) / (
            math.exp(0.0) + math.exp(2.0) + math.exp(1.0)
        )
        self.assertAlmostEqual(
            float(metrics["true_probability"][0]),
            first_probability,
            places=6,
        )
        self.assertAlmostEqual(
            float(metrics["true_probability"][1]),
            second_true_probability,
            places=6,
        )
        self.assertEqual(metrics["true_rank"].tolist(), [1, 2])
        self.assertEqual(metrics["top_token_ids"][0].tolist(), [0, 1, 2])
        self.assertEqual(metrics["top_token_ids"][1].tolist(), [1, 2, 0])
        self.assertAlmostEqual(float(metrics["top5_mass"][0]), 1.0, places=6)
        self.assertGreater(float(metrics["entropy"][0]), 0.0)

    def test_summary_separates_accuracy_from_confident_wrong_predictions(self) -> None:
        records = [
            {
                "true_log_probability": math.log(0.8),
                "true_probability": 0.8,
                "top1_probability": 0.8,
                "true_rank": 1,
                "entropy": 0.5,
                "top5_mass": 0.95,
            },
            {
                "true_log_probability": math.log(0.1),
                "true_probability": 0.1,
                "top1_probability": 0.7,
                "true_rank": 3,
                "entropy": 1.0,
                "top5_mass": 0.9,
            },
        ]
        summary = _summary(records)
        self.assertEqual(summary["target_tokens"], 2)
        self.assertAlmostEqual(float(summary["top1_accuracy"]), 0.5)
        self.assertAlmostEqual(float(summary["true_rank_le_5"]), 1.0)
        self.assertAlmostEqual(float(summary["confidently_wrong_ge_0_5"]), 0.5)

    def test_readable_target_expands_bpe_piece_to_full_word(self) -> None:
        pieces = [
            "This",
            " is",
            " a",
            " huge",
            " crowd",
            " ple",
            "aser",
            ".",
        ]
        text = "".join(pieces)
        offsets: list[int] = []
        cursor = 0
        for piece in pieces:
            offsets.append(cursor)
            cursor += len(piece)

        rendered = _annotated_target_text(
            text,
            offsets,
            5,
            before_chars=100,
            after_chars=100,
        )
        self.assertEqual(rendered, "This is a huge crowd [pleaser].")

    def test_readable_target_expands_piece_that_starts_inside_word(self) -> None:
        pieces = ["The", " se", "lect", "ion", " worked", "."]
        text = "".join(pieces)
        offsets: list[int] = []
        cursor = 0
        for piece in pieces:
            offsets.append(cursor)
            cursor += len(piece)

        rendered = _annotated_target_text(
            text,
            offsets,
            2,
            before_chars=100,
            after_chars=100,
        )
        self.assertEqual(rendered, "The [selection] worked.")

    def test_explicit_dataset_must_match_checkpoint_drive_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint"
            dataset = root / "dataset"
            checkpoint.mkdir()
            (dataset / "validation").mkdir(parents=True)
            (dataset / "manifest.json").write_text("{}\n", encoding="utf-8")
            manifest_bytes = '{"run_id":"same"}\n'
            (checkpoint / "drive_manifest.json").write_text(
                manifest_bytes,
                encoding="utf-8",
            )
            (dataset / "drive_manifest.json").write_text(
                manifest_bytes,
                encoding="utf-8",
            )
            self.assertEqual(
                resolve_validation_dataset(str(dataset), checkpoint_root=checkpoint),
                dataset.resolve(),
            )

    def test_explicit_modern_dataset_matches_checkpoint_dataset_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint"
            dataset = root / "dataset"
            checkpoint.mkdir()
            (dataset / "validation").mkdir(parents=True)
            manifest_bytes = '{"schema_version":2,"production":{"run_id":"modern"}}\n'
            (dataset / "manifest.json").write_text(manifest_bytes, encoding="utf-8")
            manifest_sha = hashlib.sha256(manifest_bytes.encode("utf-8")).hexdigest()
            (checkpoint / "drive_manifest.json").write_text(
                json.dumps(
                    {
                        "version": 1,
                        "run_id": "model-run",
                        "dataset_run_id": "modern",
                        "dataset_manifest_sha256": manifest_sha,
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

            self.assertEqual(
                resolve_validation_dataset(str(dataset), checkpoint_root=checkpoint),
                dataset.resolve(),
            )

    def test_explicit_modern_dataset_rejects_manifest_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint"
            dataset = root / "dataset"
            checkpoint.mkdir()
            (dataset / "validation").mkdir(parents=True)
            (dataset / "manifest.json").write_text("{}\n", encoding="utf-8")
            (checkpoint / "drive_manifest.json").write_text(
                json.dumps(
                    {
                        "version": 1,
                        "run_id": "model-run",
                        "dataset_run_id": "modern",
                        "dataset_manifest_sha256": "a" * 64,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "dataset identity"):
                resolve_validation_dataset(str(dataset), checkpoint_root=checkpoint)

    def test_auto_discovers_modern_dataset_under_kaggle_work_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint"
            work_root = root / "small-llm"
            dataset = (
                work_root
                / "datasets"
                / "modal-10b-b64-dataset-001"
                / "from-00046500"
            )
            checkpoint.mkdir()
            (dataset / "validation").mkdir(parents=True)
            manifest_bytes = '{"schema_version":2,"production":{"run_id":"modern"}}\n'
            (dataset / "manifest.json").write_text(manifest_bytes, encoding="utf-8")
            manifest_sha = hashlib.sha256(manifest_bytes.encode("utf-8")).hexdigest()
            (checkpoint / "drive_manifest.json").write_text(
                json.dumps(
                    {
                        "version": 1,
                        "dataset_run_id": "modern",
                        "dataset_manifest_sha256": manifest_sha,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with mock.patch.dict(
                "os.environ",
                {
                    "SMALL_LLM_KAGGLE_WORK_ROOT": str(work_root),
                    "SMALL_LLM_DATASET_DIR": "",
                },
                clear=False,
            ):
                self.assertEqual(
                    resolve_validation_dataset("auto", checkpoint_root=checkpoint),
                    dataset.resolve(),
                )

    def test_modern_auto_stage_downloads_only_frozen_validation_shards(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            work_root = Path(temporary) / "small-llm"
            validation_bytes = b"held-out"
            checksum = hashlib.sha256(validation_bytes).hexdigest()
            consumer_manifest = {
                "schema_version": 2,
                "sequence_format": "context_plus_one",
                "context_length": 2048,
                "stored_tokens_per_sequence": 2049,
                "sequences_per_block": 64,
                "production": {
                    "run_id": "modal-10b-b64-dataset-001",
                    "incremental_producer_complete": False,
                },
                "shards": [
                    {
                        "filename": "validation/validation-00000.bin",
                        "split": "validation",
                        "byte_size": len(validation_bytes),
                        "checksum": checksum,
                        "first_block_id": 0,
                        "last_block_id": 0,
                        "sequence_count": 1,
                    }
                ],
            }
            manifest_bytes = (
                json.dumps(
                    consumer_manifest,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8")
            expected_sha = hashlib.sha256(manifest_bytes).hexdigest()
            contract = {
                "run_id": "modal-10b-b64-dataset-001",
                "contract_sha256": "c" * 64,
            }
            frontier = {
                "run_id": "modal-10b-b64-dataset-001",
                "contract_sha256": "c" * 64,
                "frozen_validation_shards": [
                    {
                        "filename": "validation/validation-00000.bin",
                        "split": "validation",
                        "byte_size": len(validation_bytes),
                        "checksum": checksum,
                        "first_block_id": 0,
                        "last_block_id": 0,
                        "sequence_count": 1,
                    }
                ],
            }

            class FakeStore:
                def __init__(
                    self,
                    bucket_id: str,
                    *,
                    token=None,
                    private=True,
                    create_bucket=False,
                ) -> None:
                    del token, private, create_bucket
                    self.bucket_id = bucket_id

                @staticmethod
                def object_key(run_id: str, logical_name: str) -> str:
                    return f"run/{run_id}/{logical_name}"

                def download_shard(
                    self,
                    *,
                    run_id: str,
                    logical_name: str,
                    file_id: str,
                    destination: Path,
                    byte_size: int,
                    sha256: str,
                ) -> None:
                    del run_id, logical_name, file_id, byte_size, sha256
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(validation_bytes)

            with (
                mock.patch.dict(
                    "os.environ",
                    {
                        "SMALL_LLM_KAGGLE_WORK_ROOT": str(work_root),
                        "SMALL_LLM_HF_REPO_ID": "owner/model",
                        "SMALL_LLM_HF_DATASET_BUCKET_ID": "",
                    },
                    clear=False,
                ),
                mock.patch(
                    "dataset.src.hf_bucket_shards.HuggingFaceBucketShardStore",
                    FakeStore,
                ),
                mock.patch(
                    "dataset.incremental_frontier.read_run_contract",
                    return_value=contract,
                ),
                mock.patch(
                    "dataset.incremental_frontier.read_frontier",
                    return_value=frontier,
                ),
                mock.patch(
                    "dataset.incremental_stage._stable_consumer_manifest",
                    return_value=consumer_manifest,
                ),
            ):
                staged = _stage_incremental_validation_dataset(
                    checkpoint_metadata={
                        "dataset_run_id": "modal-10b-b64-dataset-001",
                    },
                    expected_manifest_sha256=expected_sha,
                )

            self.assertIsNotNone(staged)
            assert staged is not None
            self.assertEqual(
                hashlib.sha256((staged / "manifest.json").read_bytes()).hexdigest(),
                expected_sha,
            )
            self.assertEqual(
                (staged / "validation" / "validation-00000.bin").read_bytes(),
                validation_bytes,
            )


if __name__ == "__main__":
    unittest.main()
