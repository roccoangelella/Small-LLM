"""Tests for teacher-forced held-out confidence diagnostics."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import torch

from trainer.post_pretraining_prompt_suite import _parse_args
from trainer.teacher_forced_diagnostic import (
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


if __name__ == "__main__":
    unittest.main()
