"""Regression tests for memory-bounded held-out evaluation."""

from __future__ import annotations

import unittest

import torch

from trainer import TrainerConfig, TrainerEngine
from tests.trainer_fixtures import TinyLM, batch


class RecordingTinyLM(TinyLM):
    def __init__(self) -> None:
        super().__init__()
        self.forward_batch_sizes: list[int] = []

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.forward_batch_sizes.append(int(input_ids.shape[0]))
        return super().forward(input_ids)


class TrainerEvaluationTests(unittest.TestCase):
    def test_validation_defaults_to_one_sequence_microbatches(self) -> None:
        model = RecordingTinyLM()
        engine = TrainerEngine(
            model,
            TrainerConfig(precision="fp32", microbatch_size=2, weight_decay=0.0),
            device="cpu",
        )
        self.assertTrue(model.training)

        result = engine.evaluate([batch(0, split="validation")])

        self.assertEqual(model.forward_batch_sizes, [1, 1])
        self.assertEqual(result["target_tokens"], 6)
        self.assertEqual(result["blocks"], 1)
        self.assertGreater(result["loss"], 0.0)
        self.assertTrue(model.training)

    def test_invalid_validation_microbatch_is_rejected(self) -> None:
        engine = TrainerEngine(
            TinyLM(),
            TrainerConfig(precision="fp32", weight_decay=0.0),
            device="cpu",
        )
        with self.assertRaisesRegex(ValueError, "positive integer"):
            from trainer.evaluation import evaluate_batches

            evaluate_batches(
                engine,
                [batch(0, split="validation")],
                microbatch_size=0,
            )


if __name__ == "__main__":
    unittest.main()
