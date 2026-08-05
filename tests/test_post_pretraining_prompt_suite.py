"""Tests for the qualitative post-pretraining prompt runner."""

from __future__ import annotations

from dataclasses import asdict
import unittest

import torch
from torch import nn

from model.config import ModelConfig
from trainer.post_pretraining_prompt_suite import (
    PROMPT_CASES,
    _checkpoint_prefix,
    _normalize_model_config,
    _selected_cases,
    sample_token_ids,
)


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch, length = input_ids.shape
        logits = torch.zeros(batch, length, 4, device=input_ids.device)
        logits[..., 2] = 5.0
        return logits


class PostPretrainingPromptSuiteTests(unittest.TestCase):
    def test_suite_contains_many_english_general_knowledge_questions(self) -> None:
        questions = [case for case in PROMPT_CASES if case.category == "question"]
        self.assertGreaterEqual(len(questions), 10)
        self.assertFalse(any("translate" in case.prompt.lower() for case in PROMPT_CASES))
        self.assertEqual(
            _selected_cases(questions_only=True, max_cases=None),
            tuple(questions),
        )

    def test_best_pointer_prefix_is_bound_to_run_and_checkpoint(self) -> None:
        pointer = {
            "checkpoint_id": "step-00000100",
            "best_prefix": "run/pretrain/checkpoints/step-00000100/best",
            "metric": -2.0,
        }
        self.assertEqual(
            _checkpoint_prefix(pointer, run_id="pretrain", pointer_name="best"),
            (
                "step-00000100",
                "run/pretrain/checkpoints/step-00000100/best",
            ),
        )
        broken = dict(
            pointer,
            best_prefix="run/other/checkpoints/step-00000100/best",
        )
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            _checkpoint_prefix(broken, run_id="pretrain", pointer_name="best")

    def test_model_config_round_trip_accepts_json_layer_pattern(self) -> None:
        raw = asdict(ModelConfig.smoke())
        raw["layer_pattern"] = list(raw["layer_pattern"])
        restored = _normalize_model_config(raw)
        self.assertEqual(restored, ModelConfig.smoke())

    def test_greedy_generation_is_deterministic(self) -> None:
        model = _ToyModel()
        generated = sample_token_ids(
            model,
            [0, 1],
            max_new_tokens=3,
            max_seq_len=8,
            eos_token_id=3,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            seed=17,
            precision="fp32",
        )
        self.assertEqual(generated, [2, 2, 2])


if __name__ == "__main__":
    unittest.main()
