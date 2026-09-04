"""Tests for the qualitative post-pretraining prompt runner."""

from __future__ import annotations

from dataclasses import asdict
import math
import unittest

import torch
from torch import nn

from model.config import ModelConfig
from trainer.post_pretraining_prompt_suite import (
    PROMPT_CASES,
    _checkpoint_prefix,
    _generation_budget,
    _normalize_model_config,
    _parse_args,
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
        alias_pointer = dict(
            pointer,
            best_prefix="run/pretrain/checkpoints/step-00000100/last",
            checkpoint_manifest={"version": 1, "files": []},
        )
        self.assertEqual(
            _checkpoint_prefix(alias_pointer, run_id="pretrain", pointer_name="best"),
            (
                "step-00000100",
                "run/pretrain/checkpoints/step-00000100/last",
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

    def test_global_generation_budget_caps_native_prompt_budget(self) -> None:
        story = PROMPT_CASES[0]
        sentiment = PROMPT_CASES[5]
        self.assertEqual(_generation_budget(story, None), 128)
        self.assertEqual(_generation_budget(story, 32), 32)
        self.assertEqual(_generation_budget(sentiment, 64), 48)


    def test_default_sampled_prompt_protocol_matches_adr_0136(self) -> None:
        args = _parse_args([
            "--repo-id",
            "owner/repo",
            "--output-json",
            "out.json",
        ])
        self.assertEqual(args.temperature, 1.0)
        self.assertEqual(args.top_p, 1.0)
        self.assertEqual(args.top_k, 0)
        self.assertEqual(args.seed, 17)
        self.assertEqual(args.samples_per_prompt, 1)
        self.assertIsNone(args.max_new_tokens)

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

    def test_token_trace_uses_raw_model_probabilities(self) -> None:
        model = _ToyModel()
        trace: list[dict[str, object]] = []
        generated = sample_token_ids(
            model,
            [0, 1],
            max_new_tokens=2,
            max_seq_len=8,
            eos_token_id=3,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            seed=17,
            precision="fp32",
            trace_top_tokens=2,
            trace_out=trace,
        )
        self.assertEqual(generated, [2, 2])
        self.assertEqual(len(trace), 2)
        self.assertEqual(trace[0]["chosen_token_id"], 2)
        expected_probability = math.exp(5.0) / (math.exp(5.0) + 3.0)
        self.assertAlmostEqual(
            float(trace[0]["chosen_probability"]),
            expected_probability,
            places=6,
        )
        top_tokens = trace[0]["top_tokens"]
        self.assertIsInstance(top_tokens, list)
        assert isinstance(top_tokens, list)
        self.assertEqual(top_tokens[0]["token_id"], 2)
        self.assertAlmostEqual(
            float(top_tokens[0]["probability"]),
            expected_probability,
            places=6,
        )


if __name__ == "__main__":
    unittest.main()