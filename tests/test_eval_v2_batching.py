from __future__ import annotations

import unittest

import torch
from torch import nn

from trainer.eval_generation import GenerationRequest, sample_token_ids_batched
from trainer.post_pretraining_prompt_suite import sample_token_ids
from trainer.pretraining_eval_v2 import SmallLLMHarnessLM


class TinyNextTokenLM(nn.Module):
    def __init__(self, vocab_size: int = 32) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch, time = input_ids.shape
        logits = torch.full(
            (batch, time, self.vocab_size),
            -2.0,
            dtype=torch.float32,
            device=input_ids.device,
        )
        preferred = ((input_ids + 1) % self.vocab_size).unsqueeze(-1)
        logits.scatter_(-1, preferred, 3.0)
        return logits + self.anchor * 0.0


class BatchedGenerationTests(unittest.TestCase):
    def _legacy(self, model: nn.Module, requests: list[GenerationRequest], *, temperature: float) -> list[list[int]]:
        return [
            sample_token_ids(
                model,
                request.prompt_ids,
                max_new_tokens=request.max_new_tokens,
                max_seq_len=16,
                eos_token_id=31,
                temperature=temperature,
                top_p=1.0,
                top_k=0,
                seed=request.seed,
                precision="fp32",
            )
            for request in requests
        ]

    def test_greedy_batch_matches_legacy_and_restores_request_order(self) -> None:
        model = TinyNextTokenLM().eval()
        requests = [
            GenerationRequest((1, 2, 3, 4), 5, 17),
            GenerationRequest((7,), 3, 18),
            GenerationRequest((5, 6), 4, 19),
        ]
        expected = self._legacy(model, requests, temperature=0.0)
        actual = sample_token_ids_batched(
            model,
            requests,
            max_seq_len=16,
            eos_token_id=31,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            precision="fp32",
            batch_size=2,
        )
        self.assertEqual(actual, expected)

    def test_sampled_batch_preserves_per_request_rng_contract(self) -> None:
        model = TinyNextTokenLM().eval()
        requests = [
            GenerationRequest((1, 2), 6, 17),
            GenerationRequest((3,), 6, 18),
            GenerationRequest((4, 5, 6), 6, 19),
        ]
        expected = self._legacy(model, requests, temperature=1.0)
        actual = sample_token_ids_batched(
            model,
            requests,
            max_seq_len=16,
            eos_token_id=31,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            precision="fp32",
            batch_size=2,
        )
        self.assertEqual(actual, expected)


class L20BatchingTests(unittest.TestCase):
    def test_batched_conditional_likelihood_matches_single_request_scoring(self) -> None:
        model = TinyNextTokenLM().eval()
        adapter = SmallLLMHarnessLM(
            model,
            max_seq_len=16,
            precision="fp32",
            batch_size=3,
            max_batch_tokens=24,
        )
        pairs = [
            ([1, 2, 3], [4]),
            ([7], [8, 9]),
            ([4, 5], [6, 7, 8]),
        ]
        expected = [adapter._score_pair(context, continuation) for context, continuation in pairs]
        prepared = [adapter._prepare_pair(context, continuation) for context, continuation in pairs]
        actual = adapter._score_prepared_batch(prepared)
        self.assertEqual(len(actual), len(expected))
        for (actual_lp, actual_greedy), (expected_lp, expected_greedy) in zip(actual, expected, strict=True):
            self.assertAlmostEqual(actual_lp, expected_lp, places=6)
            self.assertEqual(actual_greedy, expected_greedy)


if __name__ == "__main__":
    unittest.main()
