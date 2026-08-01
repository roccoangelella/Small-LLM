"""CPU-only tests for the T4 qualification harness control and input contract."""

from __future__ import annotations

import unittest

import torch

from tests.t4_qualification import (
    PARITY_PROFILES,
    _make_parity_inputs,
    build_parser,
    choose_initialization_chunk,
    choose_recommendation,
    fully_qualified_chunks,
    validate_args,
)


def passing_parity(chunks=(16, 32, 64), precisions=("fp32", "fp16")):
    return [
        {
            "status": "pass",
            "chunk_size": chunk,
            "precision": precision,
            "profile": profile,
        }
        for chunk in chunks
        for precision in precisions
        for profile in PARITY_PROFILES
    ]


class T4QualificationControlTests(unittest.TestCase):
    def test_parser_defaults_cover_frozen_candidates(self):
        args = build_parser().parse_args([])
        self.assertEqual(args.chunk_sizes, [16, 32, 64])
        self.assertEqual(args.precisions, ["fp32", "fp16"])
        self.assertEqual(args.sequence_length, 2_048)
        validate_args(args)

    def test_validate_args_rejects_duplicate_chunks_and_precisions(self):
        args = build_parser().parse_args(["--chunk-sizes", "32", "32"])
        with self.assertRaises(ValueError):
            validate_args(args)
        args = build_parser().parse_args(["--precisions", "fp16", "fp16"])
        with self.assertRaises(ValueError):
            validate_args(args)

    def test_parity_inputs_normalize_q_and_k_and_zero_training_state(self):
        q, k, _, _, _, _, state = _make_parity_inputs(
            device=torch.device("cpu"),
            precision="fp32",
            sequence_length=17,
            seed=4,
            profile="training_zero_state",
        )
        torch.testing.assert_close(q.norm(dim=-1), torch.ones_like(q[..., 0]))
        torch.testing.assert_close(k.norm(dim=-1), torch.ones_like(k[..., 0]))
        self.assertTrue(torch.equal(state, torch.zeros_like(state)))
        self.assertEqual(state.dtype, torch.float32)

    def test_bounded_cache_state_is_small_and_deterministic(self):
        first = _make_parity_inputs(
            device=torch.device("cpu"),
            precision="fp16",
            sequence_length=9,
            seed=9,
            profile="bounded_cache_state",
        )
        second = _make_parity_inputs(
            device=torch.device("cpu"),
            precision="fp16",
            sequence_length=9,
            seed=9,
            profile="bounded_cache_state",
        )
        self.assertTrue(torch.equal(first[-1], second[-1]))
        self.assertLess(float(first[-1].abs().max()), 0.25)
        self.assertEqual(first[0].dtype, torch.float16)
        self.assertEqual(first[-1].dtype, torch.float32)

    def test_every_profile_is_required_for_qualification(self):
        parity = passing_parity(chunks=(16,), precisions=("fp32", "fp16"))
        self.assertEqual(fully_qualified_chunks(parity, ("fp32", "fp16")), {16})
        parity.pop()
        self.assertEqual(fully_qualified_chunks(parity, ("fp32", "fp16")), set())

    def test_recommendation_selects_fastest_fully_qualified_chunk(self):
        benchmarks = [
            {
                "status": "pass",
                "architecture": "gdn2_hybrid",
                "chunk_size": chunk,
                "precision": "fp16",
                "tokens_per_second": speed,
            }
            for chunk, speed in ((16, 100.0), (32, 140.0), (64, 120.0))
        ]
        recommendation = choose_recommendation(
            passing_parity(), benchmarks, ("fp32", "fp16")
        )
        self.assertEqual(recommendation["status"], "candidate")
        self.assertEqual(recommendation["chunk_size"], 32)
        self.assertEqual(recommendation["precision"], "fp16")

    def test_failed_profile_disqualifies_only_its_chunk(self):
        parity = passing_parity(chunks=(16, 32), precisions=("fp32", "fp16"))
        parity[-1]["status"] = "fail"
        benchmarks = [
            {
                "status": "pass",
                "architecture": "gdn2_hybrid",
                "chunk_size": 16,
                "precision": "fp16",
                "tokens_per_second": 100.0,
            },
            {
                "status": "pass",
                "architecture": "gdn2_hybrid",
                "chunk_size": 32,
                "precision": "fp16",
                "tokens_per_second": 200.0,
            },
        ]
        recommendation = choose_recommendation(parity, benchmarks, ("fp32", "fp16"))
        self.assertEqual(recommendation["chunk_size"], 16)

    def test_initialization_uses_fastest_qualified_fp16_chunk(self):
        benchmarks = [
            {
                "status": "pass",
                "architecture": "gdn2_hybrid",
                "chunk_size": 16,
                "precision": "fp16",
                "tokens_per_second": 100.0,
            },
            {
                "status": "pass",
                "architecture": "gdn2_hybrid",
                "chunk_size": 32,
                "precision": "fp16",
                "tokens_per_second": 150.0,
            },
            {
                "status": "fail",
                "architecture": "gdn2_hybrid",
                "chunk_size": 64,
                "precision": "fp16",
                "tokens_per_second": 999.0,
            },
        ]
        selected = choose_initialization_chunk(
            passing_parity(), benchmarks, ("fp32", "fp16")
        )
        self.assertEqual(selected, 32)

    def test_plan_b_is_selected_only_when_no_gdn_candidate_qualifies(self):
        parity = passing_parity(chunks=(64,), precisions=("fp32", "fp16"))
        for result in parity:
            result["status"] = "fail"
        benchmarks = [
            {
                "status": "pass",
                "architecture": "swa_hybrid",
                "chunk_size": None,
                "precision": "fp16",
                "tokens_per_second": 80.0,
            }
        ]
        recommendation = choose_recommendation(parity, benchmarks, ("fp32", "fp16"))
        self.assertEqual(recommendation["status"], "fallback_candidate")
        self.assertEqual(recommendation["architecture"], "swa_hybrid")

    def test_no_viable_path_is_blocked(self):
        recommendation = choose_recommendation([], [], ("fp32", "fp16"))
        self.assertEqual(recommendation["status"], "blocked")


if __name__ == "__main__":
    unittest.main()
