"""CPU-only tests for the T4 qualification harness control logic."""

from __future__ import annotations

import unittest

from tests.t4_qualification import build_parser, choose_recommendation, validate_args


class T4QualificationControlTests(unittest.TestCase):
    def test_parser_defaults_cover_frozen_chunk_candidates(self):
        args = build_parser().parse_args([])
        self.assertEqual(args.chunk_sizes, [16, 32, 64])
        self.assertEqual(args.precisions, ["fp32", "fp16"])
        self.assertEqual(args.sequence_length, 2_048)
        validate_args(args)

    def test_validate_args_rejects_duplicate_chunks(self):
        args = build_parser().parse_args(["--chunk-sizes", "32", "32"])
        with self.assertRaises(ValueError):
            validate_args(args)

    def test_recommendation_selects_fastest_fully_qualified_chunk(self):
        parity = [
            {"status": "pass", "chunk_size": chunk, "precision": precision}
            for chunk in (16, 32, 64)
            for precision in ("fp32", "fp16")
        ]
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
                "tokens_per_second": 140.0,
            },
            {
                "status": "pass",
                "architecture": "gdn2_hybrid",
                "chunk_size": 64,
                "precision": "fp16",
                "tokens_per_second": 120.0,
            },
        ]
        recommendation = choose_recommendation(parity, benchmarks, ("fp32", "fp16"))
        self.assertEqual(recommendation["status"], "candidate")
        self.assertEqual(recommendation["chunk_size"], 32)
        self.assertEqual(recommendation["precision"], "fp16")

    def test_chunk_must_pass_every_requested_parity_precision(self):
        parity = [
            {"status": "pass", "chunk_size": 16, "precision": "fp32"},
            {"status": "pass", "chunk_size": 16, "precision": "fp16"},
            {"status": "fail", "chunk_size": 32, "precision": "fp32"},
            {"status": "pass", "chunk_size": 32, "precision": "fp16"},
        ]
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

    def test_plan_b_is_only_selected_when_no_gdn_candidate_qualifies(self):
        parity = [
            {"status": "fail", "chunk_size": 64, "precision": "fp32"},
            {"status": "fail", "chunk_size": 64, "precision": "fp16"},
        ]
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
