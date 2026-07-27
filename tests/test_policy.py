"""Guard the frozen production defaults against accidental drift."""

from __future__ import annotations

import unittest

from dataset import config


class ProductionPolicyTest(unittest.TestCase):
    def test_frozen_source_size_cluster_and_format_policy(self) -> None:
        self.assertEqual(config.DATASET_REPOSITORY, "nvidia/Nemotron-ClimbMix")
        self.assertEqual(
            config.DATASET_REVISION,
            "5eaa64b9c0c85b7f56af01d7dffdb0795816b12b",
        )
        self.assertEqual(config.SOURCE_DATA_GLOB, "part_*.tokenized.jsonl")
        self.assertEqual(config.TARGET_ACCEPTED_SOURCE_TOKENS, 90_000_000_000)
        self.assertEqual(config.MINIMUM_ACCEPTED_SOURCE_TOKENS, 80_000_000_000)
        self.assertEqual(config.MAXIMUM_ACCEPTED_SOURCE_TOKENS, 100_000_000_000)
        self.assertEqual(
            config.ACCEPTED_CLUSTER_IDS,
            frozenset(range(1, 11)) | frozenset(range(12, 21)),
        )
        self.assertEqual(config.EXCLUDED_CLUSTER_IDS, frozenset({11}))
        self.assertEqual(config.EOD_TOKEN_ID, 50256)
        self.assertEqual(config.VALIDATION_PROBABILITY, 0.001)
        self.assertEqual(
            config.SELECTION_SEED, "small-llm-climbmix-production-v1"
        )
        self.assertEqual(config.INT_TYPE, "uint16")
        self.assertEqual(config.BYTE_ORDER, "little")


if __name__ == "__main__":
    unittest.main()
