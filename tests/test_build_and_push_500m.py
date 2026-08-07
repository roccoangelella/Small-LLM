"""Offline tests for the fixed 500M dataset publication overlay."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

KAGGLE_DIR = Path(__file__).resolve().parents[1] / "kaggle"
if str(KAGGLE_DIR) not in sys.path:
    sys.path.insert(0, str(KAGGLE_DIR))

import build_and_push_500m as suite  # noqa: E402


class BuildAndPush500MTests(unittest.TestCase):
    def test_production_identity_is_distinct_and_fixed(self) -> None:
        self.assertEqual(
            suite.production_identity(),
            {
                "run_id": "20m-500m-dataset-001",
                "target_source_tokens": 500_000_000,
                "minimum_source_tokens": 450_000_000,
                "maximum_source_tokens": 550_000_000,
                "checkpoint_source_tokens": 20_000_000,
                "target_reached": True,
                "remote_required": True,
            },
        )

    def test_default_handle_is_500m_specific(self) -> None:
        handle = suite.resolve_handle(None, {"KAGGLE_USERNAME": "owner"})
        self.assertEqual(handle, "owner/small-llm-20m-500m-dataset-001")

    def test_generic_100m_handle_variable_cannot_redirect_500m_upload(self) -> None:
        handle = suite.resolve_handle(
            None,
            {
                "KAGGLE_USERNAME": "owner",
                "SMALL_LLM_KAGGLE_DATASET_HANDLE": "owner/old-100m-handle",
            },
        )
        self.assertEqual(handle, "owner/small-llm-20m-500m-dataset-001")

    def test_explicit_500m_handle_is_supported(self) -> None:
        handle = suite.resolve_handle(
            None,
            {"SMALL_LLM_500M_KAGGLE_DATASET_HANDLE": "owner/custom-500m"},
        )
        self.assertEqual(handle, "owner/custom-500m")

    def test_producer_command_uses_500m_qualification_module(self) -> None:
        config = suite.Config(
            weights=Path("/weights.json"),
            dataset=Path("/dataset"),
            ops=Path("/ops"),
            handle="owner/data",
            force_upload=False,
            timeout=900,
        )
        command = suite.producer_command(config, resume=True)
        self.assertIn("dataset.qualification_500m", command)
        self.assertIn("--resume", command)


if __name__ == "__main__":
    unittest.main()
