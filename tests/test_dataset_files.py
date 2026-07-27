"""Fast regression tests for the deterministic local curation logic."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dataset import config
from dataset.src import sampling
from dataset.src.filters import filter_settings_snapshot, inspect_text, selection_rejection
from dataset.src.models import SelectionState, SourceDocument
from dataset.src.prompts import build_review_prompt, validate_review_response
from dataset.src.selection import selection_complete
from dataset.src.source import deterministic_accept, get_tokenizer, stable_document_id
from dataset.src.storage import read_jsonl


class DatasetFilesTest(unittest.TestCase):
    def test_cluster_policy_has_requested_balance(self) -> None:
        config.validate_config()
        self.assertEqual(
            config.CLUSTER_POLICIES[11].expected_topic,
            "Software Development, Programming, Web Development, JavaScript, Databases",
        )
        self.assertEqual(config.CLUSTER_POLICIES[11].decision, config.KEEP_WITHOUT_CODE)
        self.assertEqual(config.CLUSTER_POLICIES[20].decision, config.KEEP)
        self.assertIn("Public Safety", config.CLUSTER_POLICIES[20].expected_topic)
        self.assertEqual(
            sum(
                policy.quota_percent
                for policy in config.CLUSTER_POLICIES.values()
                if policy.decision in config.ACCEPTED_DECISIONS
            ),
            100,
        )

    def test_prose_passes_and_python_source_is_rejected(self) -> None:
        prose = " ".join(
            [
                "Network protocols let computers exchange information reliably across the network.",
                "This technical explanation describes packets, routing, and error recovery in English prose.",
                "Researchers use these systems to make communication dependable for people and organizations.",
            ]
            * 3
        )
        source = """example.py

from pathlib import Path

def load_data(path: str) -> list[str]:
    return Path(path).read_text().splitlines()

for item in load_data("input.txt"):
    print(item)
"""
        prose_metrics = inspect_text(prose)
        source_metrics = inspect_text(source)
        self.assertIsNone(selection_rejection(prose_metrics))
        self.assertTrue(source_metrics.code_dominated)
        self.assertIsNotNone(selection_rejection(source_metrics))

    def test_content_identity_and_sampling_are_stable(self) -> None:
        document = SourceDocument(10, 15, [1, 2, 3, 4], 4)
        document_id = stable_document_id(document)
        self.assertEqual(document_id, stable_document_id(document))
        self.assertEqual(
            deterministic_accept(document_id, 0.5),
            deterministic_accept(document_id, 0.5),
        )

    def test_cluster_quota_completion_handles_indivisible_documents(self) -> None:
        state = SelectionState()
        plan = {
            "clusters": {
                str(cluster_id): {"target_tokens": 10}
                for cluster_id in config.CLUSTER_POLICIES
            }
        }
        for cluster_id, policy in config.CLUSTER_POLICIES.items():
            if policy.decision in config.ACCEPTED_DECISIONS:
                state.cluster(cluster_id)["tokens"] = 10
        self.assertTrue(selection_complete(state, plan))

    def test_filter_snapshot_survives_json_round_trip(self) -> None:
        snapshot = filter_settings_snapshot()
        self.assertEqual(json.loads(json.dumps(snapshot)), snapshot)

    def test_review_prompt_and_schema_allow_topic_mismatch(self) -> None:
        policy = config.CLUSTER_POLICIES[8]
        prompt = build_review_prompt(8, policy, [{"token_count": 10, "text": "A short sample."}], audit=False)
        self.assertIn("hypotheses, not evidence", prompt)
        self.assertIn("topic_alignment=mismatch", prompt)
        response = {
            "schema_version": config.LLM_REVIEW_SCHEMA_VERSION,
            "observed_topics": ["history"],
            "topic_alignment": "mismatch",
            "alignment_evidence": ["The sample is historical rather than scientific."],
            "english_quality": "high",
            "code_prevalence": "none",
            "generated_reference_prevalence": "none",
            "recommendation": "keep",
            "quality_pass": True,
            "confidence": "high",
            "concerns": [],
            "rationale": "Useful prose, but it does not match the stated cluster topic.",
        }
        validate_review_response(response)

    def test_sampling_handles_identical_documents_in_one_cluster(self) -> None:
        """Duplicate content must not make the sampling heap compare dictionaries."""

        tokenizer = get_tokenizer()
        tokens = tokenizer.encode(
            "This English prose explains research, public systems, and useful general knowledge. " * 4
        )
        documents = [
            SourceDocument(source_index, cluster_id, tokens, len(tokens))
            for cluster_id in range(1, 21)
            for source_index in (cluster_id * 10, cluster_id * 10 + 1)
        ]
        original_stream = sampling.iter_climbmix_documents
        original_sample_count = config.SAMPLE_DOCUMENTS_PER_CLUSTER
        original_paths = (config.SAMPLES_PATH, config.INVENTORY_PATH, config.MANUAL_REVIEW_PATH)
        try:
            with tempfile.TemporaryDirectory() as temporary:
                artifacts = Path(temporary)
                config.SAMPLE_DOCUMENTS_PER_CLUSTER = 2
                config.SAMPLES_PATH = artifacts / "cluster_samples.jsonl"
                config.INVENTORY_PATH = artifacts / "cluster_inventory.json"
                config.MANUAL_REVIEW_PATH = artifacts / "manual_review.md"
                sampling.iter_climbmix_documents = lambda: iter(documents)
                sampling.collect_samples_and_inventory()
                self.assertEqual(len(list(read_jsonl(config.SAMPLES_PATH))), 40)
        finally:
            sampling.iter_climbmix_documents = original_stream
            config.SAMPLE_DOCUMENTS_PER_CLUSTER = original_sample_count
            config.SAMPLES_PATH, config.INVENTORY_PATH, config.MANUAL_REVIEW_PATH = original_paths


if __name__ == "__main__":
    unittest.main()
