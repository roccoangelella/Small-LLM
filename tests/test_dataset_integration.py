"""Synthetic end-to-end regression for the staged curation pipeline."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from dataset import config
from dataset.src import audit, sampling, selection
from dataset.src.filters import inspect_text
from dataset.src.models import SourceDocument
from dataset.src.planning import create_selection_plan
from dataset.src.shards import iter_selected_documents
from dataset.src.source import get_tokenizer
from dataset.src.storage import read_json, read_jsonl


class DatasetIntegrationTest(unittest.TestCase):
    """Exercise every local stage without source downloads or production paths."""

    def test_full_pipeline_resumes_and_audits_cleanly(self) -> None:
        config_names = (
            "ARTIFACTS_DIR", "SAMPLES_PATH", "INVENTORY_PATH", "SELECTION_PLAN_PATH",
            "MANUAL_REVIEW_PATH", "LLM_REVIEW_DIR", "REVIEW_SUMMARY_PATH", "OUTPUT_DIR",
            "SELECTION_STATE_PATH", "SELECTION_MANIFEST_PATH", "AUDIT_DIR", "MINIMUM_TOKENS",
            "TARGET_TOKENS", "MAXIMUM_TOKENS", "TARGET_TEXT_BYTES", "MAXIMUM_TEXT_BYTES",
            "PLANNING_OVERSUBSCRIPTION", "MAX_CLUSTER_QUOTA_OVERSHOOT_TOKENS",
            "CHECKPOINT_EVERY_DOCUMENTS", "OUTPUT_SHARD_MAX_BYTES", "AUDIT_USE_LLM",
            "AUDIT_DOCUMENTS_PER_CLUSTER",
        )
        original_config = {name: getattr(config, name) for name in config_names}
        original_sampling_stream = sampling.iter_climbmix_documents
        original_selection_stream = selection.iter_climbmix_documents
        try:
            with tempfile.TemporaryDirectory(prefix="climbmix-integration-") as temporary:
                root = Path(temporary)
                self._configure_temporary_pipeline(root)
                documents = self._synthetic_documents()

                def full_stream():
                    yield from documents

                sampling.iter_climbmix_documents = full_stream
                selection.iter_climbmix_documents = full_stream
                inventory = sampling.collect_samples_and_inventory()
                self.assertEqual(
                    len(list(read_jsonl(config.SAMPLES_PATH))),
                    20 * config.SAMPLE_DOCUMENTS_PER_CLUSTER,
                )
                self.assertEqual(inventory["clusters"]["20"].get("eligible_documents", 0), 0)
                self.assertGreater(inventory["clusters"]["15"]["rejected_documents"], 0)
                plan = create_selection_plan()
                self.assertTrue(
                    all(
                        details["deterministic_sampling_rate"] > 0
                        for cluster_id, details in plan["clusters"].items()
                        if config.CLUSTER_POLICIES[int(cluster_id)].decision in config.ACCEPTED_DECISIONS
                    )
                )

                faulted = {"value": False}

                def interrupted_stream():
                    for document in documents:
                        if not faulted["value"] and document.source_index == 250:
                            faulted["value"] = True
                            raise RuntimeError("intentional smoke-test interruption")
                        yield document

                selection.iter_climbmix_documents = interrupted_stream
                with self.assertRaisesRegex(RuntimeError, "intentional smoke-test interruption"):
                    selection.select_corpus(resume=False)
                self.assertGreaterEqual(read_json(config.SELECTION_STATE_PATH)["total_documents"], 3)

                selection.iter_climbmix_documents = full_stream
                manifest = selection.select_corpus(resume=True)
                self.assertTrue(manifest["complete"])
                selected = list(iter_selected_documents())
                self.assertTrue(selected)
                self.assertEqual(len({record["document_id"] for record in selected}), len(selected))
                self.assertNotIn(20, {record["cluster_id"] for record in selected})
                self.assertTrue(all(not inspect_text(record["text"]).code_dominated for record in selected))
                self.assertGreater(len(list(config.OUTPUT_DIR.glob("part-*.jsonl"))), 1)

                report = audit.audit_selected_corpus()
                self.assertTrue(report["passed_deterministic_thresholds"])
                self.assertEqual(report["metrics"]["total_code_dominated"], 0)
                self.assertEqual(report["metrics"]["likely_english_fraction"], 1)
        finally:
            sampling.iter_climbmix_documents = original_sampling_stream
            selection.iter_climbmix_documents = original_selection_stream
            for name, value in original_config.items():
                setattr(config, name, value)

    def _configure_temporary_pipeline(self, root: Path) -> None:
        """Set compact values that preserve production stage invariants."""

        config.ARTIFACTS_DIR = root / "artifacts"
        config.SAMPLES_PATH = config.ARTIFACTS_DIR / "cluster_samples.jsonl"
        config.INVENTORY_PATH = config.ARTIFACTS_DIR / "cluster_inventory.json"
        config.SELECTION_PLAN_PATH = config.ARTIFACTS_DIR / "selection_plan.json"
        config.MANUAL_REVIEW_PATH = config.ARTIFACTS_DIR / "manual_review.md"
        config.LLM_REVIEW_DIR = config.ARTIFACTS_DIR / "llm_reviews"
        config.REVIEW_SUMMARY_PATH = config.ARTIFACTS_DIR / "cluster_review_summary.json"
        config.OUTPUT_DIR = root / "output"
        config.SELECTION_STATE_PATH = config.OUTPUT_DIR / "selection_state.json"
        config.SELECTION_MANIFEST_PATH = config.OUTPUT_DIR / "selection_manifest.json"
        config.AUDIT_DIR = config.ARTIFACTS_DIR / "audit"
        config.MINIMUM_TOKENS = 1
        config.TARGET_TOKENS = 5_000
        config.MAXIMUM_TOKENS = 20_000
        config.TARGET_TEXT_BYTES = 100_000
        config.MAXIMUM_TEXT_BYTES = 10_000_000
        config.PLANNING_OVERSUBSCRIPTION = 10.0
        config.MAX_CLUSTER_QUOTA_OVERSHOOT_TOKENS = 1_000
        config.CHECKPOINT_EVERY_DOCUMENTS = 3
        config.OUTPUT_SHARD_MAX_BYTES = 1_700
        config.AUDIT_USE_LLM = False
        config.AUDIT_DOCUMENTS_PER_CLUSTER = 100
        config.validate_config()

    def _synthetic_documents(self) -> list[SourceDocument]:
        """Make 20 clusters with English prose plus deliberately rejected code."""

        tokenizer = get_tokenizer()
        documents: list[SourceDocument] = []
        source_index = 0
        for cluster_id in range(1, 21):
            for number in range(80):
                if cluster_id == 20:
                    text = """module.py\n\nfrom pathlib import Path\n\ndef load_items(path: str) -> list[str]:\n    return Path(path).read_text().splitlines()\n\nfor item in load_items('input.txt'):\n    print(item)\n""" * 4
                elif cluster_id in {15, 17} and number % 8 == 0:
                    text = """example.py\n\nimport json\n\ndef render(value: dict[str, str]) -> str:\n    return json.dumps(value)\n\nfor value in range(10):\n    print(render({'value': str(value)}))\n""" * 4
                else:
                    text = (
                        f"This is English general-knowledge article {number} for cluster {cluster_id}. "
                        "It explains scientific evidence, history, public institutions, and practical systems in clear prose. "
                        "Readers can use the information to understand research, communities, health, environment, and technology. "
                    ) * 8
                tokens = tokenizer.encode(text)
                documents.append(SourceDocument(source_index, cluster_id, tokens, len(tokens)))
                source_index += 1
        return documents


if __name__ == "__main__":
    unittest.main()
