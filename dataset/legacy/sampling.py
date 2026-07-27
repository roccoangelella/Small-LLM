"""Pre-selection sampling, inventory construction, and manual worksheet output."""

from __future__ import annotations

import heapq
import logging
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from typing import Any, Iterable

from dataset import config

from .filters import filter_settings_snapshot, inspect_text, selection_rejection
from .models import SourceDocument, TextMetrics
from .source import decode_document, iter_climbmix_documents, stable_document_id, stable_priority
from .storage import ensure_parent, write_json_atomic, write_jsonl


LOGGER = logging.getLogger(__name__)


def _sample_row(
    document: SourceDocument,
    document_id: str,
    text: str,
    metrics: TextMetrics,
) -> dict[str, Any]:
    """Convert a source document to the compact artifact review schema."""

    return {
        "cluster_id": document.cluster_id,
        "source_index": document.source_index,
        "document_id": document_id,
        "token_count": document.token_count,
        "metrics": asdict(metrics),
        "text": text[: config.SAMPLE_TEXT_CHARACTERS],
        "text_truncated": len(text) > config.SAMPLE_TEXT_CHARACTERS,
    }


def collect_samples_and_inventory() -> dict[str, Any]:
    """Create 50 deterministic examples per cluster and eligible-token inventory.

    A bounded max-heap makes the samples a uniform hash sample over the complete
    stream while retaining at most 1,000 examples in memory.  The same remote
    stream supplies the filtered-token inventory used by the quota planner.
    """

    config.validate_config()
    # ``document_id`` deliberately identifies identical content.  Exact
    # duplicates can therefore have the same priority and ID, so source_index
    # is the final stable tie-breaker before the non-orderable row dictionary.
    heaps: dict[int, list[tuple[int, str, int, dict[str, Any]]]] = defaultdict(list)
    inventory: dict[int, Counter[str]] = defaultdict(Counter)
    started = time.monotonic()

    for document in iter_climbmix_documents():
        if document.cluster_id not in config.CLUSTER_POLICIES:
            LOGGER.warning("Ignoring unknown cluster %d at row %d", document.cluster_id, document.source_index)
            continue
        document_id = stable_document_id(document)
        priority = stable_priority(document_id, "preselection-sample")
        heap = heaps[document.cluster_id]
        needs_sample_text = len(heap) < config.SAMPLE_DOCUMENTS_PER_CLUSTER or priority < -heap[0][0]
        text = decode_document(document)
        metrics = inspect_text(text)
        rejection = selection_rejection(metrics)

        counts = inventory[document.cluster_id]
        counts["source_documents"] += 1
        counts["source_tokens"] += document.token_count
        if rejection is None:
            counts["eligible_documents"] += 1
            counts["eligible_tokens"] += document.token_count
            counts["eligible_text_bytes"] += len(text.encode("utf-8"))
        else:
            counts[f"rejected_{rejection}"] += 1
            counts["rejected_documents"] += 1
            counts["rejected_tokens"] += document.token_count

        if needs_sample_text:
            row = _sample_row(document, document_id, text, metrics)
            entry = (-priority, document_id, document.source_index, row)
            if len(heap) < config.SAMPLE_DOCUMENTS_PER_CLUSTER:
                heapq.heappush(heap, entry)
            elif priority < -heap[0][0]:
                heapq.heapreplace(heap, entry)

        if document.source_index and document.source_index % config.PROGRESS_EVERY_DOCUMENTS == 0:
            LOGGER.info(
                "Sample/inventory: %,d documents in %.1f min",
                document.source_index,
                (time.monotonic() - started) / 60,
            )

    missing = [cluster_id for cluster_id in config.CLUSTER_POLICIES if not heaps[cluster_id]]
    if missing:
        raise RuntimeError(f"No source samples found for clusters: {missing}")

    rows: list[dict[str, Any]] = []
    for cluster_id in sorted(config.CLUSTER_POLICIES):
        ordered = sorted(heaps[cluster_id], key=lambda entry: (-entry[0], entry[1], entry[2]))
        rows.extend(entry[3] for entry in ordered)
    write_jsonl(config.SAMPLES_PATH, rows)

    inventory_payload = {
        "dataset": config.DATASET_REPOSITORY,
        "revision": config.DATASET_REVISION,
        "source_glob": config.DATASET_DATA_FILES_GLOB,
        "random_seed": config.RANDOM_SEED,
        "filter_settings": filter_settings_snapshot(),
        "clusters": {str(cluster_id): dict(inventory[cluster_id]) for cluster_id in sorted(config.CLUSTER_POLICIES)},
    }
    write_json_atomic(config.INVENTORY_PATH, inventory_payload)
    write_manual_review_worksheet(rows)
    return inventory_payload


def write_manual_review_worksheet(rows: Iterable[dict[str, Any]]) -> None:
    """Write a small human-readable spot-check worksheet from sampled rows."""

    by_cluster: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cluster[int(row["cluster_id"])].append(row)
    lines = [
        "# Manual cluster spot-check\n",
        "Review the excerpts below alongside `cluster_review_summary.json`. If a cluster is misclassified, edit `CLUSTER_POLICIES` in `dataset/config.py`, rerun `plan`, then run selection. Do not change a generated plan manually.\n",
    ]
    for cluster_id, policy in sorted(config.CLUSTER_POLICIES.items()):
        lines.append(f"## Cluster {cluster_id}: {policy.expected_topic}\n")
        lines.append(f"Configured decision: `{policy.decision}`; quota: `{policy.quota_percent}%`.\n")
        for number, row in enumerate(by_cluster[cluster_id][: config.MANUAL_EXCERPTS_PER_CLUSTER], start=1):
            metrics = row["metrics"]
            excerpt = row["text"][: config.MANUAL_EXCERPT_CHARACTERS].strip()
            lines.append(
                f"### Example {number} — {row['token_count']} tokens; "
                f"code={metrics['code_dominated']}; English={metrics['likely_english']}\n"
            )
            lines.append("```text\n")
            lines.append(excerpt + "\n")
            lines.append("```\n")
        lines.append("Manual decision / notes: ______________________________________________\n")
    ensure_parent(config.MANUAL_REVIEW_PATH)
    config.MANUAL_REVIEW_PATH.write_text("\n".join(lines), encoding="utf-8")
