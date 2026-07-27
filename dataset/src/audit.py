"""Final selected-corpus sampling and quality audit."""

from __future__ import annotations

import heapq
from collections import Counter, defaultdict
from dataclasses import asdict
from typing import Any

from dataset import config

from .filters import inspect_text
from .gemrouter import gemrouter_json
from .prompts import build_review_prompt, validate_review_response
from .shards import iter_selected_documents
from .source import stable_priority
from .storage import read_json, write_json_atomic, write_jsonl


def audit_selected_corpus() -> dict[str, Any]:
    """Freshly sample selected data, recompute deterministic checks, and use Gemini."""

    manifest = read_json(config.SELECTION_MANIFEST_PATH)
    if not manifest.get("complete"):
        raise RuntimeError("Cannot audit an incomplete selection")
    heaps: dict[int, list[tuple[int, str, dict[str, Any]]]] = defaultdict(list)
    counts: dict[int, Counter[str]] = defaultdict(Counter)
    for record in iter_selected_documents():
        cluster_id = int(record["cluster_id"])
        text = record["text"]
        metrics = inspect_text(text)
        counters = counts[cluster_id]
        counters["documents"] += 1
        counters["tokens"] += int(record["token_count"])
        counters["text_bytes"] += len(text.encode("utf-8"))
        counters["code_dominated"] += int(metrics.code_dominated)
        counters["likely_english"] += int(metrics.likely_english)
        document_id = str(record["document_id"])
        priority = stable_priority(document_id, "final-audit")
        heap = heaps[cluster_id]
        if len(heap) < config.AUDIT_DOCUMENTS_PER_CLUSTER or priority < -heap[0][0]:
            row = {
                **record,
                "text": text[: config.AUDIT_TEXT_CHARACTERS],
                "text_truncated": len(text) > config.AUDIT_TEXT_CHARACTERS,
                "metrics": asdict(metrics),
            }
            entry = (-priority, document_id, row)
            if len(heap) < config.AUDIT_DOCUMENTS_PER_CLUSTER:
                heapq.heappush(heap, entry)
            else:
                heapq.heapreplace(heap, entry)

    audit_rows: list[dict[str, Any]] = []
    for cluster_id in sorted(heaps):
        ordered = sorted(heaps[cluster_id], key=lambda entry: (-entry[0], entry[1]))
        audit_rows.extend(entry[2] for entry in ordered)
    samples_path = config.AUDIT_DIR / "audit_samples.jsonl"
    write_jsonl(samples_path, audit_rows)

    total_documents = sum(counter["documents"] for counter in counts.values())
    total_code = sum(counter["code_dominated"] for counter in counts.values())
    total_english = sum(counter["likely_english"] for counter in counts.values())
    metrics_payload = {
        "total_documents": total_documents,
        "total_code_dominated": total_code,
        "code_dominated_fraction": total_code / max(1, total_documents),
        "total_likely_english": total_english,
        "likely_english_fraction": total_english / max(1, total_documents),
        "clusters": {str(cluster_id): dict(counter) for cluster_id, counter in sorted(counts.items())},
    }
    write_json_atomic(config.AUDIT_DIR / "audit_metrics.json", metrics_payload)

    llm_payload: dict[str, Any] = {"model": config.GEMROUTER_MODEL, "clusters": {}}
    if config.AUDIT_USE_LLM:
        by_cluster: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in audit_rows:
            by_cluster[int(row["cluster_id"])].append(row)
        for cluster_id, rows in sorted(by_cluster.items()):
            policy = config.CLUSTER_POLICIES[cluster_id]
            batch_reviews = []
            for start in range(0, len(rows), config.AUDIT_LLM_BATCH_SIZE):
                batch = rows[start : start + config.AUDIT_LLM_BATCH_SIZE]
                batch_reviews.append(
                    gemrouter_json(
                        build_review_prompt(cluster_id, policy, batch, audit=True),
                        config.AUDIT_LLM_MAX_TOKENS,
                        validator=validate_review_response,
                    )
                )
            llm_payload["clusters"][str(cluster_id)] = batch_reviews
    write_json_atomic(config.AUDIT_DIR / "audit_llm_reviews.json", llm_payload)

    passed = (
        metrics_payload["code_dominated_fraction"] <= config.MAX_AUDIT_CODE_DOMINATED_FRACTION
        and metrics_payload["likely_english_fraction"] >= config.MIN_AUDIT_ENGLISH_FRACTION
    )
    report = {
        "passed_deterministic_thresholds": passed,
        "thresholds": {
            "maximum_code_dominated_fraction": config.MAX_AUDIT_CODE_DOMINATED_FRACTION,
            "minimum_likely_english_fraction": config.MIN_AUDIT_ENGLISH_FRACTION,
        },
        "metrics": metrics_payload,
        "audit_samples": str(samples_path),
        "llm_reviews": str(config.AUDIT_DIR / "audit_llm_reviews.json"),
    }
    write_json_atomic(config.AUDIT_DIR / "audit_report.json", report)
    return report
