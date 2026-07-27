"""Gemini-based review for the pre-selection cluster samples."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from dataset import config

from .gemrouter import gemrouter_json
from .prompts import build_review_prompt, validate_review_response
from .storage import read_jsonl, write_json_atomic


def review_samples_with_llm() -> dict[str, Any]:
    """Review every sampled cluster in small Gemini batches and save raw verdicts."""

    if not config.SAMPLES_PATH.exists():
        raise FileNotFoundError(f"Run `sample` first; no samples at {config.SAMPLES_PATH}")
    by_cluster: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(config.SAMPLES_PATH):
        by_cluster[int(row["cluster_id"])].append(row)

    config.LLM_REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "model": config.GEMROUTER_MODEL,
        "dataset": config.DATASET_REPOSITORY,
        "revision": config.DATASET_REVISION,
        "clusters": {},
    }
    for cluster_id, policy in sorted(config.CLUSTER_POLICIES.items()):
        rows = by_cluster[cluster_id]
        if not rows:
            raise RuntimeError(f"No samples found for cluster {cluster_id}")
        batch_reviews = []
        for start in range(0, len(rows), config.LLM_REVIEW_BATCH_SIZE):
            batch = rows[start : start + config.LLM_REVIEW_BATCH_SIZE]
            batch_reviews.append(
                {
                    "sample_numbers": list(range(start + 1, start + len(batch) + 1)),
                    "review": gemrouter_json(
                        build_review_prompt(cluster_id, policy, batch, audit=False),
                        config.LLM_REVIEW_MAX_TOKENS,
                        validator=validate_review_response,
                    ),
                }
            )
        payload = {
            "cluster_id": cluster_id,
            "expected_topic": policy.expected_topic,
            "configured_decision": policy.decision,
            "configured_quota_percent": policy.quota_percent,
            "batch_reviews": batch_reviews,
        }
        path = config.LLM_REVIEW_DIR / f"cluster_{cluster_id:02d}.json"
        write_json_atomic(path, payload)
        summary["clusters"][str(cluster_id)] = {
            "configured_decision": policy.decision,
            "configured_quota_percent": policy.quota_percent,
            "batch_reviews_path": str(path),
            "recommendations": [item["review"].get("recommendation") for item in batch_reviews],
            "topic_alignment": [item["review"].get("topic_alignment") for item in batch_reviews],
            "code_prevalence": [item["review"].get("code_prevalence") for item in batch_reviews],
            "english_quality": [item["review"].get("english_quality") for item in batch_reviews],
        }
    write_json_atomic(config.REVIEW_SUMMARY_PATH, summary)
    return summary
