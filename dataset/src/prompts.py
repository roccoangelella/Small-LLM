"""Prompt construction shared by pre-selection review and final audit."""

from __future__ import annotations

import json
from typing import Any

from dataset import config


_REVIEW_ENUMS = {
    "topic_alignment": {"match", "partial_match", "mismatch", "unclear"},
    "english_quality": {"high", "mixed", "low"},
    "code_prevalence": {"none", "low", "medium", "high"},
    "generated_reference_prevalence": {"none", "low", "medium", "high"},
    "recommendation": {"keep", "keep_without_code", "exclude_or_downweight"},
    "confidence": {"low", "medium", "high"},
}


def validate_review_response(response: dict[str, Any]) -> None:
    """Require Gemini's review to match the stable artifact schema exactly enough.

    Validation happens before the response is persisted. The GemRouter client
    retries a malformed response using the same prompt, rather than producing a
    partially structured review that would be hard to compare across clusters.
    """

    required_keys = {
        "schema_version", "observed_topics", "topic_alignment", "alignment_evidence",
        "english_quality", "code_prevalence", "generated_reference_prevalence",
        "recommendation", "quality_pass", "confidence", "concerns", "rationale",
    }
    missing = sorted(required_keys - response.keys())
    if missing:
        raise ValueError(f"Review response omitted required fields: {', '.join(missing)}")
    if response["schema_version"] != config.LLM_REVIEW_SCHEMA_VERSION:
        raise ValueError("Review response has an unsupported schema version")
    for field, allowed in _REVIEW_ENUMS.items():
        if response[field] not in allowed:
            raise ValueError(f"Review response has invalid {field!r}")
    for field in ("observed_topics", "alignment_evidence", "concerns"):
        if not isinstance(response[field], list) or not all(isinstance(item, str) for item in response[field]):
            raise ValueError(f"Review response field {field!r} must be a list of strings")
    if not isinstance(response["quality_pass"], bool) or not isinstance(response["rationale"], str):
        raise ValueError("Review response has invalid quality_pass or rationale")


def build_review_prompt(
    cluster_id: int,
    policy: config.ClusterPolicy,
    rows: list[dict[str, Any]],
    *,
    audit: bool,
) -> str:
    """Construct a bounded JSON-review prompt for a cluster batch."""

    mode = "final selected-corpus audit" if audit else "pre-selection cluster review"
    text_limit = config.AUDIT_TEXT_CHARACTERS if audit else config.LLM_REVIEW_TEXT_CHARACTERS
    documents = [
        {
            "sample": index + 1,
            "tokens": row["token_count"],
            "text": row["text"][:text_limit],
        }
        for index, row in enumerate(rows)
    ]
    return (
        f"You are reviewing {mode} samples from NVIDIA Nemotron-ClimbMix. "
        f"Cluster ID: {cluster_id}. The expected theme is {policy.expected_topic!r}; "
        f"the provisional policy is {policy.decision!r}.\n\n"
        "The expected theme and policy are hypotheses, not evidence. First derive the observed topics from the samples "
        "alone; do not repeat the expected theme just because it was supplied. Then compare the independent judgement "
        "against that hypothesis. Use topic_alignment=mismatch when the samples do not support it. Assess language quality, "
        "code/source/API-dump prevalence, and whether the prose is useful for English general-knowledge pretraining. "
        "Recommend exactly one of keep, keep_without_code, exclude_or_downweight. quality_pass is true only when the "
        "batch is suitable after the stated policy; it is not a request to agree with the configured policy.\n\n"
        "Return exactly one JSON object with no extra keys, using this fixed schema and enum values: "
        '{"schema_version":' + str(config.LLM_REVIEW_SCHEMA_VERSION)
        + ',"observed_topics":["..."],"topic_alignment":"match|partial_match|mismatch|unclear",'
        '"alignment_evidence":["..."],"english_quality":"high|mixed|low",'
        '"code_prevalence":"none|low|medium|high","generated_reference_prevalence":"none|low|medium|high",'
        '"recommendation":"keep|keep_without_code|exclude_or_downweight","quality_pass":true,'
        '"confidence":"low|medium|high","concerns":["..."],"rationale":"..."}.\n\n'
        "Samples:\n" + json.dumps(documents, ensure_ascii=False)
    )
