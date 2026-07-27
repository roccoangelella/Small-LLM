"""Quota calculation and deterministic sampling-rate planning."""

from __future__ import annotations

from typing import Any

from dataset import config

from .filters import filter_settings_snapshot
from .storage import read_json, write_json_atomic


def quota_tokens(cluster_id: int) -> int:
    """Return a cluster's configured target number of GPT-2 tokens."""

    return config.TARGET_TOKENS * config.CLUSTER_POLICIES[cluster_id].quota_percent // 100


def create_selection_plan() -> dict[str, Any]:
    """Turn eligible inventory and configured quotas into hash sampling rates."""

    config.validate_config()
    inventory = read_json(config.INVENTORY_PATH)
    if inventory.get("dataset") != config.DATASET_REPOSITORY or inventory.get("revision") != config.DATASET_REVISION:
        raise ValueError("Inventory belongs to a different dataset/revision; rerun `sample`")
    if inventory.get("filter_settings") != filter_settings_snapshot():
        raise ValueError("Filter settings changed since inventory; rerun `sample`")

    clusters: dict[str, Any] = {}
    insufficiencies: list[str] = []
    for cluster_id, policy in sorted(config.CLUSTER_POLICIES.items()):
        counts = inventory["clusters"].get(str(cluster_id), {})
        eligible_tokens = int(counts.get("eligible_tokens", 0))
        target = quota_tokens(cluster_id) if policy.decision in config.ACCEPTED_DECISIONS else 0
        if target:
            availability_ratio = eligible_tokens / target
            if availability_ratio < config.MINIMUM_CLUSTER_AVAILABILITY_RATIO:
                insufficiencies.append(
                    f"cluster {cluster_id}: only {eligible_tokens:,} eligible tokens for {target:,} quota"
                )
            rate = min(1.0, target * config.PLANNING_OVERSUBSCRIPTION / max(1, eligible_tokens))
        else:
            availability_ratio = 0.0
            rate = 0.0
        clusters[str(cluster_id)] = {
            "decision": policy.decision,
            "quota_percent": policy.quota_percent,
            "target_tokens": target,
            "eligible_tokens": eligible_tokens,
            "eligible_documents": int(counts.get("eligible_documents", 0)),
            "availability_ratio": round(availability_ratio, 6),
            "deterministic_sampling_rate": rate,
        }
    if insufficiencies:
        raise RuntimeError("Configured quotas cannot be met:\n" + "\n".join(insufficiencies))

    plan = {
        "dataset": config.DATASET_REPOSITORY,
        "revision": config.DATASET_REVISION,
        "source_glob": config.DATASET_DATA_FILES_GLOB,
        "random_seed": config.RANDOM_SEED,
        "target_tokens": config.TARGET_TOKENS,
        "target_text_bytes": config.TARGET_TEXT_BYTES,
        "maximum_tokens": config.MAXIMUM_TOKENS,
        "maximum_text_bytes": config.MAXIMUM_TEXT_BYTES,
        "oversubscription": config.PLANNING_OVERSUBSCRIPTION,
        "max_cluster_quota_overshoot_tokens": config.MAX_CLUSTER_QUOTA_OVERSHOOT_TOKENS,
        "filter_settings": filter_settings_snapshot(),
        "clusters": clusters,
    }
    write_json_atomic(config.SELECTION_PLAN_PATH, plan)
    return plan


def load_selection_plan() -> dict[str, Any]:
    """Load a plan only if it still matches the live source and filter config."""

    plan = read_json(config.SELECTION_PLAN_PATH)
    checks = {
        "dataset": config.DATASET_REPOSITORY,
        "revision": config.DATASET_REVISION,
        "source_glob": config.DATASET_DATA_FILES_GLOB,
        "random_seed": config.RANDOM_SEED,
        "filter_settings": filter_settings_snapshot(),
    }
    for name, expected in checks.items():
        if plan.get(name) != expected:
            raise ValueError(f"Selection plan {name!r} disagrees with config; rerun `sample` and `plan`")
    return plan
