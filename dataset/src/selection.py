"""Quota-balanced corpus selection from the streamed ClimbMix source."""

from __future__ import annotations

import logging
import time
from collections import Counter
from typing import Any

from dataset import config

from .filters import inspect_text, selection_rejection
from .models import SelectionState
from .planning import create_selection_plan, load_selection_plan
from .shards import JsonlShardWriter
from .source import decode_document, deterministic_accept, iter_climbmix_documents, stable_document_id
from .storage import write_json_atomic


LOGGER = logging.getLogger(__name__)


def selection_complete(state: SelectionState, plan: dict[str, Any]) -> bool:
    """Return true only when every accepted cluster has reached its quota."""

    return all(
        state.cluster(cluster_id)["tokens"] >= int(plan["clusters"][str(cluster_id)]["target_tokens"])
        for cluster_id, policy in config.CLUSTER_POLICIES.items()
        if policy.decision in config.ACCEPTED_DECISIONS
    )


def select_corpus(*, resume: bool) -> dict[str, Any]:
    """Stream source documents and write quota-balanced, non-code output JSONL."""

    config.validate_config()
    plan = load_selection_plan() if config.REQUIRE_SELECTION_PLAN else create_selection_plan()
    if config.SELECTION_STATE_PATH.exists() and not resume:
        raise RuntimeError(
            f"Existing selection state found at {config.SELECTION_STATE_PATH}. Use `select --resume` "
            "or move the output directory before starting a new run."
        )
    if not resume and any(config.OUTPUT_DIR.glob("part-*.jsonl")):
        raise RuntimeError(
            f"Existing output shards found in {config.OUTPUT_DIR}. Move them before starting a new run."
        )
    state = SelectionState.load(config.SELECTION_STATE_PATH) if resume else SelectionState()
    writer = JsonlShardWriter(config.OUTPUT_DIR, state)
    rejected: Counter[str] = Counter()
    started = time.monotonic()
    last_checkpoint_document_count = state.total_documents
    try:
        # Persist an initial empty checkpoint so even an early interruption is
        # resumable and cannot be mistaken for a clean new output directory.
        writer.checkpoint(state)
        write_json_atomic(config.SELECTION_STATE_PATH, state.to_dict())
        for document in iter_climbmix_documents():
            if document.source_index <= state.last_source_index:
                continue
            if selection_complete(state, plan):
                break
            policy = config.CLUSTER_POLICIES.get(document.cluster_id)
            if policy is None or policy.decision not in config.ACCEPTED_DECISIONS:
                rejected["cluster_excluded"] += 1
                continue
            cluster_plan = plan["clusters"][str(document.cluster_id)]
            cluster_state = state.cluster(document.cluster_id)
            target = int(cluster_plan["target_tokens"])
            if cluster_state["tokens"] >= target:
                rejected["cluster_quota_filled"] += 1
                continue

            document_id = stable_document_id(document)
            rate = float(cluster_plan["deterministic_sampling_rate"])
            if not deterministic_accept(document_id, rate):
                rejected["deterministic_sample"] += 1
                continue
            text = decode_document(document)
            metrics = inspect_text(text)
            rejection = selection_rejection(metrics)
            if rejection:
                rejected[rejection] += 1
                continue
            text_bytes = len(text.encode("utf-8"))
            if (
                cluster_state["tokens"] + document.token_count
                > target + config.MAX_CLUSTER_QUOTA_OVERSHOOT_TOKENS
            ):
                rejected["cluster_token_cap"] += 1
                continue
            if state.total_tokens + document.token_count > config.MAXIMUM_TOKENS:
                raise RuntimeError("Selection reached MAXIMUM_TOKENS before all cluster quotas were filled")
            if state.total_text_bytes + text_bytes > config.MAXIMUM_TEXT_BYTES:
                raise RuntimeError("Selection reached MAXIMUM_TEXT_BYTES before all cluster quotas were filled")

            record = {
                "text": text,
                "cluster_id": document.cluster_id,
                "token_count": document.token_count,
                "source_index": document.source_index,
                "document_id": document_id,
            }
            writer.write(record)
            state.total_documents += 1
            state.total_tokens += document.token_count
            state.total_text_bytes += text_bytes
            cluster_state["documents"] += 1
            cluster_state["tokens"] += document.token_count
            cluster_state["text_bytes"] += text_bytes
            state.last_source_index = document.source_index

            if state.total_documents - last_checkpoint_document_count >= config.CHECKPOINT_EVERY_DOCUMENTS:
                writer.checkpoint(state)
                write_json_atomic(config.SELECTION_STATE_PATH, state.to_dict())
                last_checkpoint_document_count = state.total_documents
            if document.source_index and document.source_index % config.PROGRESS_EVERY_DOCUMENTS == 0:
                LOGGER.info(
                    "Selection: %,d source docs; %,d selected; %.2fB tokens; %.1f GB text; %.1f min",
                    document.source_index,
                    state.total_documents,
                    state.total_tokens / 1e9,
                    state.total_text_bytes / 1e9,
                    (time.monotonic() - started) / 60,
                )
        writer.checkpoint(state)
        write_json_atomic(config.SELECTION_STATE_PATH, state.to_dict())
    finally:
        writer.close()

    complete = selection_complete(state, plan)
    manifest = {
        "complete": complete,
        "dataset": config.DATASET_REPOSITORY,
        "revision": config.DATASET_REVISION,
        "source_glob": config.DATASET_DATA_FILES_GLOB,
        "random_seed": config.RANDOM_SEED,
        "selection_plan": str(config.SELECTION_PLAN_PATH),
        "total_documents": state.total_documents,
        "total_tokens": state.total_tokens,
        "total_text_bytes": state.total_text_bytes,
        "target_tokens": config.TARGET_TOKENS,
        "target_text_bytes": config.TARGET_TEXT_BYTES,
        "per_cluster": state.per_cluster,
        "run_rejections": dict(rejected),
    }
    write_json_atomic(config.SELECTION_MANIFEST_PATH, manifest)
    if not complete:
        raise RuntimeError(
            "Source stream ended before every quota was fulfilled. Inspect the manifest and revise "
            "quotas/filter settings, then rebuild the sample inventory and plan."
        )
    return manifest
