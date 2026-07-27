"""Manifest construction and file hashing for the finalized corpus."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dataset import config

from .checkpoint import Progress
from .workplan import WorkPlan


def sha256_file(path: Path, *, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    """Stream SHA-256 over a file without loading it fully into RAM."""

    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_bytes)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


def build_manifest(
    *,
    progress: Progress,
    plan: WorkPlan,
    output_dir: Path,
    repo_commit: str | None,
    accepted_cluster_ids: frozenset[int],
    excluded_cluster_ids: frozenset[int],
    start_time: str,
    complete: bool,
) -> dict[str, Any]:
    """Assemble the complete manifest describing the corpus and its format."""

    train_path = output_dir / config.TRAIN_FILENAME
    validation_path = output_dir / config.VALIDATION_FILENAME
    plan_path = output_dir / config.WORK_PLAN_FILENAME

    train_size = train_path.stat().st_size
    validation_size = validation_path.stat().st_size

    per_cluster = {}
    for cluster_id in sorted(config.ALL_CLUSTER_IDS):
        counters = progress.per_cluster.get(str(cluster_id), {})
        if counters:
            per_cluster[str(cluster_id)] = {
                "topic": config.CLUSTER_TOPICS.get(cluster_id, ""),
                "documents": counters.get("documents", 0),
                "source_tokens": counters.get("source_tokens", 0),
                "written_tokens": counters.get("written_tokens", 0),
                "inserted_eods": counters.get("inserted_eods", 0),
            }

    total_written_tokens = progress.train_written_tokens + progress.validation_written_tokens
    expected_train_tokens = train_size // 2
    expected_validation_tokens = validation_size // 2

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    manifest: dict[str, Any] = {
        "schema_version": config.MANIFEST_SCHEMA_VERSION,
        "complete": complete,
        "dataset": config.DATASET_REPOSITORY,
        "dataset_repository": config.DATASET_REPOSITORY,
        "source_revision": config.DATASET_REVISION,
        "source_files": [{"path": f.path, "size": f.size} for f in plan.source_files],
        "source_glob": config.SOURCE_DATA_GLOB,
        "excluded_subdirectories": ["climbmix_small", "assets", "nanoGPT"],
        "work_plan_hash": plan.hash,
        "selection_seed": config.SELECTION_SEED,
        "region_bytes": plan.region_bytes,
        "accepted_cluster_ids": sorted(accepted_cluster_ids),
        "excluded_cluster_ids": sorted(excluded_cluster_ids),
        "cluster_topics": config.CLUSTER_TOPICS,
        "validation_split_rule": {
            "method": "versioned length-prefixed SHA-256 draw < probability",
            "hash_version": config.SPLIT_HASH_VERSION,
            "identity": "selection_seed + revision + filename + absolute_record_start",
            "probability": config.VALIDATION_PROBABILITY,
        },
        "tokenizer": {
            "id": config.TOKENIZER_ID,
            "description": config.TOKENIZER_DESCRIPTION,
            "vocab_size": config.VOCAB_SIZE,
        },
        "eod_token_id": config.EOD_TOKEN_ID,
        "binary_format": {
            "int_type": config.INT_TYPE,
            "byte_order": config.BYTE_ORDER,
            "bytes_per_token": 2,
            "header": "none",
            "framing": "none",
            "compression": "none",
        },
        "counts": {
            "train_file_size": train_size,
            "validation_file_size": validation_size,
            "train_token_count": expected_train_tokens,
            "validation_token_count": expected_validation_tokens,
            "train_source_token_count": progress.train_source_tokens,
            "validation_source_token_count": progress.validation_source_tokens,
            "train_inserted_eod_count": progress.train_inserted_eod_count,
            "validation_inserted_eod_count": progress.validation_inserted_eod_count,
            "train_document_count": progress.train_document_count,
            "validation_document_count": progress.validation_document_count,
            "total_written_tokens": total_written_tokens,
            "accepted_source_tokens": progress.accepted_source_tokens,
            "inserted_eod_count": progress.inserted_eod_count,
            "accepted_document_count": progress.accepted_document_count,
            "inspected_document_count": progress.inspected_document_count,
            "source_bytes_processed": progress.source_bytes_processed,
            "per_cluster": per_cluster,
            "structural_rejections": progress.structural_rejections,
            "cluster_exclusions": progress.cluster_exclusions,
        },
        "targets": {
            "target_accepted_source_tokens": progress.run_config.get(
                "target_accepted_source_tokens"
            ),
            "minimum_accepted_source_tokens": progress.run_config.get(
                "minimum_accepted_source_tokens"
            ),
            "maximum_accepted_source_tokens": progress.run_config.get(
                "maximum_accepted_source_tokens"
            ),
        },
        "timestamps": {
            "build_start": start_time,
            "completion": now if complete else None,
            "last_checkpoint": progress.last_checkpoint_time,
        },
        "software_commit": repo_commit or "",
        "hashes": {
            "train_sha256": sha256_file(train_path),
            "validation_sha256": sha256_file(validation_path),
            "work_plan_sha256": sha256_file(plan_path),
        },
        "license": config.DATASET_LICENSE,
        "license_url": "https://creativecommons.org/licenses/by-nc/4.0/",
        "attribution": config.ATTRIBUTION,
        "corpus_description": config.CORPUS_DESCRIPTION,
        "known_limitation": (
            "The corpus is selected by numeric cluster_id only; cluster 11 "
            "(programming) is excluded, but incidental code or off-topic content "
            "inside accepted clusters is not removed. It is programming-cluster-"
            "excluded, not guaranteed code-free."
        ),
    }
    return manifest
