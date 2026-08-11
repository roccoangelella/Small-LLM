"""Accelerated production builder for the frozen ``eval_core_v1`` corpus.

This module preserves the exact eval-core selection policy and output ordering while
making the remote source scan practical:

* record identity is hashed before JSON/token parsing, so only the ~0.1% validation
  candidates are deserialized;
* immutable 256 MiB source regions are scanned concurrently;
* completed region results are consumed strictly in the frozen work-plan order, so
  concurrency cannot change which documents are selected or their manifest order.

The verifier and corpus schema remain owned by :mod:`dataset.eval_core`.
"""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Iterator, Sequence

from dataset import config
from dataset import eval_core as core
from dataset.src.bytesource import HttpRangeReader, SourceFile, list_source_files
from dataset.src.records import ParsedRecord, iter_owned_records, record_identity_str, validate_record
from dataset.src.split import is_validation
from dataset.src.workplan import WorkItem, build_work_plan

DEFAULT_SCAN_WORKERS = 8
EVAL_FETCH_CHUNK_BYTES = 32 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class _ValidationCandidate:
    record: ParsedRecord
    scanned_through: int


@dataclass(frozen=True, slots=True)
class _SourceBatch:
    ordinal: int
    filename: str
    scanned_records: int
    candidates: tuple[_ValidationCandidate, ...]


def _scan_workers() -> int:
    raw = os.environ.get("SMALL_LLM_EVAL_SCAN_WORKERS", str(DEFAULT_SCAN_WORKERS))
    try:
        workers = int(raw)
    except ValueError as error:
        raise ValueError("SMALL_LLM_EVAL_SCAN_WORKERS must be an integer") from error
    if not 1 <= workers <= 32:
        raise ValueError("SMALL_LLM_EVAL_SCAN_WORKERS must be in [1, 32]")
    return workers


def _scan_item(
    *,
    ordinal: int,
    item: WorkItem,
    source: SourceFile,
    validation_probability: float,
) -> _SourceBatch:
    reader = HttpRangeReader(source, config.DATASET_REPOSITORY, config.DATASET_REVISION)
    candidates: list[_ValidationCandidate] = []
    scanned = 0
    for record in iter_owned_records(
        item,
        reader,
        fetch_chunk=EVAL_FETCH_CHUNK_BYTES,
    ):
        scanned += 1
        # The split is a pure function of permanent source identity. Checking it
        # before JSON parsing is exactly equivalent to the legacy builder but avoids
        # deserializing the token arrays of the ~99.9% non-validation records.
        if is_validation(
            seed=config.SELECTION_SEED,
            revision=config.DATASET_REVISION,
            filename=item.filename,
            record_start=record.record_start,
            probability=validation_probability,
        ):
            candidates.append(
                _ValidationCandidate(record=record, scanned_through=scanned)
            )
    return _SourceBatch(
        ordinal=ordinal,
        filename=item.filename,
        scanned_records=scanned,
        candidates=tuple(candidates),
    )


def _ordered_validation_batches(
    source_files: Sequence[SourceFile],
    *,
    validation_probability: float,
    max_work_items: int | None = None,
) -> Iterator[_SourceBatch]:
    plan = build_work_plan(
        list(source_files),
        region_bytes=config.REGION_BYTES,
        seed=config.SELECTION_SEED,
        repository=config.DATASET_REPOSITORY,
        revision=config.DATASET_REVISION,
    )
    items = list(plan.work_items)
    if max_work_items is not None:
        items = items[:max_work_items]
    by_name = {source.path: source for source in source_files}
    workers = min(_scan_workers(), max(1, len(items)))

    print(
        "eval_core accelerated scan: "
        f"workers={workers} region_mib={config.REGION_BYTES / (1024 * 1024):.0f} "
        f"fetch_mib={EVAL_FETCH_CHUNK_BYTES / (1024 * 1024):.0f}",
        flush=True,
    )

    if workers == 1:
        for ordinal, item in enumerate(items):
            yield _scan_item(
                ordinal=ordinal,
                item=item,
                source=by_name[item.filename],
                validation_probability=validation_probability,
            )
        return

    executor = ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="eval-core-scan",
    )
    pending: dict[int, Future[_SourceBatch]] = {}
    next_submit = 0

    def submit(ordinal: int) -> None:
        item = items[ordinal]
        pending[ordinal] = executor.submit(
            _scan_item,
            ordinal=ordinal,
            item=item,
            source=by_name[item.filename],
            validation_probability=validation_probability,
        )

    try:
        while next_submit < len(items) and len(pending) < workers:
            submit(next_submit)
            next_submit += 1

        for ordinal in range(len(items)):
            batch = pending.pop(ordinal).result()
            yield batch
            if next_submit < len(items):
                submit(next_submit)
                next_submit += 1
    finally:
        for future in pending.values():
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)


def build_eval_core_accelerated(
    output_dir: Path,
    *,
    max_work_items: int | None = None,
    validation_probability: float = config.VALIDATION_PROBABILITY,
    fast_documents_per_cluster: int = core.FAST_DOCUMENTS_PER_CLUSTER,
    fast_targets_per_cluster: int = core.FAST_TARGETS_PER_CLUSTER,
    full_documents_per_cluster: int = core.FULL_DOCUMENTS_PER_CLUSTER,
    full_targets_per_cluster: int = core.FULL_TARGETS_PER_CLUSTER,
) -> dict[str, object]:
    """Build ``eval_core_v1`` with frozen semantics and an accelerated scan."""
    for name, value in {
        "fast_documents_per_cluster": fast_documents_per_cluster,
        "fast_targets_per_cluster": fast_targets_per_cluster,
        "full_documents_per_cluster": full_documents_per_cluster,
        "full_targets_per_cluster": full_targets_per_cluster,
    }.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if fast_documents_per_cluster > full_documents_per_cluster:
        raise ValueError("fast document floor cannot exceed full")
    if fast_targets_per_cluster > full_targets_per_cluster:
        raise ValueError("fast target floor cannot exceed full")
    if not 0.0 <= validation_probability <= 1.0:
        raise ValueError("validation_probability must be in [0, 1]")

    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to replace existing eval directory: {output_dir}")
    temporary = output_dir.with_name(f".{output_dir.name}.partial-{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)

    fast = core._SuiteWriter(temporary, "fast")
    full = core._SuiteWriter(temporary, "full")
    selected_full: set[str] = set()
    scanned_records = 0
    validation_records = 0
    complete = False

    try:
        sources = list_source_files(config.DATASET_REPOSITORY, config.DATASET_REVISION)
        batches = _ordered_validation_batches(
            sources,
            validation_probability=validation_probability,
            max_work_items=max_work_items,
        )

        for batch in batches:
            scanned_within_batch = 0
            for candidate in batch.candidates:
                # Reconstruct the legacy counter exactly even though the worker has
                # already scanned the rest of this immutable region.
                scanned_records += candidate.scanned_through - scanned_within_batch
                scanned_within_batch = candidate.scanned_through
                record = candidate.record

                result = validate_record(record)
                if (
                    not result.valid
                    or result.cluster_id not in config.ACCEPTED_CLUSTER_IDS
                    or result.tokens is None
                ):
                    continue

                validation_records += 1
                cluster_id = int(result.cluster_id)
                if not core._needs(
                    full,
                    cluster_id,
                    documents=full_documents_per_cluster,
                    targets=full_targets_per_cluster,
                ):
                    if core._complete(
                        full,
                        documents=full_documents_per_cluster,
                        targets=full_targets_per_cluster,
                    ):
                        complete = True
                        break
                    continue

                windows = core.document_windows(result.tokens)
                if not windows:
                    continue
                document_id = record_identity_str(
                    config.DATASET_REVISION,
                    batch.filename,
                    record.record_start,
                )
                if document_id in selected_full:
                    raise RuntimeError(f"source record selected twice: {document_id}")
                selected_full.add(document_id)
                full.add_document(
                    document_id=document_id,
                    cluster_id=cluster_id,
                    filename=batch.filename,
                    record_start=record.record_start,
                    windows=windows,
                )
                if core._needs(
                    fast,
                    cluster_id,
                    documents=fast_documents_per_cluster,
                    targets=fast_targets_per_cluster,
                ):
                    fast.add_document(
                        document_id=document_id,
                        cluster_id=cluster_id,
                        filename=batch.filename,
                        record_start=record.record_start,
                        windows=windows,
                    )

                if validation_records % 1_000 == 0:
                    print(
                        f"validation_docs={validation_records:,} "
                        f"full_targets={full.target_tokens:,} "
                        f"full_docs={len(full.document_ids):,}",
                        flush=True,
                    )

            if complete:
                break

            scanned_records += batch.scanned_records - scanned_within_batch
            if (batch.ordinal + 1) % 8 == 0:
                missing = sum(
                    core._needs(
                        full,
                        cluster,
                        documents=full_documents_per_cluster,
                        targets=full_targets_per_cluster,
                    )
                    for cluster in core.ACCEPTED_CLUSTERS
                )
                print(
                    "eval_core source scan: "
                    f"regions={batch.ordinal + 1:,} "
                    f"scanned_records={scanned_records:,} "
                    f"validation_docs={validation_records:,} "
                    f"full_docs={len(full.document_ids):,} "
                    f"clusters_remaining={missing}",
                    flush=True,
                )

        if not core._complete(
            full,
            documents=full_documents_per_cluster,
            targets=full_targets_per_cluster,
        ):
            missing = {
                str(cluster): full.per_cluster[cluster]
                for cluster in core.ACCEPTED_CLUSTERS
                if core._needs(
                    full,
                    cluster,
                    documents=full_documents_per_cluster,
                    targets=full_targets_per_cluster,
                )
            }
            raise RuntimeError(
                "source scan ended before the frozen full quotas were met: "
                + json.dumps(missing, sort_keys=True)
            )
        if not core._complete(
            fast,
            documents=fast_documents_per_cluster,
            targets=fast_targets_per_cluster,
        ):
            raise RuntimeError("full quotas passed but nested fast quotas did not")
    except Exception:
        fast.close()
        full.close()
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    fast.close()
    full.close()

    manifest: dict[str, object] = {
        "schema_version": core.SCHEMA_VERSION,
        "name": core.EVAL_NAME,
        "dataset": config.DATASET_REPOSITORY,
        "revision": config.DATASET_REVISION,
        "tokenizer": config.TOKENIZER_ID,
        "semantic_vocab_size": config.VOCAB_SIZE,
        "eod_token_id": config.EOD_TOKEN_ID,
        "context_length": core.CONTEXT_LENGTH,
        "stored_tokens_per_sequence": core.STORED_TOKENS,
        "dtype": "uint16-le",
        "split": {
            "seed": config.SELECTION_SEED,
            "hash_version": config.SPLIT_HASH_VERSION,
            "validation_probability": validation_probability,
        },
        "accepted_clusters": list(core.ACCEPTED_CLUSTERS),
        "mixture_source_tokens": {
            str(cluster): core.MIXTURE_SOURCE_TOKENS[cluster]
            for cluster in core.ACCEPTED_CLUSTERS
        },
        "mixture_weights_sha256": core.MIXTURE_WEIGHTS_SHA256,
        "selection_order": "frozen hash-shuffled source work plan",
        "scanned_records": scanned_records,
        "validation_records_seen": validation_records,
        "source_file_count": len(sources),
        "minimums": {
            "fast": {
                "documents_per_cluster": fast_documents_per_cluster,
                "target_tokens_per_cluster": fast_targets_per_cluster,
            },
            "full": {
                "documents_per_cluster": full_documents_per_cluster,
                "target_tokens_per_cluster": full_targets_per_cluster,
            },
        },
        "suites": {"fast": fast.summary(), "full": full.summary()},
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        core.canonical_json_bytes(manifest)
    ).hexdigest()
    manifest_path = temporary / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with manifest_path.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, output_dir)
    return manifest


__all__ = [
    "DEFAULT_SCAN_WORKERS",
    "EVAL_FETCH_CHUNK_BYTES",
    "build_eval_core_accelerated",
]
