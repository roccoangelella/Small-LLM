"""Accelerated production builder for the frozen ``eval_core_v1`` corpus.

This module preserves the exact eval-core selection policy and output ordering while
making the remote source scan practical:

* record boundaries are scanned from raw bytes and the frozen validation identity is
  hashed before a JSONL line is materialized, so ~99.9% of source records are never
  copied into Python record objects or JSON/token-deserialized;
* immutable 256 MiB source regions are scanned concurrently using conservative
  8 MiB HTTP range reads, which proved substantially more reliable than large reads;
* more work is queued than there are workers, so one slow early region does not leave
  the remaining workers idle;
* completed region results are consumed strictly in frozen work-plan order, so
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
import threading
from typing import Iterator, Sequence

from dataset import config
from dataset import eval_core as core
from dataset.src.bytesource import HttpRangeReader, SourceFile, list_source_files
from dataset.src.records import ParsedRecord, record_identity_str, validate_record
from dataset.src.split import is_validation
from dataset.src.workplan import WorkItem, build_work_plan

DEFAULT_SCAN_WORKERS = 4
EVAL_FETCH_CHUNK_BYTES = 8 * 1024 * 1024
PREFETCH_PER_WORKER = 4

_PROGRESS_LOCK = threading.Lock()
_PROGRESS: dict[str, int] = {
    "total_regions": 0,
    "regions_finished": 0,
    "regions_committed": 0,
    "downloaded_bytes": 0,
    "records_scanned": 0,
}


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


def _reset_scan_progress(total_regions: int) -> None:
    with _PROGRESS_LOCK:
        _PROGRESS.update(
            total_regions=total_regions,
            regions_finished=0,
            regions_committed=0,
            downloaded_bytes=0,
            records_scanned=0,
        )


def _progress_add(
    *,
    regions_finished: int = 0,
    regions_committed: int = 0,
    downloaded_bytes: int = 0,
    records_scanned: int = 0,
) -> None:
    with _PROGRESS_LOCK:
        _PROGRESS["regions_finished"] += regions_finished
        _PROGRESS["regions_committed"] += regions_committed
        _PROGRESS["downloaded_bytes"] += downloaded_bytes
        _PROGRESS["records_scanned"] += records_scanned


def scan_progress_snapshot() -> dict[str, int]:
    """Return a thread-safe snapshot for user-facing build heartbeats."""
    with _PROGRESS_LOCK:
        return dict(_PROGRESS)


def _scan_workers() -> int:
    raw = os.environ.get("SMALL_LLM_EVAL_SCAN_WORKERS", str(DEFAULT_SCAN_WORKERS))
    try:
        workers = int(raw)
    except ValueError as error:
        raise ValueError("SMALL_LLM_EVAL_SCAN_WORKERS must be an integer") from error
    if not 1 <= workers <= 16:
        raise ValueError("SMALL_LLM_EVAL_SCAN_WORKERS must be in [1, 16]")
    return workers


def _read_counted(reader: HttpRangeReader, offset: int, length: int) -> bytes:
    data = reader.read_range(offset, length)
    _progress_add(downloaded_bytes=len(data))
    return data


def _record_floor(reader: HttpRangeReader, start: int) -> int:
    """Return the start of the JSONL record containing ``start``."""
    pos = start
    while pos > 0:
        lo = max(0, pos - config.BOUNDARY_SCAN_CHUNK_BYTES)
        data = _read_counted(reader, lo, pos - lo)
        newline = data.rfind(b"\n")
        if newline != -1:
            return lo + newline + 1
        if lo == 0:
            return 0
        pos = lo
    return 0


def _is_candidate(filename: str, record_start: int, probability: float) -> bool:
    return is_validation(
        seed=config.SELECTION_SEED,
        revision=config.DATASET_REVISION,
        filename=filename,
        record_start=record_start,
        probability=probability,
    )


def _scan_item(
    *,
    ordinal: int,
    item: WorkItem,
    source: SourceFile,
    validation_probability: float,
    stop_event: threading.Event | None = None,
) -> _SourceBatch:
    """Scan one frozen work-plan region without materializing rejected records."""
    reader = HttpRangeReader(source, config.DATASET_REPOSITORY, config.DATASET_REVISION)
    start, end = item.range_start, item.range_end
    file_size = reader.file_size()
    floor = 0 if start == 0 else _record_floor(reader, start)

    cursor = floor
    line_start = floor
    scanned = 0
    reported_scanned = 0
    candidates: list[_ValidationCandidate] = []
    candidate_parts: list[bytes] | None = (
        []
        if start <= line_start < end
        and _is_candidate(item.filename, line_start, validation_probability)
        else None
    )

    def report_records() -> None:
        nonlocal reported_scanned
        delta = scanned - reported_scanned
        if delta:
            _progress_add(records_scanned=delta)
            reported_scanned = scanned

    def finish() -> _SourceBatch:
        report_records()
        _progress_add(regions_finished=1)
        return _SourceBatch(
            ordinal=ordinal,
            filename=item.filename,
            scanned_records=scanned,
            candidates=tuple(candidates),
        )

    while cursor < file_size:
        if stop_event is not None and stop_event.is_set():
            return finish()
        length = min(EVAL_FETCH_CHUNK_BYTES, file_size - cursor)
        chunk = _read_counted(reader, cursor, length)
        if not chunk:
            break

        pos = 0
        while True:
            newline = chunk.find(b"\n", pos)
            if newline == -1:
                if candidate_parts is not None:
                    candidate_parts.append(chunk[pos:])
                break

            if candidate_parts is not None:
                candidate_parts.append(chunk[pos:newline])

            if start <= line_start < end:
                scanned += 1
                if candidate_parts is not None:
                    raw = b"".join(candidate_parts)
                    if raw.endswith(b"\r"):
                        raw = raw[:-1]
                    candidates.append(
                        _ValidationCandidate(
                            record=ParsedRecord(record_start=line_start, raw=raw),
                            scanned_through=scanned,
                        )
                    )

            line_start = cursor + newline + 1
            pos = newline + 1
            if line_start >= end:
                return finish()
            candidate_parts = (
                []
                if line_start >= start
                and _is_candidate(item.filename, line_start, validation_probability)
                else None
            )

        cursor += len(chunk)
        report_records()

    # EOF may terminate the final JSONL record without a trailing newline.
    if start <= line_start < end:
        scanned += 1
        if candidate_parts is not None:
            raw = b"".join(candidate_parts)
            if raw.endswith(b"\r"):
                raw = raw[:-1]
            candidates.append(
                _ValidationCandidate(
                    record=ParsedRecord(record_start=line_start, raw=raw),
                    scanned_through=scanned,
                )
            )
    return finish()


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
    _reset_scan_progress(len(items))
    by_name = {source.path: source for source in source_files}
    workers = min(_scan_workers(), max(1, len(items)))
    queue_limit = min(len(items), max(workers, workers * PREFETCH_PER_WORKER))

    print(
        "eval_core accelerated scan: "
        f"workers={workers} queued={queue_limit} "
        f"region_mib={config.REGION_BYTES / (1024 * 1024):.0f} "
        f"fetch_mib={EVAL_FETCH_CHUNK_BYTES / (1024 * 1024):.0f}",
        flush=True,
    )

    stop_event = threading.Event()
    if workers == 1:
        try:
            for ordinal, item in enumerate(items):
                batch = _scan_item(
                    ordinal=ordinal,
                    item=item,
                    source=by_name[item.filename],
                    validation_probability=validation_probability,
                    stop_event=stop_event,
                )
                _progress_add(regions_committed=1)
                yield batch
        finally:
            stop_event.set()
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
            stop_event=stop_event,
        )

    try:
        while next_submit < len(items) and len(pending) < queue_limit:
            submit(next_submit)
            next_submit += 1

        for ordinal in range(len(items)):
            batch = pending.pop(ordinal).result()
            _progress_add(regions_committed=1)
            yield batch
            while next_submit < len(items) and len(pending) < queue_limit:
                submit(next_submit)
                next_submit += 1
    finally:
        stop_event.set()
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
    batches: Iterator[_SourceBatch] | None = None
    sources: Sequence[SourceFile] = ()

    try:
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
                    # Reconstruct the legacy counter exactly even though a worker may
                    # already have scanned the remainder of this immutable region.
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
        finally:
            close = getattr(batches, "close", None)
            if close is not None:
                close()

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
    "PREFETCH_PER_WORKER",
    "build_eval_core_accelerated",
    "scan_progress_snapshot",
]
