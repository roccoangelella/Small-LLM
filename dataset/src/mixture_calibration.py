"""Exact, resumable cluster-token calibration for Nemotron-ClimbMix."""

from __future__ import annotations

import hashlib
import json
import logging
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Mapping

from dataset import config

from .bytesource import RangeReader, SourceFile
from .manifest import sha256_file
from .records import iter_owned_records
from .storage import canonical_json_bytes, read_json, write_json_atomic
from .workplan import WorkItem, WorkPlan

LOGGER = logging.getLogger(__name__)

MIXTURE_SCAN_SCHEMA_VERSION = 1
MIXTURE_PROGRESS_FILENAME = "mixture_progress.json"
MIXTURE_REPORT_FILENAME = "mixture_report.json"
MIXTURE_WEIGHTS_FILENAME = "climbmix_code_free_weights.json"


@dataclass(frozen=True)
class RecordMetadata:
    cluster_id: int
    token_count: int


@dataclass(frozen=True)
class WorkItemMixture:
    index: int
    source_bytes: int
    record_count: int
    cluster_source_tokens: dict[int, int]
    cluster_document_counts: dict[int, int]


def _skip_ws(raw: bytes, index: int) -> int:
    while index < len(raw) and raw[index] in b" \t\r\n":
        index += 1
    return index


def _scan_string_end(raw: bytes, index: int) -> int:
    if index >= len(raw) or raw[index] != ord('"'):
        raise ValueError("expected JSON string")
    index += 1
    escaped = False
    while index < len(raw):
        value = raw[index]
        if escaped:
            escaped = False
        elif value == ord("\\"):
            escaped = True
        elif value == ord('"'):
            return index + 1
        elif value < 0x20:
            raise ValueError("unescaped control byte in JSON string")
        index += 1
    raise ValueError("unterminated JSON string")


def _skip_compound(raw: bytes, index: int) -> int:
    opening = raw[index]
    expected = ord("}") if opening == ord("{") else ord("]")
    stack = [expected]
    index += 1
    while index < len(raw) and stack:
        value = raw[index]
        if value == ord('"'):
            index = _scan_string_end(raw, index)
            continue
        if value == ord("{"):
            stack.append(ord("}"))
        elif value == ord("["):
            stack.append(ord("]"))
        elif value in (ord("}"), ord("]")):
            if value != stack[-1]:
                raise ValueError("mismatched JSON brackets")
            stack.pop()
        index += 1
    if stack:
        raise ValueError("unterminated JSON compound value")
    return index


def _skip_value(raw: bytes, index: int) -> int:
    index = _skip_ws(raw, index)
    if index >= len(raw):
        raise ValueError("missing JSON value")
    value = raw[index]
    if value == ord('"'):
        return _scan_string_end(raw, index)
    if value in (ord("{"), ord("[")):
        return _skip_compound(raw, index)
    end = index
    while end < len(raw) and raw[end] not in b",}":
        end += 1
    if not raw[index:end].strip():
        raise ValueError("empty JSON scalar")
    return end


def _parse_integer(raw: bytes, index: int) -> tuple[int, int]:
    index = _skip_ws(raw, index)
    start = index
    if index < len(raw) and raw[index] == ord("-"):
        index += 1
    digit_start = index
    while index < len(raw) and ord("0") <= raw[index] <= ord("9"):
        index += 1
    if index == digit_start:
        raise ValueError("metadata value is not an integer")
    if index < len(raw) and raw[index] not in b" \t\r\n,}":
        raise ValueError("metadata value is not a plain integer")
    return int(raw[start:index]), index


def extract_record_metadata(raw: bytes) -> RecordMetadata:
    """Extract top-level metadata without constructing the large ``tokens`` list."""

    index = _skip_ws(raw, 0)
    if index >= len(raw) or raw[index] != ord("{"):
        raise ValueError("record is not a JSON object")
    index += 1
    found: dict[str, int] = {}

    while True:
        index = _skip_ws(raw, index)
        if index >= len(raw):
            raise ValueError("unterminated JSON object")
        if raw[index] == ord("}"):
            index += 1
            break
        key_start = index
        key_end = _scan_string_end(raw, index)
        try:
            key = json.loads(raw[key_start:key_end].decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as error:
            raise ValueError("invalid JSON object key") from error
        if not isinstance(key, str):
            raise ValueError("JSON object key is not a string")
        index = _skip_ws(raw, key_end)
        if index >= len(raw) or raw[index] != ord(":"):
            raise ValueError("missing colon after JSON key")
        index = _skip_ws(raw, index + 1)
        if key in {"cluster_id", "token_count"}:
            if key in found:
                raise ValueError(f"duplicate {key}")
            found[key], index = _parse_integer(raw, index)
        else:
            index = _skip_value(raw, index)
        index = _skip_ws(raw, index)
        if index >= len(raw):
            raise ValueError("unterminated JSON object")
        if raw[index] == ord(","):
            index += 1
            continue
        if raw[index] == ord("}"):
            index += 1
            break
        raise ValueError("expected comma or object end")

    if _skip_ws(raw, index) != len(raw):
        raise ValueError("trailing bytes after JSON object")
    if set(found) != {"cluster_id", "token_count"}:
        missing = sorted({"cluster_id", "token_count"} - set(found))
        raise ValueError(f"record is missing metadata fields: {missing}")
    cluster_id = found["cluster_id"]
    token_count = found["token_count"]
    if cluster_id not in config.ALL_CLUSTER_IDS:
        raise ValueError(f"cluster_id {cluster_id} is outside 1..20")
    if token_count <= 0:
        raise ValueError("token_count must be positive")
    return RecordMetadata(cluster_id, token_count)


def scan_work_item(
    item: WorkItem,
    source_file: SourceFile,
    reader_factory: Callable[[SourceFile], RangeReader],
) -> WorkItemMixture:
    reader = reader_factory(source_file)
    if reader.file_size() != source_file.size:
        raise RuntimeError(
            f"source size changed for {source_file.path}: "
            f"expected {source_file.size}, got {reader.file_size()}"
        )
    token_counts = {cluster: 0 for cluster in config.ALL_CLUSTER_IDS}
    document_counts = {cluster: 0 for cluster in config.ALL_CLUSTER_IDS}
    records = 0
    for record in iter_owned_records(item, reader):
        try:
            metadata = extract_record_metadata(record.raw)
        except ValueError as error:
            raise ValueError(
                f"invalid mixture metadata at {item.filename}:{record.record_start}: {error}"
            ) from error
        token_counts[metadata.cluster_id] += metadata.token_count
        document_counts[metadata.cluster_id] += 1
        records += 1
    return WorkItemMixture(
        index=item.index,
        source_bytes=item.range_end - item.range_start,
        record_count=records,
        cluster_source_tokens=token_counts,
        cluster_document_counts=document_counts,
    )


def _ordered_parallel_results(
    plan: WorkPlan,
    *,
    start_index: int,
    reader_factory: Callable[[SourceFile], RangeReader],
    workers: int,
    max_in_flight: int,
) -> Iterator[WorkItemMixture]:
    if workers <= 0 or max_in_flight <= 0:
        raise ValueError("workers and max_in_flight must be positive")
    source_by_name = {source.path: source for source in plan.source_files}
    items = plan.work_items[start_index:]
    pending: dict[int, Future[WorkItemMixture]] = {}
    submit_index = 0
    yield_index = start_index

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="mixture-scan") as pool:
        while submit_index < len(items) or pending:
            while submit_index < len(items) and len(pending) < max_in_flight:
                item = items[submit_index]
                source = source_by_name.get(item.filename)
                if source is None:
                    raise RuntimeError(f"work item references unknown source {item.filename}")
                pending[item.index] = pool.submit(scan_work_item, item, source, reader_factory)
                submit_index += 1
            future = pending.pop(yield_index)
            result = future.result()
            if result.index != yield_index:
                raise RuntimeError("mixture scan result order changed")
            yield result
            yield_index += 1


def _empty_cluster_map() -> dict[str, int]:
    return {str(cluster): 0 for cluster in sorted(config.ALL_CLUSTER_IDS)}


def _initial_state(plan: WorkPlan) -> dict[str, object]:
    return {
        "schema_version": MIXTURE_SCAN_SCHEMA_VERSION,
        "dataset": plan.dataset,
        "revision": plan.revision,
        "source_glob": plan.source_glob,
        "work_plan_hash": plan.hash,
        "next_work_item_index": 0,
        "completed_work_items": 0,
        "source_bytes_covered": 0,
        "record_count": 0,
        "cluster_source_tokens": _empty_cluster_map(),
        "cluster_document_counts": _empty_cluster_map(),
        "complete": False,
    }


def _validate_state(raw: object, plan: WorkPlan) -> dict[str, object]:
    if not isinstance(raw, Mapping):
        raise ValueError("mixture progress must be a JSON object")
    state = dict(raw)
    expected = {
        "schema_version": MIXTURE_SCAN_SCHEMA_VERSION,
        "dataset": plan.dataset,
        "revision": plan.revision,
        "source_glob": plan.source_glob,
        "work_plan_hash": plan.hash,
    }
    for key, value in expected.items():
        if state.get(key) != value:
            raise ValueError(f"mixture progress {key} does not match the pinned work plan")
    next_index = state.get("next_work_item_index")
    if isinstance(next_index, bool) or not isinstance(next_index, int):
        raise ValueError("mixture progress has an invalid next_work_item_index")
    if not 0 <= next_index <= len(plan.work_items):
        raise ValueError("mixture progress next_work_item_index is outside the work plan")
    if state.get("completed_work_items") != next_index:
        raise ValueError("mixture progress completed_work_items is inconsistent")
    expected_bytes = sum(
        item.range_end - item.range_start for item in plan.work_items[:next_index]
    )
    if state.get("source_bytes_covered") != expected_bytes:
        raise ValueError("mixture progress source byte coverage is inconsistent")
    for field in ("cluster_source_tokens", "cluster_document_counts"):
        value = state.get(field)
        if not isinstance(value, Mapping) or set(value) != set(_empty_cluster_map()):
            raise ValueError(f"mixture progress has an invalid {field}")
        if any(
            isinstance(count, bool) or not isinstance(count, int) or count < 0
            for count in value.values()
        ):
            raise ValueError(f"mixture progress has negative or non-integer {field}")
    return state


def _report_hash(report: Mapping[str, object]) -> str:
    return hashlib.sha256(
        canonical_json_bytes(dict(report), exclude_keys=("report_sha256",))
    ).hexdigest()


def _load_completed_report(output_dir: Path, state: Mapping[str, object]) -> dict[str, object]:
    report = read_json(output_dir / MIXTURE_REPORT_FILENAME)
    if not isinstance(report, dict):
        raise ValueError("completed mixture report must be a JSON object")
    if report.get("report_sha256") != _report_hash(report):
        raise ValueError("completed mixture report hash mismatch")
    weights_path = output_dir / MIXTURE_WEIGHTS_FILENAME
    if sha256_file(weights_path) != report.get("weights_sha256"):
        raise ValueError("completed mixture weights hash mismatch")
    if report.get("work_plan_hash") != state.get("work_plan_hash"):
        raise ValueError("completed mixture report belongs to a different work plan")
    return report


def _finish(output_dir: Path, plan: WorkPlan, state: dict[str, object]) -> dict[str, object]:
    if int(state["next_work_item_index"]) != len(plan.work_items):
        raise RuntimeError("cannot finish an incomplete mixture scan")
    total_source_bytes = sum(source.size for source in plan.source_files)
    if int(state["source_bytes_covered"]) != total_source_bytes:
        raise RuntimeError("mixture scan did not cover the exact pinned source byte size")

    raw_tokens = dict(state["cluster_source_tokens"])
    token_counts = {int(cluster): int(count) for cluster, count in raw_tokens.items()}
    missing = [
        cluster for cluster in sorted(config.ALL_CLUSTER_IDS) if token_counts[cluster] <= 0
    ]
    if missing:
        raise RuntimeError(f"complete source scan found no tokens for clusters {missing}")

    weights = {
        str(cluster): token_counts[cluster]
        for cluster in sorted(config.ACCEPTED_CLUSTER_IDS)
    }
    weights_path = output_dir / MIXTURE_WEIGHTS_FILENAME
    write_json_atomic(weights_path, weights)
    weights_sha256 = sha256_file(weights_path)
    all_tokens = sum(token_counts.values())
    accepted_tokens = sum(token_counts[cluster] for cluster in config.ACCEPTED_CLUSTER_IDS)

    report: dict[str, object] = {
        "schema_version": MIXTURE_SCAN_SCHEMA_VERSION,
        "complete": True,
        "dataset": plan.dataset,
        "revision": plan.revision,
        "source_glob": plan.source_glob,
        "work_plan_hash": plan.hash,
        "source_files": [
            {"path": source.path, "size": source.size} for source in plan.source_files
        ],
        "source_bytes_scanned": total_source_bytes,
        "record_count": int(state["record_count"]),
        "all_cluster_source_tokens": {
            str(cluster): token_counts[cluster] for cluster in sorted(token_counts)
        },
        "all_cluster_document_counts": dict(state["cluster_document_counts"]),
        "all_source_tokens": all_tokens,
        "accepted_cluster_ids": sorted(config.ACCEPTED_CLUSTER_IDS),
        "excluded_cluster_ids": sorted(config.EXCLUDED_CLUSTER_IDS),
        "accepted_source_tokens": accepted_tokens,
        "conditioning_rule": (
            "Production weights are the exact released-corpus token totals for retained "
            "clusters, conditioned on excluding cluster 11."
        ),
        "weights_file": MIXTURE_WEIGHTS_FILENAME,
        "weights_sha256": weights_sha256,
    }
    report["report_sha256"] = _report_hash(report)
    write_json_atomic(output_dir / MIXTURE_REPORT_FILENAME, report)
    state["complete"] = True
    state["weights_sha256"] = weights_sha256
    state["report_sha256"] = report["report_sha256"]
    write_json_atomic(output_dir / MIXTURE_PROGRESS_FILENAME, state)
    return report


def scan_mixture(
    output_dir: Path | str,
    plan: WorkPlan,
    reader_factory: Callable[[SourceFile], RangeReader],
    *,
    resume: bool = False,
    workers: int = 8,
    max_in_flight: int = 16,
    checkpoint_every_work_items: int = 4,
    simulate_crash_after_work_items: int | None = None,
) -> dict[str, object]:
    """Scan every pinned source record and emit exact code-free scheduler weights."""

    if checkpoint_every_work_items <= 0:
        raise ValueError("checkpoint_every_work_items must be positive")
    if simulate_crash_after_work_items is not None and simulate_crash_after_work_items <= 0:
        raise ValueError("simulate_crash_after_work_items must be positive")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / MIXTURE_PROGRESS_FILENAME

    if resume:
        state = _validate_state(read_json(progress_path), plan)
        if state.get("complete") is True:
            return _load_completed_report(output_dir, state)
    else:
        for path in (
            progress_path,
            output_dir / MIXTURE_REPORT_FILENAME,
            output_dir / MIXTURE_WEIGHTS_FILENAME,
        ):
            if path.exists():
                raise FileExistsError(
                    f"mixture calibration state already exists at {path}; "
                    "use --resume or a new directory"
                )
        state = _initial_state(plan)
        write_json_atomic(progress_path, state)

    start_index = int(state["next_work_item_index"])
    token_counts = dict(state["cluster_source_tokens"])
    document_counts = dict(state["cluster_document_counts"])

    for result in _ordered_parallel_results(
        plan,
        start_index=start_index,
        reader_factory=reader_factory,
        workers=workers,
        max_in_flight=max_in_flight,
    ):
        for cluster in config.ALL_CLUSTER_IDS:
            key = str(cluster)
            token_counts[key] = int(token_counts[key]) + result.cluster_source_tokens[cluster]
            document_counts[key] = (
                int(document_counts[key]) + result.cluster_document_counts[cluster]
            )
        state["cluster_source_tokens"] = token_counts
        state["cluster_document_counts"] = document_counts
        state["record_count"] = int(state["record_count"]) + result.record_count
        state["source_bytes_covered"] = (
            int(state["source_bytes_covered"]) + result.source_bytes
        )
        state["next_work_item_index"] = result.index + 1
        state["completed_work_items"] = result.index + 1

        completed = result.index + 1
        if completed % checkpoint_every_work_items == 0:
            write_json_atomic(progress_path, state)
            LOGGER.info(
                "mixture calibration: %d/%d work items, %d records, %.1f GiB covered",
                completed,
                len(plan.work_items),
                int(state["record_count"]),
                int(state["source_bytes_covered"]) / (1024**3),
            )
        if simulate_crash_after_work_items == completed:
            write_json_atomic(progress_path, state)
            raise RuntimeError("simulated mixture calibration interruption")

    write_json_atomic(progress_path, state)
    return _finish(output_dir, plan, state)
