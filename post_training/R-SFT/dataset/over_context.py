#!/usr/bin/env python3
"""Dataset-agnostic GemRouter compression for over-context R-SFT examples.

Sources are responsible for semantic filtering and for emitting canonical
over-context candidates. This module owns the shared curation, Gemini-only
GemRouter transport gate, fidelity-first compression prompt, retries/split
recovery, exact 2,048-token revalidation, resumability, and adapted JSONL
finalization.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import ModuleType
from typing import Any, Iterator, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent
RSFT_DIR = HERE.parent
REPO = RSFT_DIR.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _load(name: str, path: Path) -> ModuleType:
    module_name = f"small_llm_rsft_over_context_{name}"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


common = _load("common", HERE / "common.py")
transport = _load("transport", RSFT_DIR / "dataset.py")

MAX_BATCH_SIZE = 4
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_RETRY_DELAY_SECONDS = 2.0
DEFAULT_TIMEOUT_SECONDS = 190.0
CURATION_SCHEMA = "small-llm-rsft-manual-curation-v1"
LEGACY_CURATION_SCHEMA = "small-llm-superior-manual-curation-v1"
CURATION_DECISIONS = {"keep", "exclude_math", "exclude_code", "exclude_safety"}
KEEP_MANIFEST_SCHEMA = "small-llm-rsft-overcontext-keep-v1"
ATTEMPT_SCHEMA = "small-llm-rsft-overcontext-attempt-v1"
BATCH_SCHEMA = "small-llm-rsft-overcontext-batch-v1"
ADAPTED_SCHEMA = "small-llm-rsft-overcontext-adapted-v1"

SIMPLIFICATION_SYSTEM_PROMPT = """You are a fidelity-first curriculum compressor for reasoning-SFT examples that must fit a ~100M parameter model with a 2,048-token context window.

Process every input item independently and return one rewritten item for each input id.

Priority order:
1. Preserve the original task type, domain, conclusion, named entities, factual relationships, constraints, and numerical givens whenever they can fit.
2. First shorten by removing repetition, digressions, verbose exposition, redundant examples, and unnecessary formatting. Make the reasoning explicit, linear, and easy to imitate.
3. If the ORIGINAL REQUIRED OUTPUT is itself too large for a compact example, make the smallest scope reduction necessary: shorten source material, reduce repeated list items/subparts, or narrow the requested deliverable while preserving the same instruction-following skill and core reasoning pattern. The answer must then be correct for the rewritten problem.
4. Do NOT change scientific constants, numerical values, entities, or conclusions merely to make an example look simpler. Change them only if unavoidable for a scope-reduced analogous task, and prefer deleting irrelevant context instead.
5. Never invent unsupported facts. Keep each problem self-contained and solvable.
6. Target <= 160 words for problem, <= 300 words for reasoning, <= 160 words for answer, preferably <= 600 words total. Concision is more important than filling the budget.
7. Output strict RFC 8259 JSON only: one JSON array, same item order, objects with exactly id, problem, reasoning, answer. Use valid JSON escaping. Prefer plain-text math notation instead of LaTeX/backslashes. No Markdown code fence and no commentary outside the JSON."""

STRICT_RECOVERY_SUFFIX = """

RETRY CORRECTION — the previous fidelity-first rewrite did not pass the local 2,048-token training-context validator. Keep the same task type, essential facts, constraints, and answer semantics, but compress materially harder. Prefer <= 100 words for problem, <= 180 words for reasoning, and <= 100 words for answer. Remove examples, repetition, framing, and nonessential source material first. If the original required output length is itself the blocker, make the smallest scope reduction allowed by the base policy and make the rewritten answer correct for that reduced task. Return strict JSON only and do not add commentary."""


def _row_text(row: Mapping[str, Any], field: str, *, index: int) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"candidate {index} field {field!r} must be non-empty text")
    return value.strip()


def build_messages(
    rows: Sequence[Mapping[str, Any]], *, strict: bool = False
) -> tuple[dict[str, str], dict[str, str]]:
    if not rows or len(rows) > MAX_BATCH_SIZE:
        raise ValueError(f"batch size must be between 1 and {MAX_BATCH_SIZE}")
    payload: list[dict[str, str]] = []
    ids: set[str] = set()
    for index, row in enumerate(rows):
        item_id = _row_text(row, "id", index=index)
        if item_id in ids:
            raise ValueError(f"duplicate candidate id {item_id!r}")
        ids.add(item_id)
        item = {
            "id": item_id,
            "problem": _row_text(row, "problem", index=index),
            "reasoning": _row_text(row, "reasoning", index=index),
            "answer": _row_text(row, "answer", index=index),
        }
        skill = row.get("skill")
        if isinstance(skill, str) and skill.strip():
            item["skill"] = skill.strip()
        payload.append(item)
    system = SIMPLIFICATION_SYSTEM_PROMPT + (STRICT_RECOVERY_SUFFIX if strict else "")
    return (
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        },
    )


def parse_response(text: str, *, expected_ids: Sequence[str]) -> tuple[dict[str, str], ...]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("compression response must be non-empty text")
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError("compression response must be strict JSON") from error
    if not isinstance(raw, list) or len(raw) != len(expected_ids):
        raise ValueError("compression response record count does not match request")
    expected_fields = {"id", "problem", "reasoning", "answer"}
    result: list[dict[str, str]] = []
    for index, (item, expected_id) in enumerate(zip(raw, expected_ids)):
        if not isinstance(item, Mapping) or set(item) != expected_fields:
            raise ValueError(
                f"compression item {index} must contain exactly {sorted(expected_fields)}"
            )
        normalized = {
            field: _row_text(item, field, index=index) for field in expected_fields
        }
        if normalized["id"] != expected_id:
            raise ValueError(
                f"compression item {index} id {normalized['id']!r} != {expected_id!r}"
            )
        result.append(normalized)
    return tuple(result)


def validate_rewrite(
    candidate: Mapping[str, Any], rewritten: Mapping[str, str]
) -> dict[str, Any]:
    common.validate_reasoning_text(rewritten)
    serialized = common.atomic_rsft_serialized_tokens(
        problem=rewritten["problem"],
        reasoning=rewritten["reasoning"],
        answer=rewritten["answer"],
    )
    if serialized > common.CONTEXT_LENGTH:
        raise ValueError(
            f"rewrite {rewritten['id']!r} remains over context: {serialized} tokens"
        )
    if str(candidate.get("id")) != rewritten["id"]:
        raise ValueError("rewrite id drifted")
    preserved = {
        key: value
        for key, value in candidate.items()
        if key
        not in {
            "schema",
            "problem",
            "reasoning",
            "answer",
            "original_serialized_tokens",
            "serialized_token_count",
            "target_token_count",
        }
    }
    return {
        **preserved,
        "schema": common.FIT_SCHEMA,
        "problem": rewritten["problem"],
        "reasoning": rewritten["reasoning"],
        "answer": rewritten["answer"],
        "serialized_token_count": serialized,
        "target_token_count": common.assistant_target_tokens(
            reasoning=rewritten["reasoning"], answer=rewritten["answer"]
        ),
        "adaptation": "gemrouter-variant-d-fidelity-first",
    }


def read_curation(path: Path | str) -> dict[str, dict[str, str]]:
    decisions: dict[str, dict[str, str]] = {}
    for line_number, row in enumerate(common.read_jsonl(path), start=1):
        if row.get("schema") not in {CURATION_SCHEMA, LEGACY_CURATION_SCHEMA}:
            raise RuntimeError(f"curation row {line_number} has the wrong schema")
        row_id = row.get("id")
        decision = row.get("decision")
        reason = row.get("reason")
        if not isinstance(row_id, str) or not row_id.strip():
            raise RuntimeError(f"curation row {line_number} has no id")
        if decision not in CURATION_DECISIONS:
            raise RuntimeError(f"curation row {line_number} has invalid decision {decision!r}")
        if not isinstance(reason, str) or not reason.strip():
            raise RuntimeError(f"curation row {line_number} has no reason")
        if row_id in decisions:
            raise RuntimeError(f"duplicate curation id {row_id!r}")
        decisions[row_id] = {"decision": str(decision), "reason": reason.strip()}
    return decisions


def prepare_keep(
    *,
    candidates_jsonl: Path,
    curation_jsonl: Path,
    work_dir: Path,
    batch_size: int = MAX_BATCH_SIZE,
) -> dict[str, object]:
    if not 1 <= batch_size <= MAX_BATCH_SIZE:
        raise ValueError(f"batch_size must be in [1, {MAX_BATCH_SIZE}]")
    rows = list(common.read_jsonl(candidates_jsonl))
    decisions = read_curation(curation_jsonl)
    ids = [str(row.get("id")) for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("over-context candidate IDs are not unique")
    missing = [row_id for row_id in ids if row_id not in decisions]
    extra = sorted(set(decisions).difference(ids))
    if missing or extra:
        raise RuntimeError(
            f"curation must cover every candidate exactly once: missing={len(missing)} extra={len(extra)}"
        )
    keep = [row for row in rows if decisions[str(row["id"])]["decision"] == "keep"]
    root = Path(work_dir)
    root.mkdir(parents=True, exist_ok=True)
    keep_path = root / "keep.jsonl"
    manifest_path = root / "keep.manifest.json"
    if keep_path.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to replace existing adaptation state {root}")
    common.write_jsonl(keep_path, keep)
    counts = Counter(decision["decision"] for decision in decisions.values())
    manifest = {
        "schema": KEEP_MANIFEST_SCHEMA,
        "prompt_sha256": common.sha256_text(SIMPLIFICATION_SYSTEM_PROMPT),
        "candidates_sha256": common.sha256_path(candidates_jsonl),
        "curation_sha256": common.sha256_path(curation_jsonl),
        "curation_counts": dict(sorted(counts.items())),
        "batch_size": batch_size,
        "keep_records": len(keep),
        "keep_sha256": common.sha256_path(keep_path),
    }
    common.atomic_json(manifest_path, manifest)
    return {**manifest, "manifest": str(manifest_path), "keep_jsonl": str(keep_path)}


def _health_url(endpoint: str) -> str:
    parsed = urlsplit(endpoint)
    path = parsed.path.rstrip("/")
    for suffix in ("/v1/chat/completions", "/chat/completions"):
        if path.endswith(suffix):
            path = path[: -len(suffix)]
            break
    return urlunsplit((parsed.scheme, parsed.netloc, (path or "") + "/health", "", ""))


def assert_gemini_only_health() -> dict[str, Any]:
    endpoint = transport.resolve_endpoint()
    api_key = transport.resolve_api_key()
    request = Request(
        _health_url(endpoint),
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )
    try:
        with urlopen(request, timeout=30.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as error:  # noqa: BLE001 - fail closed before teacher traffic
        raise RuntimeError("GemRouter health gate failed before teacher traffic") from error
    if not isinstance(payload, Mapping):
        raise RuntimeError("GemRouter health response is not an object")
    config: Mapping[str, Any] = payload
    if isinstance(payload.get("config"), Mapping):
        config = payload["config"]
    backend_order = config.get("backendOrder")
    fallback = config.get("fallbackEnabled")
    if backend_order != ["gemini-api"] or fallback is not False:
        raise RuntimeError(
            "GemRouter must report backendOrder=['gemini-api'] and fallbackEnabled=false"
        )
    return dict(payload)


def _read_keep_manifest(root: Path) -> dict[str, Any]:
    path = root / "keep.manifest.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as error:
        raise RuntimeError(f"invalid keep manifest: {path}") from error
    if not isinstance(payload, dict) or payload.get("schema") != KEEP_MANIFEST_SCHEMA:
        raise RuntimeError("wrong keep manifest schema")
    keep = root / "keep.jsonl"
    if payload.get("keep_sha256") != common.sha256_path(keep):
        raise RuntimeError("keep JSONL hash drifted")
    if payload.get("prompt_sha256") != common.sha256_text(SIMPLIFICATION_SYSTEM_PROMPT):
        raise RuntimeError("compression prompt drifted")
    return payload


def _batches(root: Path) -> Iterator[tuple[int, list[dict[str, Any]]]]:
    manifest = _read_keep_manifest(root)
    size = int(manifest["batch_size"])
    current: list[dict[str, Any]] = []
    index = 1
    for row in common.read_jsonl(root / "keep.jsonl"):
        current.append(row)
        if len(current) == size:
            yield index, current
            current = []
            index += 1
    if current:
        yield index, current


def _accepted_path(root: Path, batch_index: int) -> Path:
    return root / "batches" / f"batch-{batch_index:05d}.json"


def _attempt_path(root: Path, batch_index: int, label: str, attempt: int) -> Path:
    return root / "attempts" / f"batch-{batch_index:05d}-{label}-attempt-{attempt:02d}.json"


def _adapt_part(
    *,
    client: Any,
    root: Path,
    batch_index: int,
    label: str,
    rows: Sequence[Mapping[str, Any]],
    max_attempts: int,
    retry_delay_seconds: float,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    ids = [str(row["id"]) for row in rows]
    usage: Counter[str] = Counter()
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        response_text: str | None = None
        try:
            messages = build_messages(rows, strict=attempt > 1)
            response = client.complete(messages)
            response_text = response.content
            parsed = parse_response(response_text, expected_ids=ids)
            accepted = [
                validate_rewrite(candidate, rewritten)
                for candidate, rewritten in zip(rows, parsed)
            ]
            if isinstance(response.usage, Mapping):
                for key, value in response.usage.items():
                    if isinstance(value, int) and not isinstance(value, bool):
                        usage[str(key)] += value
            common.atomic_json(
                _attempt_path(root, batch_index, label, attempt),
                {
                    "schema": ATTEMPT_SCHEMA,
                    "status": "accepted",
                    "batch_index": batch_index,
                    "part": label,
                    "attempt": attempt,
                    "ids": ids,
                    "records": accepted,
                    "model": response.model,
                    "finish_reason": response.finish_reason,
                    "usage": dict(usage),
                },
            )
            return accepted, usage
        except Exception as error:  # noqa: BLE001 - persisted retry boundary
            last_error = error
            payload: dict[str, object] = {
                "schema": ATTEMPT_SCHEMA,
                "status": "rejected",
                "batch_index": batch_index,
                "part": label,
                "attempt": attempt,
                "ids": ids,
                "error_type": type(error).__name__,
                "error": str(error),
            }
            if response_text is not None:
                payload["response_text"] = response_text
            common.atomic_json(_attempt_path(root, batch_index, label, attempt), payload)
            if attempt < max_attempts and retry_delay_seconds:
                time.sleep(retry_delay_seconds)
    raise RuntimeError(
        f"GemRouter compression failed for batch {batch_index}/{label}"
    ) from last_error


def _adapt_recursive(
    *,
    client: Any,
    root: Path,
    batch_index: int,
    label: str,
    rows: Sequence[Mapping[str, Any]],
    max_attempts: int,
    retry_delay_seconds: float,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    try:
        return _adapt_part(
            client=client,
            root=root,
            batch_index=batch_index,
            label=label,
            rows=rows,
            max_attempts=max_attempts,
            retry_delay_seconds=retry_delay_seconds,
        )
    except Exception:
        if len(rows) == 1:
            raise
        midpoint = len(rows) // 2
        left, left_usage = _adapt_recursive(
            client=client,
            root=root,
            batch_index=batch_index,
            label=f"{label}a",
            rows=rows[:midpoint],
            max_attempts=max_attempts,
            retry_delay_seconds=retry_delay_seconds,
        )
        right, right_usage = _adapt_recursive(
            client=client,
            root=root,
            batch_index=batch_index,
            label=f"{label}b",
            rows=rows[midpoint:],
            max_attempts=max_attempts,
            retry_delay_seconds=retry_delay_seconds,
        )
        left_usage.update(right_usage)
        return [*left, *right], left_usage


def _process_batch(
    root: Path,
    *,
    batch_index: int,
    rows: Sequence[Mapping[str, Any]],
    max_attempts: int,
    retry_delay_seconds: float,
) -> dict[str, object]:
    accepted_path = _accepted_path(root, batch_index)
    if accepted_path.is_file():
        payload = json.loads(accepted_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping) or payload.get("schema") != BATCH_SCHEMA:
            raise RuntimeError(f"accepted batch {batch_index} is malformed")
        if payload.get("ids") != [str(row["id"]) for row in rows]:
            raise RuntimeError(f"accepted batch {batch_index} identity drifted")
        return {"batch_index": batch_index, "records": len(rows), "resumed": True}
    client = transport.GeminiDistillationClient(timeout_seconds=DEFAULT_TIMEOUT_SECONDS)
    records, usage = _adapt_recursive(
        client=client,
        root=root,
        batch_index=batch_index,
        label="root",
        rows=rows,
        max_attempts=max_attempts,
        retry_delay_seconds=retry_delay_seconds,
    )
    payload = {
        "schema": BATCH_SCHEMA,
        "batch_index": batch_index,
        "ids": [str(row["id"]) for row in rows],
        "records": records,
        "prompt_sha256": common.sha256_text(SIMPLIFICATION_SYSTEM_PROMPT),
        "usage": dict(sorted(usage.items())),
    }
    common.atomic_json(accepted_path, payload)
    return {"batch_index": batch_index, "records": len(rows), "resumed": False}


def adapt_wave(
    work_dir: Path | str,
    *,
    first_batch: int,
    batch_count: int,
    workers: int = 4,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    retry_delay_seconds: float = DEFAULT_RETRY_DELAY_SECONDS,
) -> dict[str, object]:
    if min(first_batch, batch_count, workers, max_attempts) <= 0:
        raise ValueError("batch/wave arguments must be positive")
    root = Path(work_dir)
    _read_keep_manifest(root)
    assert_gemini_only_health()
    selected = [
        (index, rows)
        for index, rows in _batches(root)
        if first_batch <= index < first_batch + batch_count
    ]
    if not selected:
        raise ValueError("requested adaptation wave contains no batches")
    failures: list[dict[str, object]] = []
    completed = resumed = records_done = 0
    with ThreadPoolExecutor(max_workers=min(workers, len(selected))) as executor:
        futures = {
            executor.submit(
                _process_batch,
                root,
                batch_index=index,
                rows=rows,
                max_attempts=max_attempts,
                retry_delay_seconds=retry_delay_seconds,
            ): index
            for index, rows in selected
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                result = future.result()
                completed += 1
                records_done += int(result["records"])
                resumed += int(bool(result["resumed"]))
            except Exception as error:  # noqa: BLE001 - report full wave
                failures.append(
                    {
                        "batch_index": index,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
    return {
        "requested_batches": len(selected),
        "completed_batches": completed,
        "completed_records": records_done,
        "resumed_batches": resumed,
        "failures": sorted(failures, key=lambda row: int(row["batch_index"])),
    }


def status(work_dir: Path | str) -> dict[str, int]:
    root = Path(work_dir)
    manifest = _read_keep_manifest(root)
    total_batches = accepted_batches = accepted_records = 0
    for index, rows in _batches(root):
        total_batches += 1
        path = _accepted_path(root, index)
        if path.is_file():
            accepted_batches += 1
            accepted_records += len(rows)
    keep_records = int(manifest["keep_records"])
    return {
        "keep_records": keep_records,
        "total_batches": total_batches,
        "accepted_batches": accepted_batches,
        "accepted_records": accepted_records,
        "pending_records": keep_records - accepted_records,
    }


def finalize(work_dir: Path | str, *, output_jsonl: Path | str) -> dict[str, object]:
    root = Path(work_dir)
    state = status(root)
    if state["pending_records"]:
        raise RuntimeError(
            f"cannot finalize: {state['pending_records']} curated keepers remain pending"
        )
    records: list[dict[str, Any]] = []
    for index, rows in _batches(root):
        payload = json.loads(_accepted_path(root, index).read_text(encoding="utf-8"))
        batch_records = payload.get("records")
        if not isinstance(batch_records, list) or len(batch_records) != len(rows):
            raise RuntimeError(f"accepted batch {index} record count drifted")
        records.extend(dict(row) for row in batch_records)
    destination = common.write_jsonl(output_jsonl, records)
    manifest = {
        "schema": ADAPTED_SCHEMA,
        "records": len(records),
        "prompt_sha256": common.sha256_text(SIMPLIFICATION_SYSTEM_PROMPT),
        "output_jsonl": str(destination),
        "output_sha256": common.sha256_path(destination),
        "output_byte_size": destination.stat().st_size,
    }
    common.atomic_json(destination.with_suffix(destination.suffix + ".manifest.json"), manifest)
    return manifest


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare")
    prepare.add_argument("--candidates-jsonl", type=Path, required=True)
    prepare.add_argument("--curation-jsonl", type=Path, required=True)
    prepare.add_argument("--work-dir", type=Path, required=True)
    prepare.add_argument("--batch-size", type=_positive_int, default=MAX_BATCH_SIZE)

    wave = sub.add_parser("adapt-wave")
    wave.add_argument("--work-dir", type=Path, required=True)
    wave.add_argument("--first-batch", type=_positive_int, required=True)
    wave.add_argument("--batch-count", type=_positive_int, required=True)
    wave.add_argument("--workers", type=_positive_int, default=4)
    wave.add_argument("--max-attempts", type=_positive_int, default=DEFAULT_MAX_ATTEMPTS)
    wave.add_argument("--retry-delay-seconds", type=float, default=DEFAULT_RETRY_DELAY_SECONDS)

    status_parser = sub.add_parser("status")
    status_parser.add_argument("--work-dir", type=Path, required=True)

    finalize_parser = sub.add_parser("finalize")
    finalize_parser.add_argument("--work-dir", type=Path, required=True)
    finalize_parser.add_argument("--output-jsonl", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare_keep(
            candidates_jsonl=args.candidates_jsonl,
            curation_jsonl=args.curation_jsonl,
            work_dir=args.work_dir,
            batch_size=args.batch_size,
        )
    elif args.command == "adapt-wave":
        result = adapt_wave(
            args.work_dir,
            first_batch=args.first_batch,
            batch_count=args.batch_count,
            workers=args.workers,
            max_attempts=args.max_attempts,
            retry_delay_seconds=args.retry_delay_seconds,
        )
    elif args.command == "status":
        result = status(args.work_dir)
    else:
        result = finalize(args.work_dir, output_jsonl=args.output_jsonl)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
