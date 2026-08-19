#!/usr/bin/env python3
"""Resumably compress over-context Superior instruction rows with GemRouter.

This pipeline implements ADR 0103's selected Variant-D policy exactly:

- retain the production instruction-only/no-primary-math/no-primary-code filter;
- preserve every already-fitting production Superior row unchanged;
- send only clean rows whose true atomic R-SFT serialization exceeds 2,048 tokens;
- use at most four examples per GemRouter request;
- use ``superior_reasoning.SIMPLIFICATION_SYSTEM_PROMPT`` verbatim;
- accept only strict RFC-8259 JSON with exact IDs/order and exact output fields;
- validate every rewrite with the real 2,048-token atomic R-SFT serialization;
- checkpoint every attempt and every accepted batch for lossless resume;
- never truncate or silently repair malformed teacher output.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
import time
from types import ModuleType
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

HERE = Path(__file__).resolve().parent
RSFT_DIR = HERE.parent
REPO = RSFT_DIR.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

CANDIDATE_SCHEMA = "small-llm-superior-overcontext-candidates-v1"
ATTEMPT_SCHEMA = "small-llm-superior-gemrouter-attempt-v1"
BATCH_SCHEMA = "small-llm-superior-gemrouter-batch-v1"
ADAPTED_SCHEMA = "small-llm-superior-gemrouter-adapted-v1"
COMPLETE_SCHEMA = "small-llm-superior-reasoning-complete-v1"
CHECKPOINT_SCHEMA = "small-llm-superior-reasoning-checkpoint-v1"
ADAPTED_DIFFICULTY = "simplified_fit"
DEFAULT_BATCH_SIZE = 4
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_RETRY_DELAY_SECONDS = 2.0
DEFAULT_REQUEST_INTERVAL_SECONDS = 0.0
ADAPTATION_REQUEST_TIMEOUT_SECONDS = 190.0
STRICT_RECOVERY_SUFFIX = """

RETRY CORRECTION — the previous fidelity-first rewrite did not pass the local 2,048-token training-context validator. Keep the same task type, essential facts, constraints, and answer semantics, but compress materially harder. Prefer <= 100 words for problem, <= 180 words for reasoning, and <= 100 words for answer. Remove examples, repetition, framing, and nonessential source material first. If the original required output length is itself the blocker, make the smallest scope reduction allowed by the base policy and make the rewritten answer correct for that reduced task. Return strict JSON only and do not add commentary."""
MANUAL_CURATION_SCHEMA = "small-llm-superior-manual-curation-v1"
MANUAL_CURATION_DECISIONS = {"keep", "exclude_math", "exclude_code", "exclude_safety"}


def _load_module(name: str, path: Path) -> ModuleType:
    module_name = f"small_llm_rsft_adapt_{name}"
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


superior = _load_module("superior_reasoning", HERE / "superior_reasoning.py")
transport = _load_module("transport", RSFT_DIR / "dataset.py")


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as error:
        raise RuntimeError(f"{label} is missing or invalid: {path}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must be a JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise RuntimeError(f"JSONL contains a blank line at {path}:{line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"invalid JSON at {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise RuntimeError(f"JSONL row must be an object at {path}:{line_number}")
            yield row


def _candidate_paths(root: Path) -> tuple[Path, Path]:
    return root / "candidates.jsonl", root / "candidates.manifest.json"


def _adapted_paths(root: Path) -> tuple[Path, Path]:
    return root / "adapted.jsonl", root / "adapted.manifest.json"


def _validate_baseline_manifest(path: Path) -> dict[str, Any]:
    payload = _read_json(path, label="baseline production manifest")
    if payload.get("schema") != "small-llm-superior-reasoning-production-v1":
        raise RuntimeError("baseline manifest has the wrong schema")
    if payload.get("policy") != superior.PRODUCTION_FILTER_VERSION:
        raise RuntimeError("baseline manifest uses a different instruction filter")
    if payload.get("context_length") != superior.PRODUCTION_CONTEXT_LENGTH:
        raise RuntimeError("baseline manifest uses a different context length")
    return payload


def read_manual_curation(path: Path | str) -> dict[str, dict[str, str]]:
    """Read explicit human-style row decisions keyed by Superior source UUID."""

    decisions: dict[str, dict[str, str]] = {}
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise RuntimeError(f"manual curation file is missing: {source}")
    for line_number, row in enumerate(_read_jsonl(source), start=1):
        if row.get("schema") != MANUAL_CURATION_SCHEMA:
            raise RuntimeError(f"manual curation row {line_number} has the wrong schema")
        row_id = row.get("id")
        decision = row.get("decision")
        reason = row.get("reason")
        if not isinstance(row_id, str) or not row_id.strip():
            raise RuntimeError(f"manual curation row {line_number} has no id")
        if decision not in MANUAL_CURATION_DECISIONS:
            raise RuntimeError(f"manual curation row {line_number} has invalid decision {decision!r}")
        if not isinstance(reason, str) or not reason.strip():
            raise RuntimeError(f"manual curation row {line_number} has no reason")
        if row_id in decisions:
            raise RuntimeError(f"manual curation contains duplicate id {row_id!r}")
        decisions[row_id] = {"decision": decision, "reason": reason.strip()}
    return decisions


def _validate_candidate_manifest(root: Path, *, baseline_manifest: Path | None = None) -> dict[str, Any]:
    candidates, manifest_path = _candidate_paths(root)
    manifest = _read_json(manifest_path, label="over-context candidate manifest")
    if manifest.get("schema") != CANDIDATE_SCHEMA:
        raise RuntimeError("candidate manifest has the wrong schema")
    if manifest.get("prompt_sha256") != _sha256_text(superior.SIMPLIFICATION_SYSTEM_PROMPT):
        raise RuntimeError("candidate manifest simplification prompt drifted")
    identity = manifest.get("candidates_jsonl")
    if not isinstance(identity, Mapping):
        raise RuntimeError("candidate manifest has no candidates_jsonl identity")
    if identity.get("sha256") != _sha256_path(candidates):
        raise RuntimeError("candidate JSONL hash drifted")
    if identity.get("byte_size") != candidates.stat().st_size:
        raise RuntimeError("candidate JSONL byte size drifted")
    count = sum(1 for _ in _read_jsonl(candidates))
    if identity.get("records") != count:
        raise RuntimeError("candidate JSONL record count drifted")
    if baseline_manifest is not None:
        baseline = _validate_baseline_manifest(baseline_manifest)
        if manifest.get("baseline_output_sha256") != baseline.get("output_sha256"):
            raise RuntimeError("candidate manifest no longer matches the frozen baseline corpus")
    return manifest


def prepare_candidates(
    output_dir: Path | str,
    *,
    baseline_manifest: Path | str,
    rows: Iterable[Mapping[str, Any]] | None = None,
    token_counter: Callable[[str], int] | None = None,
    progress_every: int = 10_000,
) -> dict[str, object]:
    """Freeze all clean instruction rows that need GemRouter compression."""

    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    candidates_path, manifest_path = _candidate_paths(root)
    baseline_path = Path(baseline_manifest).expanduser().resolve()
    baseline = _validate_baseline_manifest(baseline_path)
    if candidates_path.is_file() and manifest_path.is_file():
        manifest = _validate_candidate_manifest(root, baseline_manifest=baseline_path)
        return {
            "candidates_jsonl": str(candidates_path),
            "records": manifest["candidates_jsonl"]["records"],
            "resumed_complete": True,
        }
    if candidates_path.exists() or manifest_path.exists():
        raise RuntimeError("refusing to replace incomplete candidate cache; remove it explicitly first")

    count_tokens = token_counter or superior._default_token_counter()
    source = superior.iter_stage1_rows() if rows is None else rows
    source_rows = 0
    domain_counts: Counter[str] = Counter()
    exclusion_counts: Counter[str] = Counter()
    seen_inputs: set[str] = set()
    rejected_output_count = 0
    duplicate_input_count = 0
    valid_unique_instruction_rows = 0
    fit_unchanged_count = 0
    candidate_count = 0
    candidate_ids: set[str] = set()

    temporary = candidates_path.with_suffix(candidates_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for source_index, row in enumerate(source):
            source_rows += 1
            if progress_every > 0 and source_rows % progress_every == 0:
                print(
                    f"[superior-adapt:prepare] source_rows={source_rows} candidates={candidate_count} fit={fit_unchanged_count}",
                    flush=True,
                )
            if not isinstance(row, Mapping):
                raise ValueError(f"Superior row {source_index} must be an object")
            raw_domain = row.get("domain")
            domain = raw_domain.strip() if isinstance(raw_domain, str) else "<invalid>"
            domain_counts[domain] += 1
            if domain != superior.PRODUCTION_DOMAIN:
                continue

            problem = superior._row_text(row, "input", index=source_index)
            source_id = superior._row_text(row, "uuid", index=source_index)
            output = superior._row_text(row, "output", index=source_index)
            try:
                parsed = superior.parse_teacher_output(output)
            except ValueError:
                rejected_output_count += 1
                continue

            prompt_hash = superior._normalized_input_hash(problem)
            if prompt_hash in seen_inputs:
                duplicate_input_count += 1
                continue
            seen_inputs.add(prompt_hash)
            valid_unique_instruction_rows += 1

            exclusion = superior.instruction_exclusion_reason(problem)
            if exclusion is not None:
                exclusion_counts[exclusion] += 1
                continue
            if any(
                marker in text
                for marker in superior.PRODUCTION_RESERVED_MARKERS
                for text in (problem, parsed.reasoning, parsed.answer)
            ):
                exclusion_counts["reserved_marker_collision"] += 1
                continue

            serialized_tokens = superior.atomic_rsft_serialized_tokens(
                problem=problem,
                reasoning=parsed.reasoning,
                answer=parsed.answer,
                token_counter=count_tokens,
            )
            if serialized_tokens <= superior.PRODUCTION_CONTEXT_LENGTH:
                fit_unchanged_count += 1
                continue

            exclusion_counts["over_context"] += 1
            if source_id in candidate_ids:
                raise RuntimeError(f"duplicate Superior source UUID in adaptation pool: {source_id}")
            candidate_ids.add(source_id)
            candidate = {
                "id": source_id,
                "source_index": source_index,
                "domain": superior.PRODUCTION_DOMAIN,
                "difficulty": ADAPTED_DIFFICULTY,
                "problem": problem,
                "reasoning": parsed.reasoning,
                "answer": parsed.answer,
                "original_serialized_tokens": serialized_tokens,
            }
            handle.write(json.dumps(candidate, ensure_ascii=False, separators=(",", ":")) + "\n")
            candidate_count += 1
    temporary.replace(candidates_path)

    expected = {
        "source_rows": source_rows,
        "domain_counts": dict(sorted(domain_counts.items())),
        "valid_unique_instruction_rows": valid_unique_instruction_rows,
        "rejected_output_count": rejected_output_count,
        "duplicate_input_count": duplicate_input_count,
        "exclusion_counts": dict(sorted(exclusion_counts.items())),
        "selected_count": fit_unchanged_count,
    }
    for key, value in expected.items():
        if baseline.get(key) != value:
            raise RuntimeError(
                f"candidate scan drifted from frozen production baseline at {key}: "
                f"baseline={baseline.get(key)!r} current={value!r}"
            )

    manifest: dict[str, object] = {
        "schema": CANDIDATE_SCHEMA,
        "dataset_id": superior.DATASET_ID,
        "dataset_config": superior.DATASET_CONFIG,
        "dataset_split": superior.DATASET_SPLIT,
        "dataset_revision": superior.DATASET_REVISION,
        "filter_policy": superior.PRODUCTION_FILTER_VERSION,
        "context_length": superior.PRODUCTION_CONTEXT_LENGTH,
        "prompt_sha256": _sha256_text(superior.SIMPLIFICATION_SYSTEM_PROMPT),
        "batch_size_max": superior.SIMPLIFICATION_MAX_BATCH_SIZE,
        "baseline_output_sha256": baseline["output_sha256"],
        "fit_unchanged_rows": fit_unchanged_count,
        "over_context_rows": candidate_count,
        "candidates_jsonl": {
            "path": candidates_path.name,
            "sha256": _sha256_path(candidates_path),
            "byte_size": candidates_path.stat().st_size,
            "records": candidate_count,
        },
        **expected,
    }
    _atomic_json(manifest_path, manifest)
    return {
        "candidates_jsonl": str(candidates_path),
        "records": candidate_count,
        "fit_unchanged_rows": fit_unchanged_count,
        "resumed_complete": False,
    }


def _candidate_batches(path: Path, *, batch_size: int) -> Iterator[tuple[int, list[dict[str, Any]]]]:
    batch: list[dict[str, Any]] = []
    index = 0
    for row in _read_jsonl(path):
        batch.append(row)
        if len(batch) == batch_size:
            index += 1
            yield index, batch
            batch = []
    if batch:
        index += 1
        yield index, batch


def _batch_path(root: Path, index: int) -> Path:
    return root / "batches" / f"batch-{index:05d}.json"


def _attempt_path(root: Path, index: int, attempt: int) -> Path:
    return root / "attempts" / f"batch-{index:05d}-attempt-{attempt:02d}.json"


def _existing_attempt_state(
    root: Path,
    *,
    batch_index: int,
    expected_ids: Sequence[str],
) -> tuple[int, dict[str, Any] | None]:
    """Return the highest persisted attempt number and any recoverable success.

    Rejected attempts remain immutable audit evidence but do not permanently exhaust
    a batch after an outage. A later invocation starts after the highest persisted
    attempt number. If a prior accepted attempt was written just before a crash, it
    is recovered without another provider call.
    """

    highest = 0
    recovered: dict[str, Any] | None = None
    pattern = f"batch-{batch_index:05d}-attempt-*.json"
    for path in sorted((root / "attempts").glob(pattern)):
        payload = _read_json(path, label=f"adaptation attempt {batch_index}")
        attempt = payload.get("attempt")
        if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt <= 0:
            raise RuntimeError(f"adaptation attempt metadata is invalid: {path}")
        if payload.get("batch_index") != batch_index or payload.get("ids") != list(expected_ids):
            raise RuntimeError(f"adaptation attempt identity drifted: {path}")
        highest = max(highest, attempt)
        if payload.get("status") == "accepted":
            accepted = payload.get("accepted_batch")
            if not isinstance(accepted, Mapping):
                raise RuntimeError(f"accepted adaptation attempt has no batch payload: {path}")
            recovered = dict(accepted)
    return highest, recovered


def _strict_recovery_needed(root: Path, *, batch_index: int) -> bool:
    """Use a stronger Variant-D correction only after content-validation failures."""

    pattern = f"batch-{batch_index:05d}-attempt-*.json"
    for path in (root / "attempts").glob(pattern):
        payload = _read_json(path, label=f"adaptation attempt {batch_index}")
        if payload.get("status") != "rejected":
            continue
        error = str(payload.get("error", ""))
        if "remains over context" in error or "reserved R-SFT marker" in error:
            return True
    return False


def _adaptation_messages(
    root: Path,
    *,
    batch_index: int,
    batch: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, str], dict[str, str], bool]:
    system_message, user_message = superior.build_simplification_messages(
        [
            {
                "id": row["id"],
                "skill": superior.SOURCE_SKILLS[superior.PRODUCTION_DOMAIN],
                "problem": row["problem"],
                "reasoning": row["reasoning"],
                "answer": row["answer"],
            }
            for row in batch
        ]
    )
    strict = _strict_recovery_needed(root, batch_index=batch_index)
    if strict:
        system_message = {
            "role": system_message["role"],
            "content": system_message["content"] + STRICT_RECOVERY_SUFFIX,
        }
    return system_message, user_message, strict


def _validate_rewrite(candidate: Mapping[str, Any], rewritten: Mapping[str, str]) -> dict[str, Any]:
    rewritten_tokens = superior.atomic_rsft_serialized_tokens(
        problem=rewritten["problem"],
        reasoning=rewritten["reasoning"],
        answer=rewritten["answer"],
    )
    if rewritten_tokens > superior.PRODUCTION_CONTEXT_LENGTH:
        raise ValueError(
            f"rewrite {rewritten['id']!r} remains over context: {rewritten_tokens} tokens"
        )
    if any(
        marker in text
        for marker in superior.PRODUCTION_RESERVED_MARKERS
        for text in (rewritten["problem"], rewritten["reasoning"], rewritten["answer"])
    ):
        raise ValueError(f"rewrite {rewritten['id']!r} contains a reserved R-SFT marker")
    return {
        "id": rewritten["id"],
        "source_index": int(candidate["source_index"]),
        "domain": superior.PRODUCTION_DOMAIN,
        "difficulty": ADAPTED_DIFFICULTY,
        "problem": rewritten["problem"],
        "reasoning": rewritten["reasoning"],
        "answer": rewritten["answer"],
        "original_serialized_tokens": int(candidate["original_serialized_tokens"]),
        "serialized_token_count": rewritten_tokens,
    }


def _validate_batch_file(path: Path, *, candidates: Sequence[Mapping[str, Any]], index: int) -> dict[str, Any]:
    payload = _read_json(path, label=f"accepted adaptation batch {index}")
    if payload.get("schema") != BATCH_SCHEMA or payload.get("batch_index") != index:
        raise RuntimeError(f"adaptation batch {index} metadata drifted")
    if payload.get("prompt_sha256") != _sha256_text(superior.SIMPLIFICATION_SYSTEM_PROMPT):
        raise RuntimeError(f"adaptation batch {index} prompt drifted")
    expected_ids = [str(row["id"]) for row in candidates]
    if payload.get("ids") != expected_ids:
        raise RuntimeError(f"adaptation batch {index} IDs drifted")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != len(candidates):
        raise RuntimeError(f"adaptation batch {index} record count drifted")
    for candidate, record in zip(candidates, records):
        if not isinstance(record, Mapping):
            raise RuntimeError(f"adaptation batch {index} contains a malformed record")
        normalized = {field: str(record[field]) for field in superior.SIMPLIFICATION_OUTPUT_FIELDS}
        if normalized["id"] != str(candidate["id"]):
            raise RuntimeError(f"adaptation batch {index} record order drifted")
        validated = _validate_rewrite(candidate, normalized)
        if record.get("serialized_token_count") != validated["serialized_token_count"]:
            raise RuntimeError(f"adaptation batch {index} token count drifted")
    return payload


def _usage_add(total: Counter[str], usage: Mapping[str, Any] | None) -> None:
    if usage is None:
        return
    for key, value in usage.items():
        if isinstance(value, int) and not isinstance(value, bool):
            total[key] += value


def _process_adaptation_batch(
    root: Path,
    *,
    batch_index: int,
    batch: Sequence[Mapping[str, Any]],
    max_attempts: int,
    retry_delay_seconds: float,
) -> dict[str, object]:
    """Process one independent four-document GemRouter batch, resumably."""

    accepted_path = _batch_path(root, batch_index)
    if accepted_path.is_file():
        payload = _validate_batch_file(accepted_path, candidates=batch, index=batch_index)
        return {
            "batch_index": batch_index,
            "records": len(batch),
            "api_calls": 0,
            "resumed": True,
            "usage": payload.get("usage") if isinstance(payload.get("usage"), Mapping) else None,
        }

    expected_ids = [str(row["id"]) for row in batch]
    system_message, user_message, strict_recovery = _adaptation_messages(
        root,
        batch_index=batch_index,
        batch=batch,
    )
    client = transport.GeminiDistillationClient(
        timeout_seconds=ADAPTATION_REQUEST_TIMEOUT_SECONDS,
    )
    api_calls = 0
    last_error: Exception | None = None
    highest_attempt, success_payload = _existing_attempt_state(
        root,
        batch_index=batch_index,
        expected_ids=expected_ids,
    )

    for run_attempt in range(1, max_attempts + 1):
        if success_payload is not None:
            break
        attempt = highest_attempt + run_attempt
        attempt_path = _attempt_path(root, batch_index, attempt)

        response_text: str | None = None
        try:
            response = client.complete((system_message, user_message))
            api_calls += 1
            response_text = getattr(response, "content", None)
            if not isinstance(response_text, str) or not response_text.strip():
                raise RuntimeError("GemRouter returned no textual content")
            parsed = superior.parse_simplification_response(
                response_text,
                expected_ids=expected_ids,
            )
            accepted_records = [
                _validate_rewrite(candidate, rewritten)
                for candidate, rewritten in zip(batch, parsed)
            ]
            usage = getattr(response, "usage", None)
            success_payload = {
                "schema": BATCH_SCHEMA,
                "batch_index": batch_index,
                "ids": expected_ids,
                "records": accepted_records,
                "prompt_sha256": _sha256_text(superior.SIMPLIFICATION_SYSTEM_PROMPT),
                "strict_recovery": strict_recovery,
                "teacher_prompt_sha256": _sha256_text(system_message["content"]),
                "model": getattr(response, "model", None),
                "finish_reason": getattr(response, "finish_reason", None),
                "usage": dict(usage) if isinstance(usage, Mapping) else None,
                "attempt": attempt,
            }
            _atomic_json(
                attempt_path,
                {
                    "schema": ATTEMPT_SCHEMA,
                    "batch_index": batch_index,
                    "attempt": attempt,
                    "ids": expected_ids,
                    "status": "accepted",
                    "response_text": response_text,
                    "accepted_batch": success_payload,
                },
            )
            break
        except Exception as error:
            last_error = error
            rejected: dict[str, object] = {
                "schema": ATTEMPT_SCHEMA,
                "batch_index": batch_index,
                "attempt": attempt,
                "ids": expected_ids,
                "status": "rejected",
                "error_type": type(error).__name__,
                "error": str(error),
            }
            if response_text is not None:
                rejected["response_text"] = response_text
            _atomic_json(attempt_path, rejected)
            if run_attempt < max_attempts and retry_delay_seconds:
                time.sleep(retry_delay_seconds)

    if success_payload is None:
        raise RuntimeError(
            f"GemRouter adaptation batch {batch_index} failed after {max_attempts} attempts"
        ) from last_error

    accepted_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(accepted_path, success_payload)
    validated = _validate_batch_file(accepted_path, candidates=batch, index=batch_index)
    return {
        "batch_index": batch_index,
        "records": len(batch),
        "api_calls": api_calls,
        "resumed": False,
        "usage": validated.get("usage") if isinstance(validated.get("usage"), Mapping) else None,
    }


def adapt_wave(
    work_dir: Path | str,
    *,
    first_batch: int,
    batch_count: int,
    workers: int = 8,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    retry_delay_seconds: float = DEFAULT_RETRY_DELAY_SECONDS,
) -> dict[str, object]:
    """Run a bounded concurrent wave of independent Variant-D batches."""

    if first_batch <= 0 or batch_count <= 0 or workers <= 0:
        raise ValueError("first_batch, batch_count, and workers must be positive")
    if not 1 <= batch_size <= superior.SIMPLIFICATION_MAX_BATCH_SIZE:
        raise ValueError(f"batch_size must be in [1, {superior.SIMPLIFICATION_MAX_BATCH_SIZE}]")
    root = Path(work_dir).expanduser().resolve()
    candidates_path, _ = _candidate_paths(root)
    manifest = _validate_candidate_manifest(root)
    total_candidates = int(manifest["candidates_jsonl"]["records"])
    total_batches = (total_candidates + batch_size - 1) // batch_size
    last_batch = min(total_batches, first_batch + batch_count - 1)
    if first_batch > total_batches:
        raise ValueError(f"first_batch {first_batch} exceeds total batches {total_batches}")

    selected = [
        (index, batch)
        for index, batch in _candidate_batches(candidates_path, batch_size=batch_size)
        if first_batch <= index <= last_batch
    ]
    usage_total: Counter[str] = Counter()
    completed_records = 0
    api_calls = 0
    resumed_batches = 0
    failures: list[dict[str, object]] = []

    with ThreadPoolExecutor(max_workers=min(workers, len(selected))) as executor:
        future_map = {
            executor.submit(
                _process_adaptation_batch,
                root,
                batch_index=index,
                batch=batch,
                max_attempts=max_attempts,
                retry_delay_seconds=retry_delay_seconds,
            ): index
            for index, batch in selected
        }
        for future in as_completed(future_map):
            index = future_map[future]
            try:
                result = future.result()
            except Exception as error:
                failures.append(
                    {"batch_index": index, "error_type": type(error).__name__, "error": str(error)}
                )
                continue
            completed_records += int(result["records"])
            api_calls += int(result["api_calls"])
            resumed_batches += int(bool(result["resumed"]))
            usage = result.get("usage")
            _usage_add(usage_total, usage if isinstance(usage, Mapping) else None)
            print(
                f"[superior-adapt:wave] batch={index}/{total_batches} "
                f"records_done={completed_records}/{sum(len(batch) for _, batch in selected)}",
                flush=True,
            )

    return {
        "first_batch": first_batch,
        "last_batch": last_batch,
        "requested_batches": len(selected),
        "completed_batches": len(selected) - len(failures),
        "completed_records": completed_records,
        "api_calls_this_wave": api_calls,
        "resumed_batches": resumed_batches,
        "usage": dict(sorted(usage_total.items())),
        "failures": sorted(failures, key=lambda row: int(row["batch_index"])),
        "total_batches": total_batches,
    }


def adapt_candidates(
    work_dir: Path | str,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    retry_delay_seconds: float = DEFAULT_RETRY_DELAY_SECONDS,
    request_interval_seconds: float = DEFAULT_REQUEST_INTERVAL_SECONDS,
    client: Any | None = None,
    max_batches: int | None = None,
) -> dict[str, object]:
    """Run/resume Variant-D GemRouter compression for every frozen candidate."""

    if not 1 <= batch_size <= superior.SIMPLIFICATION_MAX_BATCH_SIZE:
        raise ValueError(f"batch_size must be in [1, {superior.SIMPLIFICATION_MAX_BATCH_SIZE}]")
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    if retry_delay_seconds < 0 or request_interval_seconds < 0:
        raise ValueError("delays cannot be negative")

    root = Path(work_dir).expanduser().resolve()
    candidates_path, _candidate_manifest_path = _candidate_paths(root)
    candidate_manifest = _validate_candidate_manifest(root)
    adapted_path, adapted_manifest_path = _adapted_paths(root)
    if adapted_path.is_file() and adapted_manifest_path.is_file():
        manifest = _read_json(adapted_manifest_path, label="adapted corpus manifest")
        if manifest.get("schema") != ADAPTED_SCHEMA:
            raise RuntimeError("adapted corpus manifest has the wrong schema")
        if manifest.get("adapted_jsonl", {}).get("sha256") != _sha256_path(adapted_path):
            raise RuntimeError("adapted corpus hash drifted")
        return {
            "adapted_jsonl": str(adapted_path),
            "records": manifest["adapted_jsonl"]["records"],
            "resumed_complete": True,
        }
    if adapted_path.exists() or adapted_manifest_path.exists():
        raise RuntimeError("refusing to replace incomplete final adapted corpus")

    total_candidates = int(candidate_manifest["candidates_jsonl"]["records"])
    total_batches = (total_candidates + batch_size - 1) // batch_size
    live_client = client
    usage_total: Counter[str] = Counter()
    batch_files: list[dict[str, object]] = []
    api_calls = 0
    completed = 0

    for batch_index, batch in _candidate_batches(candidates_path, batch_size=batch_size):
        if max_batches is not None and batch_index > max_batches:
            break
        accepted_path = _batch_path(root, batch_index)
        if accepted_path.is_file():
            payload = _validate_batch_file(accepted_path, candidates=batch, index=batch_index)
            usage = payload.get("usage")
            _usage_add(usage_total, usage if isinstance(usage, Mapping) else None)
            batch_files.append(
                {
                    "path": accepted_path.relative_to(root).as_posix(),
                    "sha256": _sha256_path(accepted_path),
                    "byte_size": accepted_path.stat().st_size,
                    "batch_index": batch_index,
                    "records": len(batch),
                }
            )
            completed += len(batch)
            continue

        expected_ids = [str(row["id"]) for row in batch]
        system_message, user_message, strict_recovery = _adaptation_messages(
            root,
            batch_index=batch_index,
            batch=batch,
        )
        last_error: Exception | None = None
        highest_attempt, success_payload = _existing_attempt_state(
            root,
            batch_index=batch_index,
            expected_ids=expected_ids,
        )
        for run_attempt in range(1, max_attempts + 1):
            if success_payload is not None:
                break
            attempt = highest_attempt + run_attempt
            attempt_path = _attempt_path(root, batch_index, attempt)
            if live_client is None:
                live_client = transport.GeminiDistillationClient(
                    timeout_seconds=ADAPTATION_REQUEST_TIMEOUT_SECONDS,
                )
            if api_calls and request_interval_seconds:
                time.sleep(request_interval_seconds)
            try:
                response = live_client.complete((system_message, user_message))
                api_calls += 1
                response_text = getattr(response, "content", None)
                if not isinstance(response_text, str) or not response_text.strip():
                    raise RuntimeError("GemRouter returned no textual content")
                parsed = superior.parse_simplification_response(
                    response_text,
                    expected_ids=expected_ids,
                )
                accepted_records = [
                    _validate_rewrite(candidate, rewritten)
                    for candidate, rewritten in zip(batch, parsed)
                ]
                usage = getattr(response, "usage", None)
                model = getattr(response, "model", None)
                finish_reason = getattr(response, "finish_reason", None)
                success_payload = {
                    "schema": BATCH_SCHEMA,
                    "batch_index": batch_index,
                    "ids": expected_ids,
                    "records": accepted_records,
                    "prompt_sha256": _sha256_text(superior.SIMPLIFICATION_SYSTEM_PROMPT),
                    "strict_recovery": strict_recovery,
                    "teacher_prompt_sha256": _sha256_text(system_message["content"]),
                    "model": model,
                    "finish_reason": finish_reason,
                    "usage": dict(usage) if isinstance(usage, Mapping) else None,
                    "attempt": attempt,
                }
                _atomic_json(
                    attempt_path,
                    {
                        "schema": ATTEMPT_SCHEMA,
                        "batch_index": batch_index,
                        "attempt": attempt,
                        "ids": expected_ids,
                        "status": "accepted",
                        "response_text": response_text,
                        "accepted_batch": success_payload,
                    },
                )
                break
            except Exception as error:  # fail-closed; persisted for deterministic resume/audit
                last_error = error
                _atomic_json(
                    attempt_path,
                    {
                        "schema": ATTEMPT_SCHEMA,
                        "batch_index": batch_index,
                        "attempt": attempt,
                        "ids": expected_ids,
                        "status": "rejected",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    },
                )
                if run_attempt < max_attempts and retry_delay_seconds:
                    time.sleep(retry_delay_seconds)

        if success_payload is None:
            raise RuntimeError(
                f"GemRouter adaptation batch {batch_index}/{total_batches} failed after "
                f"{max_attempts} attempts; resume after inspecting persisted attempts"
            ) from last_error

        accepted_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_json(accepted_path, success_payload)
        validated = _validate_batch_file(accepted_path, candidates=batch, index=batch_index)
        usage = validated.get("usage")
        _usage_add(usage_total, usage if isinstance(usage, Mapping) else None)
        batch_files.append(
            {
                "path": accepted_path.relative_to(root).as_posix(),
                "sha256": _sha256_path(accepted_path),
                "byte_size": accepted_path.stat().st_size,
                "batch_index": batch_index,
                "records": len(batch),
            }
        )
        completed += len(batch)
        print(
            f"[superior-adapt:gemrouter] batch={batch_index}/{total_batches} "
            f"records={completed}/{total_candidates} api_calls_this_run={api_calls}",
            flush=True,
        )

    if max_batches is not None and max_batches < total_batches:
        return {
            "records_completed": completed,
            "records_total": total_candidates,
            "batches_total": total_batches,
            "api_calls_this_run": api_calls,
            "complete": False,
        }
    if completed != total_candidates or len(batch_files) != total_batches:
        raise RuntimeError(
            f"adaptation assembled {completed}/{total_candidates} records across "
            f"{len(batch_files)}/{total_batches} batches"
        )

    temporary = adapted_path.with_suffix(adapted_path.suffix + ".tmp")
    rewritten_tokens: list[int] = []
    original_tokens: list[int] = []
    with temporary.open("w", encoding="utf-8") as handle:
        for batch_meta in sorted(batch_files, key=lambda row: int(row["batch_index"])):
            payload = _read_json(root / str(batch_meta["path"]), label="accepted adaptation batch")
            records = payload["records"]
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                rewritten_tokens.append(int(record["serialized_token_count"]))
                original_tokens.append(int(record["original_serialized_tokens"]))
    temporary.replace(adapted_path)

    manifest: dict[str, object] = {
        "schema": ADAPTED_SCHEMA,
        "candidate_manifest_sha256": _sha256_path(_candidate_paths(root)[1]),
        "candidates_jsonl_sha256": candidate_manifest["candidates_jsonl"]["sha256"],
        "prompt_sha256": _sha256_text(superior.SIMPLIFICATION_SYSTEM_PROMPT),
        "prompt_policy": "ADR-0103-variant-D-fidelity-first",
        "batch_size": batch_size,
        "total_batches": total_batches,
        "provider_model": transport.DEFAULT_MODEL,
        "context_length": superior.PRODUCTION_CONTEXT_LENGTH,
        "usage": dict(sorted(usage_total.items())),
        "original_serialized_tokens": {
            "min": min(original_tokens),
            "max": max(original_tokens),
        },
        "rewritten_serialized_tokens": {
            "min": min(rewritten_tokens),
            "max": max(rewritten_tokens),
        },
        "adapted_jsonl": {
            "path": adapted_path.name,
            "sha256": _sha256_path(adapted_path),
            "byte_size": adapted_path.stat().st_size,
            "records": completed,
        },
        "batches": batch_files,
    }
    _atomic_json(adapted_manifest_path, manifest)
    return {
        "adapted_jsonl": str(adapted_path),
        "records": completed,
        "batches": total_batches,
        "api_calls_this_run": api_calls,
        "usage": dict(sorted(usage_total.items())),
        "resumed_complete": False,
    }



def finalize_checkpoint_dataset(
    work_dir: Path | str,
    *,
    baseline_jsonl: Path | str,
    baseline_manifest: Path | str,
    manual_curation_jsonl: Path | str,
    output_jsonl: Path | str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int = superior.DEFAULT_SEED,
) -> dict[str, object]:
    """Freeze the currently available manually-kept Variant-D checkpoint.

    Unlike ``finalize_complete_dataset``, this intentionally permits missing
    adaptation batches. Only validated accepted batch files are harvested, and
    only rows with an explicit ``keep`` curation decision are emitted. The
    manifest records the still-pending kept-row count so this checkpoint cannot
    be mistaken for the eventual complete corpus.
    """

    if not 1 <= batch_size <= superior.SIMPLIFICATION_MAX_BATCH_SIZE:
        raise ValueError(f"batch_size must be in [1, {superior.SIMPLIFICATION_MAX_BATCH_SIZE}]")

    root = Path(work_dir).expanduser().resolve()
    candidate_manifest = _validate_candidate_manifest(root, baseline_manifest=Path(baseline_manifest))
    candidates_path, candidate_manifest_path = _candidate_paths(root)
    candidate_rows = list(_read_jsonl(candidates_path))
    candidate_ids = [str(row["id"]) for row in candidate_rows]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise RuntimeError("frozen over-context candidates contain duplicate IDs")

    decisions = read_manual_curation(manual_curation_jsonl)
    missing_decisions = [row_id for row_id in candidate_ids if row_id not in decisions]
    extra_decisions = sorted(set(decisions).difference(candidate_ids))
    if missing_decisions or extra_decisions:
        raise RuntimeError(
            "manual curation must cover every over-context candidate exactly once; "
            f"missing={len(missing_decisions)} extra={len(extra_decisions)}"
        )
    decision_counts = Counter(row["decision"] for row in decisions.values())

    baseline_path = Path(baseline_jsonl).expanduser().resolve()
    baseline_manifest_path = Path(baseline_manifest).expanduser().resolve()
    baseline = _validate_baseline_manifest(baseline_manifest_path)
    if baseline.get("output_sha256") != _sha256_path(baseline_path):
        raise RuntimeError("baseline production JSONL hash drifted")

    five_fields = ("skill", "difficulty", "problem", "reasoning", "answer")
    combined: list[dict[str, str]] = []
    normalized_problems: set[str] = set()
    unchanged_superior = 0
    gemini_rows = 0
    for line_number, raw in enumerate(_read_jsonl(baseline_path), start=1):
        if set(raw) != set(five_fields):
            raise RuntimeError(f"baseline row {line_number} has the wrong schema")
        record = {field: str(raw[field]).strip() for field in five_fields}
        if record["skill"] == superior.SOURCE_SKILLS[superior.PRODUCTION_DOMAIN]:
            unchanged_superior += 1
        else:
            gemini_rows += 1
        normalized = superior._normalized_input_hash(record["problem"])
        if normalized in normalized_problems:
            raise RuntimeError("baseline reasoning corpus contains duplicate normalized prompts")
        normalized_problems.add(normalized)
        combined.append(record)

    accepted_batches = 0
    accepted_records = 0
    adapted_rows = 0
    accepted_kept_rows = 0
    excluded_adapted_rows = 0
    duplicate_rewrite_ids: list[str] = []
    seen_adapted_ids: set[str] = set()
    accepted_identity_lines: list[str] = []
    for index, batch in _candidate_batches(candidates_path, batch_size=batch_size):
        batch_path = _batch_path(root, index)
        if not batch_path.is_file():
            continue
        payload = _validate_batch_file(batch_path, candidates=batch, index=index)
        accepted_batches += 1
        records = payload["records"]
        accepted_records += len(records)
        accepted_identity_lines.append(f"{index}:{_sha256_path(batch_path)}")
        for raw in records:
            row_id = str(raw["id"])
            if row_id in seen_adapted_ids:
                raise RuntimeError(f"accepted adaptation batches contain duplicate id {row_id!r}")
            seen_adapted_ids.add(row_id)
            if decisions[row_id]["decision"] != "keep":
                excluded_adapted_rows += 1
                continue
            accepted_kept_rows += 1
            record = superior.to_rsft_mapping(raw)
            normalized = superior._normalized_input_hash(record["problem"])
            if normalized in normalized_problems:
                duplicate_rewrite_ids.append(row_id)
                continue
            normalized_problems.add(normalized)
            combined.append(record)
            adapted_rows += 1

    expected_keep = int(decision_counts.get("keep", 0))
    pending_keep = expected_keep - accepted_kept_rows
    if pending_keep < 0:
        raise RuntimeError("accepted kept adaptations exceed manual-curation keep count")
    if unchanged_superior != int(baseline["selected_count"]):
        raise RuntimeError("unchanged Superior row count drifted from baseline manifest")
    if gemini_rows != int(baseline["gemini_rows"]):
        raise RuntimeError("Gemini anchor count drifted from baseline manifest")
    if adapted_rows <= 0:
        raise RuntimeError("checkpoint contains no manually-kept accepted adaptations")

    max_tokens = 0
    min_tokens: int | None = None
    for record in combined:
        tokens = superior.atomic_rsft_serialized_tokens(
            problem=record["problem"],
            reasoning=record["reasoning"],
            answer=record["answer"],
        )
        if tokens > superior.PRODUCTION_CONTEXT_LENGTH:
            raise RuntimeError(f"final checkpoint corpus contains an over-context record: {tokens}")
        max_tokens = max(max_tokens, tokens)
        min_tokens = tokens if min_tokens is None else min(min_tokens, tokens)

    random.Random(seed).shuffle(combined)
    destination = Path(output_jsonl).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in combined:
            ordered = {field: record[field] for field in five_fields}
            handle.write(json.dumps(ordered, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(destination)

    accepted_identity_sha = hashlib.sha256(
        ("\n".join(accepted_identity_lines) + "\n").encode("utf-8")
    ).hexdigest()
    manifest: dict[str, object] = {
        "schema": CHECKPOINT_SCHEMA,
        "policy": superior.PRODUCTION_FILTER_VERSION,
        "production_domain": superior.PRODUCTION_DOMAIN,
        "adaptation_policy": "ADR-0103-variant-D-fidelity-first",
        "checkpoint_contract": "validated-accepted-batches-only-v1",
        "prompt_sha256": _sha256_text(superior.SIMPLIFICATION_SYSTEM_PROMPT),
        "context_length": superior.PRODUCTION_CONTEXT_LENGTH,
        "seed": seed,
        "candidate_rows": len(candidate_ids),
        "candidate_manifest_sha256": _sha256_path(candidate_manifest_path),
        "candidates_jsonl_sha256": candidate_manifest["candidates_jsonl"]["sha256"],
        "manual_curation_counts": dict(sorted(decision_counts.items())),
        "manual_curation_sha256": _sha256_path(Path(manual_curation_jsonl).expanduser().resolve()),
        "accepted_batches": accepted_batches,
        "accepted_adapted_records": accepted_records,
        "accepted_kept_adapted_rows": accepted_kept_rows,
        "accepted_batches_identity_sha256": accepted_identity_sha,
        "unchanged_superior_rows": unchanged_superior,
        "adapted_superior_rows": adapted_rows,
        "duplicate_rewrite_exclusions": len(duplicate_rewrite_ids),
        "duplicate_rewrite_excluded_ids": duplicate_rewrite_ids,
        "pending_kept_adaptation_rows": pending_keep,
        "manually_excluded_adapted_rows": excluded_adapted_rows,
        "clean_superior_instruction_rows": unchanged_superior + adapted_rows,
        "gemini_rows": gemini_rows,
        "combined_rows": len(combined),
        "serialized_token_range": {"min": min_tokens, "max": max_tokens},
        "baseline_jsonl_sha256": _sha256_path(baseline_path),
        "output_jsonl": str(destination.relative_to(REPO)) if destination.is_relative_to(REPO) else str(destination),
        "output_sha256": _sha256_path(destination),
        "output_byte_size": destination.stat().st_size,
    }
    manifest_path = destination.with_suffix(destination.suffix + ".manifest.json")
    _atomic_json(manifest_path, manifest)
    return {**manifest, "manifest": str(manifest_path)}


def finalize_complete_dataset(
    work_dir: Path | str,
    *,
    baseline_jsonl: Path | str,
    baseline_manifest: Path | str,
    manual_curation_jsonl: Path | str,
    output_jsonl: Path | str,
    seed: int = superior.DEFAULT_SEED,
) -> dict[str, object]:
    """Merge unchanged rows plus manually kept rewrites and Gemini anchors.

    Finalization fails unless every frozen over-context candidate has exactly one
    explicit manual-curation decision. Excluded rows may have been adapted during
    the parallel GemRouter pass, but they are never emitted into the final corpus.
    """

    root = Path(work_dir).expanduser().resolve()
    adapted_path, adapted_manifest_path = _adapted_paths(root)
    adapted_manifest = _read_json(adapted_manifest_path, label="adapted corpus manifest")
    if adapted_manifest.get("schema") != ADAPTED_SCHEMA:
        raise RuntimeError("adapted manifest has the wrong schema")
    if adapted_manifest.get("adapted_jsonl", {}).get("sha256") != _sha256_path(adapted_path):
        raise RuntimeError("adapted corpus hash drifted")

    baseline_path = Path(baseline_jsonl).expanduser().resolve()
    baseline_manifest_path = Path(baseline_manifest).expanduser().resolve()
    baseline = _validate_baseline_manifest(baseline_manifest_path)
    candidates_path, _ = _candidate_paths(root)
    candidate_ids = [str(row["id"]) for row in _read_jsonl(candidates_path)]
    decisions = read_manual_curation(manual_curation_jsonl)
    missing_decisions = [row_id for row_id in candidate_ids if row_id not in decisions]
    extra_decisions = sorted(set(decisions).difference(candidate_ids))
    if missing_decisions or extra_decisions:
        raise RuntimeError(
            "manual curation must cover every over-context candidate exactly once; "
            f"missing={len(missing_decisions)} extra={len(extra_decisions)}"
        )
    decision_counts = Counter(row["decision"] for row in decisions.values())
    if baseline.get("output_sha256") != _sha256_path(baseline_path):
        raise RuntimeError("baseline production JSONL hash drifted")

    five_fields = ("skill", "difficulty", "problem", "reasoning", "answer")
    combined: list[dict[str, str]] = []
    normalized_problems: set[str] = set()
    unchanged_superior = 0
    gemini_rows = 0
    for line_number, raw in enumerate(_read_jsonl(baseline_path), start=1):
        if set(raw) != set(five_fields):
            raise RuntimeError(f"baseline row {line_number} has the wrong schema")
        record = {field: str(raw[field]).strip() for field in five_fields}
        if record["skill"] == superior.SOURCE_SKILLS[superior.PRODUCTION_DOMAIN]:
            unchanged_superior += 1
        else:
            gemini_rows += 1
        normalized = superior._normalized_input_hash(record["problem"])
        if normalized in normalized_problems:
            raise RuntimeError("baseline reasoning corpus contains duplicate normalized prompts")
        normalized_problems.add(normalized)
        combined.append(record)

    adapted_rows = 0
    excluded_adapted_rows = 0
    seen_adapted_ids: set[str] = set()
    for raw in _read_jsonl(adapted_path):
        row_id = str(raw["id"])
        if row_id in seen_adapted_ids:
            raise RuntimeError(f"adapted corpus contains duplicate id {row_id!r}")
        seen_adapted_ids.add(row_id)
        decision = decisions[row_id]["decision"]
        if decision != "keep":
            excluded_adapted_rows += 1
            continue
        record = superior.to_rsft_mapping(raw)
        normalized = superior._normalized_input_hash(record["problem"])
        if normalized in normalized_problems:
            raise RuntimeError(f"adapted rewrite created a duplicate normalized prompt: {row_id}")
        normalized_problems.add(normalized)
        combined.append(record)
        adapted_rows += 1

    if seen_adapted_ids != set(candidate_ids):
        raise RuntimeError("adapted corpus IDs do not exactly match frozen over-context candidates")
    expected_adapted = int(decision_counts.get("keep", 0))
    if adapted_rows != expected_adapted:
        raise RuntimeError(f"kept adapted row count {adapted_rows} != expected {expected_adapted}")
    if unchanged_superior != int(baseline["selected_count"]):
        raise RuntimeError("unchanged Superior row count drifted from baseline manifest")
    if gemini_rows != int(baseline["gemini_rows"]):
        raise RuntimeError("Gemini anchor count drifted from baseline manifest")

    max_tokens = 0
    min_tokens: int | None = None
    for record in combined:
        tokens = superior.atomic_rsft_serialized_tokens(
            problem=record["problem"],
            reasoning=record["reasoning"],
            answer=record["answer"],
        )
        if tokens > superior.PRODUCTION_CONTEXT_LENGTH:
            raise RuntimeError(f"final complete corpus contains an over-context record: {tokens}")
        max_tokens = max(max_tokens, tokens)
        min_tokens = tokens if min_tokens is None else min(min_tokens, tokens)

    random.Random(seed).shuffle(combined)
    destination = Path(output_jsonl).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in combined:
            ordered = {field: record[field] for field in five_fields}
            handle.write(json.dumps(ordered, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(destination)

    manifest: dict[str, object] = {
        "schema": COMPLETE_SCHEMA,
        "policy": superior.PRODUCTION_FILTER_VERSION,
        "adaptation_policy": "ADR-0103-variant-D-fidelity-first",
        "prompt_sha256": _sha256_text(superior.SIMPLIFICATION_SYSTEM_PROMPT),
        "context_length": superior.PRODUCTION_CONTEXT_LENGTH,
        "seed": seed,
        "unchanged_superior_rows": unchanged_superior,
        "adapted_superior_rows": adapted_rows,
        "manually_excluded_adapted_rows": excluded_adapted_rows,
        "manual_curation_counts": dict(sorted(decision_counts.items())),
        "manual_curation_sha256": _sha256_path(Path(manual_curation_jsonl).expanduser().resolve()),
        "clean_superior_instruction_rows": unchanged_superior + adapted_rows,
        "gemini_rows": gemini_rows,
        "combined_rows": len(combined),
        "serialized_token_range": {"min": min_tokens, "max": max_tokens},
        "baseline_jsonl_sha256": _sha256_path(baseline_path),
        "adapted_jsonl_sha256": _sha256_path(adapted_path),
        "output_jsonl": str(destination),
        "output_sha256": _sha256_path(destination),
        "output_byte_size": destination.stat().st_size,
    }
    manifest_path = destination.with_suffix(destination.suffix + ".manifest.json")
    _atomic_json(manifest_path, manifest)
    return {**manifest, "manifest": str(manifest_path)}


def _nonnegative_float(value: str) -> float:
    result = float(value)
    if result < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return result


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="scan Stage 1 and freeze over-context clean candidates")
    prepare.add_argument("--work-dir", type=Path, required=True)
    prepare.add_argument("--baseline-manifest", type=Path, required=True)

    adapt = sub.add_parser("adapt", help="run/resume GemRouter Variant-D compression")
    adapt.add_argument("--work-dir", type=Path, required=True)
    adapt.add_argument("--batch-size", type=_positive_int, default=DEFAULT_BATCH_SIZE)
    adapt.add_argument("--max-attempts", type=_positive_int, default=DEFAULT_MAX_ATTEMPTS)
    adapt.add_argument("--retry-delay-seconds", type=_nonnegative_float, default=DEFAULT_RETRY_DELAY_SECONDS)
    adapt.add_argument("--request-interval-seconds", type=_nonnegative_float, default=DEFAULT_REQUEST_INTERVAL_SECONDS)
    adapt.add_argument("--max-batches", type=_positive_int)

    wave = sub.add_parser("adapt-wave", help="run a bounded concurrent wave of Variant-D batches")
    wave.add_argument("--work-dir", type=Path, required=True)
    wave.add_argument("--first-batch", type=_positive_int, required=True)
    wave.add_argument("--batch-count", type=_positive_int, required=True)
    wave.add_argument("--workers", type=_positive_int, default=8)
    wave.add_argument("--batch-size", type=_positive_int, default=DEFAULT_BATCH_SIZE)
    wave.add_argument("--max-attempts", type=_positive_int, default=DEFAULT_MAX_ATTEMPTS)
    wave.add_argument("--retry-delay-seconds", type=_nonnegative_float, default=DEFAULT_RETRY_DELAY_SECONDS)

    checkpoint = sub.add_parser("finalize-checkpoint", help="freeze the currently accepted kept adaptations plus baseline")
    checkpoint.add_argument("--work-dir", type=Path, required=True)
    checkpoint.add_argument("--baseline-jsonl", type=Path, required=True)
    checkpoint.add_argument("--baseline-manifest", type=Path, required=True)
    checkpoint.add_argument("--manual-curation-jsonl", type=Path, required=True)
    checkpoint.add_argument("--output-jsonl", type=Path, required=True)
    checkpoint.add_argument("--batch-size", type=_positive_int, default=DEFAULT_BATCH_SIZE)
    checkpoint.add_argument("--seed", type=int, default=superior.DEFAULT_SEED)

    finalize = sub.add_parser("finalize", help="freeze the complete fit+adapted+Gemini R-SFT JSONL")
    finalize.add_argument("--work-dir", type=Path, required=True)
    finalize.add_argument("--baseline-jsonl", type=Path, required=True)
    finalize.add_argument("--baseline-manifest", type=Path, required=True)
    finalize.add_argument("--manual-curation-jsonl", type=Path, required=True)
    finalize.add_argument("--output-jsonl", type=Path, required=True)
    finalize.add_argument("--seed", type=int, default=superior.DEFAULT_SEED)

    run = sub.add_parser("run", help="prepare, adapt, and finalize the complete corpus")
    run.add_argument("--work-dir", type=Path, required=True)
    run.add_argument("--baseline-jsonl", type=Path, required=True)
    run.add_argument("--baseline-manifest", type=Path, required=True)
    run.add_argument("--manual-curation-jsonl", type=Path, required=True)
    run.add_argument("--output-jsonl", type=Path, required=True)
    run.add_argument("--batch-size", type=_positive_int, default=DEFAULT_BATCH_SIZE)
    run.add_argument("--max-attempts", type=_positive_int, default=DEFAULT_MAX_ATTEMPTS)
    run.add_argument("--retry-delay-seconds", type=_nonnegative_float, default=DEFAULT_RETRY_DELAY_SECONDS)
    run.add_argument("--request-interval-seconds", type=_nonnegative_float, default=DEFAULT_REQUEST_INTERVAL_SECONDS)
    run.add_argument("--seed", type=int, default=superior.DEFAULT_SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare_candidates(args.work_dir, baseline_manifest=args.baseline_manifest)
    elif args.command == "adapt":
        result = adapt_candidates(
            args.work_dir,
            batch_size=args.batch_size,
            max_attempts=args.max_attempts,
            retry_delay_seconds=args.retry_delay_seconds,
            request_interval_seconds=args.request_interval_seconds,
            max_batches=args.max_batches,
        )
    elif args.command == "adapt-wave":
        result = adapt_wave(
            args.work_dir,
            first_batch=args.first_batch,
            batch_count=args.batch_count,
            workers=args.workers,
            batch_size=args.batch_size,
            max_attempts=args.max_attempts,
            retry_delay_seconds=args.retry_delay_seconds,
        )
    elif args.command == "finalize-checkpoint":
        result = finalize_checkpoint_dataset(
            args.work_dir,
            baseline_jsonl=args.baseline_jsonl,
            baseline_manifest=args.baseline_manifest,
            manual_curation_jsonl=args.manual_curation_jsonl,
            output_jsonl=args.output_jsonl,
            batch_size=args.batch_size,
            seed=args.seed,
        )
    elif args.command == "finalize":
        result = finalize_complete_dataset(
            args.work_dir,
            baseline_jsonl=args.baseline_jsonl,
            baseline_manifest=args.baseline_manifest,
            manual_curation_jsonl=args.manual_curation_jsonl,
            output_jsonl=args.output_jsonl,
            seed=args.seed,
        )
    else:
        prepare_candidates(args.work_dir, baseline_manifest=args.baseline_manifest)
        adapt_candidates(
            args.work_dir,
            batch_size=args.batch_size,
            max_attempts=args.max_attempts,
            retry_delay_seconds=args.retry_delay_seconds,
            request_interval_seconds=args.request_interval_seconds,
        )
        result = finalize_complete_dataset(
            args.work_dir,
            baseline_jsonl=args.baseline_jsonl,
            baseline_manifest=args.baseline_manifest,
            manual_curation_jsonl=args.manual_curation_jsonl,
            output_jsonl=args.output_jsonl,
            seed=args.seed,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
