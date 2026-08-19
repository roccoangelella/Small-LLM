#!/usr/bin/env python3
"""Semantic row-level curation for Superior instruction candidates via OpenCode.

Every candidate receives an explicit UUID-level decision. The reviewer is DeepSeek
V4 Flash through OpenCode with maximum reasoning effort; batching is transport-only,
not a heuristic classifier. The final manual-curation JSONL is validated to contain
exactly one decision for every candidate before downstream dataset finalization.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Iterator, Mapping, Sequence

SCHEMA = "small-llm-superior-manual-curation-v1"
MODEL = "opencode-go/deepseek-v4-flash"
VARIANT = "max"
DECISIONS = {"keep", "exclude_math", "exclude_code", "exclude_safety", "uncertain"}
DEFAULT_BATCH_SIZE = 10
DEFAULT_WORKERS = 8
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_OPENCODE_TIMEOUT_SECONDS = 300.0
CURATOR_AGENT_NAME = "superior_curator"
CURATOR_AGENT_MARKDOWN = """---
description: Pure semantic dataset curator with no tool access.
mode: primary
model: opencode-go/deepseek-v4-flash
variant: max
steps: 4
permission:
  read: deny
  glob: deny
  grep: deny
  list: deny
  lsp: deny
  edit: deny
  bash: deny
  task: deny
  websearch: deny
  webfetch: deny
  question: deny
---

You are a pure semantic dataset curator. Never use tools or inspect the repository. Classify only the items supplied in the user message or attached file. Follow the requested JSON schema exactly and return only the final classification JSON.
"""

SYSTEM_PROMPT = """You are semantically curating instruction-following examples for the first large reasoning-SFT stage of a ~100M language model.

Review EACH item independently. The exclusions are:
- exclude_math: the primary task teaches or requires mathematical/numerical computation, algebra, geometry, calculus, probability calculation, formula manipulation, or proof-style mathematics.
- exclude_code: the primary task asks to write, implement, debug, fix, execute, configure, or materially reason through source code, shell commands, programming APIs, or software-system setup.
- exclude_safety: the task requests clearly unsafe training content, especially sexual content involving minors, sexual exploitation, or similarly severe harmful material.

KEEP conceptual science, technical explanations, software/AI career advice, architecture descriptions, product/tool explanations, ordinary business/data prose, instruction constraints involving counts/percentages, and any row where math/code is merely incidental context rather than the primary requested capability. Do not overuse exclude_safety for ordinary mature, political, medical, or controversial discussion; reserve it for clearly unsafe requested behavior/content.

Rows may contain multiple concatenated user tasks. Judge the actual combined task. If ANY substantial requested subtask is primary math/computation, programming/code execution, or clearly unsafe content, exclude it using the matching category. Use uncertain only when the task is genuinely ambiguous after careful reading.

Return strict RFC 8259 JSON only: one array in the same order as input. Every object must contain exactly: id, decision, reason. decision must be one of keep, exclude_math, exclude_code, exclude_safety, uncertain. reason must be a concise task-specific semantic justification. No markdown fences and no text outside the JSON."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise RuntimeError(f"blank line at {path}:{line_number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise RuntimeError(f"non-object row at {path}:{line_number}")
            yield row


def _batches(
    rows: Sequence[dict[str, Any]],
    size: int,
    *,
    max_chars: int | None = None,
) -> Iterator[tuple[int, list[dict[str, Any]]]]:
    if max_chars is None:
        for offset in range(0, len(rows), size):
            yield offset // size + 1, list(rows[offset : offset + size])
        return

    batch: list[dict[str, Any]] = []
    batch_chars = 0
    batch_index = 0
    for row in rows:
        row_chars = len(str(row["problem"]))
        if batch and (len(batch) >= size or batch_chars + row_chars > max_chars):
            batch_index += 1
            yield batch_index, batch
            batch = []
            batch_chars = 0
        batch.append(row)
        batch_chars += row_chars
    if batch:
        batch_index += 1
        yield batch_index, batch


def _review_payload(batch: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    return [{"id": str(row["id"]), "problem": str(row["problem"]).strip()} for row in batch]


def _clean_json_text(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```json") and cleaned.endswith("```"):
        return cleaned[len("```json") : -len("```")].strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        return cleaned[len("```") : -len("```")].strip()
    return cleaned


def _parse_json_events(stdout: str) -> str:
    text_parts: list[str] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, Mapping):
            continue
        part = event.get("part")
        if isinstance(part, Mapping) and part.get("type") == "text":
            text = part.get("text")
            if isinstance(text, str):
                text_parts.append(text)
        text = event.get("text")
        if isinstance(text, str):
            text_parts.append(text)
    for text in reversed(text_parts):
        cleaned = _clean_json_text(text)
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, list):
            return cleaned
    raise RuntimeError("OpenCode produced no parseable JSON-array assistant text event")


def _validate_response(text: str, expected_ids: Sequence[str]) -> list[dict[str, str]]:
    cleaned = _clean_json_text(text)
    try:
        raw = json.loads(cleaned)
    except json.JSONDecodeError as error:
        raise RuntimeError("reviewer response did not contain one parseable JSON decision array") from error
    if not isinstance(raw, list) or len(raw) != len(expected_ids):
        raise RuntimeError("reviewer returned the wrong number of decisions")
    result: list[dict[str, str]] = []
    for index, (item, expected_id) in enumerate(zip(raw, expected_ids)):
        if not isinstance(item, Mapping) or set(item) != {"id", "decision", "reason"}:
            raise RuntimeError(f"review item {index} has the wrong fields")
        item_id = item["id"]
        decision = item["decision"]
        reason = item["reason"]
        if item_id != expected_id:
            raise RuntimeError(f"review item {index} id drifted")
        if decision not in DECISIONS:
            raise RuntimeError(f"review item {index} has invalid decision {decision!r}")
        if not isinstance(reason, str) or not reason.strip():
            raise RuntimeError(f"review item {index} has no reason")
        result.append({"id": item_id, "decision": decision, "reason": reason.strip()})
    return result


def _ensure_curator_config(root: Path) -> Path:
    config_home = root / "opencode-config"
    agent_dir = config_home / "opencode" / "agent"
    agent_dir.mkdir(parents=True, exist_ok=True)
    agent_path = agent_dir / f"{CURATOR_AGENT_NAME}.md"
    if not agent_path.is_file() or agent_path.read_text(encoding="utf-8") != CURATOR_AGENT_MARKDOWN:
        temporary = agent_path.with_suffix(".md.tmp")
        temporary.write_text(CURATOR_AGENT_MARKDOWN, encoding="utf-8")
        temporary.replace(agent_path)
    return config_home


def review_batch(
    batch_index: int,
    batch: Sequence[Mapping[str, Any]],
    root: Path,
    *,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> Path:
    destination = root / "batches" / f"batch-{batch_index:05d}.json"
    expected_ids = [str(row["id"]) for row in batch]
    if destination.is_file():
        payload = json.loads(destination.read_text(encoding="utf-8"))
        if payload.get("ids") != expected_ids:
            raise RuntimeError(f"existing review batch {batch_index} IDs drifted")
        _validate_response(json.dumps(payload["decisions"]), expected_ids)
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    request_file = root / "requests" / f"batch-{batch_index:05d}.json"
    request_file.parent.mkdir(parents=True, exist_ok=True)
    request_file.write_text(json.dumps(_review_payload(batch), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    message = SYSTEM_PROMPT + "\n\nReview the attached JSON array."
    command = [
        "opencode", "run", message,
        "--agent", CURATOR_AGENT_NAME,
        "--pure",
        "--model", MODEL,
        "--variant", VARIANT,
        "--format", "json",
        "--file", str(request_file),
    ]
    isolated_data_home = root / "opencode-data" / f"batch-{batch_index:05d}"
    isolated_opencode_dir = isolated_data_home / "opencode"
    isolated_opencode_dir.mkdir(parents=True, exist_ok=True)
    source_auth = Path.home() / ".local" / "share" / "opencode" / "auth.json"
    destination_auth = isolated_opencode_dir / "auth.json"
    if not destination_auth.is_file():
        if not source_auth.is_file():
            raise RuntimeError(f"OpenCode auth file is missing: {source_auth}")
        shutil.copy2(source_auth, destination_auth)
    process_env = os.environ.copy()
    process_env["XDG_DATA_HOME"] = str(isolated_data_home)
    process_env["XDG_CONFIG_HOME"] = str(root / "opencode-config")
    last_error: Exception | None = None
    decisions: list[dict[str, str]] | None = None
    for attempt in range(1, max_attempts + 1):
        completed = None
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                env=process_env,
                timeout=DEFAULT_OPENCODE_TIMEOUT_SECONDS,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"OpenCode exited {completed.returncode}: {completed.stderr[-1200:]}"
                )
            assistant_text = _parse_json_events(completed.stdout)
            decisions = _validate_response(assistant_text, expected_ids)
            break
        except Exception as error:
            last_error = error
            attempt_path = root / "attempts" / f"batch-{batch_index:05d}-attempt-{attempt:02d}.json"
            attempt_path.parent.mkdir(parents=True, exist_ok=True)
            stderr_tail = (
                completed.stderr[-2000:]
                if completed is not None and isinstance(completed.stderr, str)
                else ""
            )
            attempt_path.write_text(
                json.dumps(
                    {
                        "batch_index": batch_index,
                        "attempt": attempt,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "stderr_tail": stderr_tail,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
    if decisions is None:
        raise RuntimeError(f"OpenCode review batch {batch_index} failed after {max_attempts} attempts") from last_error
    payload = {
        "schema": SCHEMA,
        "batch_index": batch_index,
        "model": MODEL,
        "variant": VARIANT,
        "ids": expected_ids,
        "decisions": decisions,
        "request_sha256": _sha256(request_file),
    }
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return destination


def assemble(candidates: Path, root: Path, output: Path) -> dict[str, Any]:
    rows = list(_read_jsonl(candidates))
    expected = [str(row["id"]) for row in rows]
    decisions: dict[str, dict[str, str]] = {}
    for path in sorted((root / "batches").glob("batch-*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for item in payload["decisions"]:
            item_id = item["id"]
            if item_id in decisions:
                raise RuntimeError(f"duplicate review decision for {item_id}")
            decisions[item_id] = item
    missing = [item_id for item_id in expected if item_id not in decisions]
    extra = sorted(set(decisions) - set(expected))
    if missing or extra:
        raise RuntimeError(f"curation is incomplete: missing={len(missing)} extra={len(extra)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    counts = {decision: 0 for decision in sorted(DECISIONS)}
    with temporary.open("w", encoding="utf-8") as handle:
        for item_id in expected:
            item = decisions[item_id]
            counts[item["decision"]] += 1
            handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(output)
    return {"records": len(expected), "counts": counts, "output": str(output), "sha256": _sha256(output)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--max-batch-chars", type=int)
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.workers <= 0 or args.max_attempts <= 0:
        raise SystemExit("batch-size, workers, and max-attempts must be positive")
    if args.max_batch_chars is not None and args.max_batch_chars <= 0:
        raise SystemExit("max-batch-chars must be positive")

    rows = list(_read_jsonl(args.candidates))
    batches = list(_batches(rows, args.batch_size, max_chars=args.max_batch_chars))
    if args.max_batches is not None:
        batches = batches[: args.max_batches]
    args.work_dir.mkdir(parents=True, exist_ok=True)
    _ensure_curator_config(args.work_dir)
    failures: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                review_batch,
                index,
                batch,
                args.work_dir,
                max_attempts=args.max_attempts,
            ): index
            for index, batch in batches
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                path = future.result()
            except Exception as error:
                failures.append(
                    {
                        "batch_index": index,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
                print(
                    f"[superior-review] batch={index} failed={type(error).__name__}: {error}",
                    flush=True,
                )
                continue
            print(f"[superior-review] batch={index} saved={path}", flush=True)

    total_batches = len(list(_batches(rows, args.batch_size, max_chars=args.max_batch_chars)))
    completed_batches = len(list((args.work_dir / "batches").glob("batch-*.json")))
    if completed_batches == total_batches:
        result = assemble(args.candidates, args.work_dir, args.output)
        result["failures"] = failures
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            json.dumps(
                {
                    "complete": False,
                    "completed_batches": completed_batches,
                    "total_batches": total_batches,
                    "failures": sorted(failures, key=lambda row: int(row["batch_index"])),
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
