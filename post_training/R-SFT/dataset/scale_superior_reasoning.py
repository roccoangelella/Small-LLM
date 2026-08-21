#!/usr/bin/env python3
"""Scale the frozen R-SFT reasoning corpus with Superior Reasoning Stage 2.

The Stage-1 production/reproduction path stays immutable.  This module adds a
scaling lane that applies the same accepted R0 contract to Stage-2
``instruction_following`` examples:

- strict ``<think>...</think>`` teacher-output parsing;
- normalized-prompt deduplication against the frozen Stage-1-expanded corpus;
- the same primary-math / primary-code exclusion policy;
- reserved reasoning-marker rejection;
- exact 2,048-token atomic chat fit validation;
- over-context rows frozen in the same candidate schema consumed by the
  existing Variant-D GemRouter adapter;
- deterministic 1% / 2% / 4% corpus assembly by *loss-bearing reasoning target
  tokens*, with the existing 1%/1% per-group held-out policy projected exactly.

For Stage 2, the preferred transport is Hugging Face ``datasets`` streaming of
config ``stage2``, split ``instruction_following``.  ``--source-jsonl`` can be
used to run the identical logic against a previously downloaded Stage-2 JSONL.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable, Iterator, Mapping, Sequence

HERE = Path(__file__).resolve().parent
RSFT_DIR = HERE.parent
REPO = RSFT_DIR.parents[1]

# Import the frozen Stage-1 policy implementation rather than duplicating it.
import importlib.util
import sys


def _load_module(name: str, path: Path):
    module_name = f"small_llm_rsft_scale_{name}"
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

STAGE2_CONFIG = "stage2"
STAGE2_SPLIT = "instruction_following"
STAGE2_FILENAME = "Superior-Reasoning-SFT-gpt-oss-120b-stage2-train-data.jsonl"
SCALING_SCHEMA = "small-llm-superior-stage2-scaling-v1"
CANDIDATE_SCHEMA = "small-llm-superior-overcontext-candidates-v1"
PARENT_TRAIN_TARGETS = 2_001_000_448
REASONING_SHARE = 0.90
RETENTION_SHARE = 0.10
HELDOUT_FRACTION = 0.01
SEED = 17
FIVE_FIELDS = ("skill", "difficulty", "problem", "reasoning", "answer")


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise RuntimeError(f"blank JSONL row at {path}:{line_number}")
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"invalid JSON at {path}:{line_number}") from error
            if not isinstance(raw, dict):
                raise RuntimeError(f"JSONL row must be an object at {path}:{line_number}")
            yield raw


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _normalize_source_row(raw: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    """Normalize either raw JSONL or HF split rows to the Stage-1 row contract."""
    row = dict(raw)
    row.setdefault("domain", superior.PRODUCTION_DOMAIN)
    if row.get("domain") != superior.PRODUCTION_DOMAIN:
        raise ValueError(
            f"Stage-2 scaling source must be instruction_following; row {index} has {row.get('domain')!r}"
        )
    return row


def iter_stage2_instruction_rows(*, source_jsonl: Path | None = None) -> Iterator[Mapping[str, Any]]:
    if source_jsonl is not None:
        for index, raw in enumerate(_read_jsonl(source_jsonl)):
            yield _normalize_source_row(raw, index=index)
        return

    try:
        from datasets import load_dataset
    except ImportError as error:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "Stage-2 streaming requires `datasets`; install it or pass --source-jsonl"
        ) from error

    stream = load_dataset(
        superior.DATASET_ID,
        STAGE2_CONFIG,
        split=STAGE2_SPLIT,
        streaming=True,
    )
    for index, raw in enumerate(stream):
        if not isinstance(raw, Mapping):
            raise RuntimeError(f"Stage-2 dataset row {index} is not an object")
        yield _normalize_source_row(raw, index=index)


def _base_prompt_hashes(base_jsonl: Path) -> tuple[set[str], int]:
    hashes: set[str] = set()
    rows = 0
    for line_number, raw in enumerate(_read_jsonl(base_jsonl), start=1):
        if set(raw) != set(FIVE_FIELDS):
            raise RuntimeError(f"base reasoning row {line_number} has the wrong schema")
        prompt = str(raw["problem"]).strip()
        identity = superior._normalized_input_hash(prompt)
        if identity in hashes:
            raise RuntimeError("base reasoning corpus contains duplicate normalized prompts")
        hashes.add(identity)
        rows += 1
    if not rows:
        raise RuntimeError("base reasoning corpus is empty")
    return hashes, rows


def _assistant_target_tokens(*, reasoning: str, answer: str, token_counter=None) -> int:
    """Exact atomic assistant loss-bearing targets: 3 markers + text + EOS."""
    count = token_counter or superior._default_token_counter()
    return 3 + count(reasoning) + count(answer) + 1


def prepare_stage2(
    *,
    base_jsonl: Path,
    work_dir: Path,
    source_jsonl: Path | None = None,
    max_source_rows: int | None = None,
    progress_every: int = 5_000,
) -> dict[str, object]:
    """Freeze Stage-2 fit rows and over-context candidates under the Stage-1 policy."""
    work_dir.mkdir(parents=True, exist_ok=True)
    fit_path = work_dir / "fit.jsonl"
    candidates_path = work_dir / "candidates.jsonl"
    manifest_path = work_dir / "candidates.manifest.json"
    if any(path.exists() for path in (fit_path, candidates_path, manifest_path)):
        raise FileExistsError("refusing to replace existing Stage-2 scaling preparation")

    base_hashes, base_rows = _base_prompt_hashes(base_jsonl)
    seen = set(base_hashes)
    token_counter = superior._default_token_counter()
    source_rows = 0
    accepted_fit = 0
    over_context = 0
    duplicate_count = 0
    rejected_output_count = 0
    exclusion_counts: Counter[str] = Counter()
    fit_target_tokens = 0

    fit_tmp = fit_path.with_suffix(".jsonl.tmp")
    candidates_tmp = candidates_path.with_suffix(".jsonl.tmp")
    with fit_tmp.open("w", encoding="utf-8") as fit_handle, candidates_tmp.open(
        "w", encoding="utf-8"
    ) as candidate_handle:
        for source_index, raw in enumerate(
            iter_stage2_instruction_rows(source_jsonl=source_jsonl)
        ):
            if max_source_rows is not None and source_rows >= max_source_rows:
                break
            source_rows += 1
            if progress_every and source_rows % progress_every == 0:
                print(
                    f"[stage2:prepare] source={source_rows} fit={accepted_fit} "
                    f"over_context={over_context} duplicates={duplicate_count}",
                    flush=True,
                )
            row = _normalize_source_row(raw, index=source_index)
            problem = superior._row_text(row, "input", index=source_index)
            source_id = superior._row_text(row, "uuid", index=source_index)
            output = superior._row_text(row, "output", index=source_index)
            try:
                parsed = superior.parse_teacher_output(output)
            except ValueError:
                rejected_output_count += 1
                continue

            prompt_hash = superior._normalized_input_hash(problem)
            if prompt_hash in seen:
                duplicate_count += 1
                continue
            seen.add(prompt_hash)

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
                token_counter=token_counter,
            )
            common = {
                "id": source_id,
                "source_index": source_index,
                "domain": superior.PRODUCTION_DOMAIN,
                "difficulty": superior.PRODUCTION_DIFFICULTY,
                "problem": problem,
                "reasoning": parsed.reasoning,
                "answer": parsed.answer,
                "serialized_token_count": serialized_tokens,
            }
            if serialized_tokens <= superior.PRODUCTION_CONTEXT_LENGTH:
                target_tokens = _assistant_target_tokens(
                    reasoning=parsed.reasoning,
                    answer=parsed.answer,
                    token_counter=token_counter,
                )
                common["target_token_count"] = target_tokens
                fit_handle.write(json.dumps(common, ensure_ascii=False, separators=(",", ":")) + "\n")
                fit_target_tokens += target_tokens
                accepted_fit += 1
            else:
                exclusion_counts["over_context"] += 1
                candidate = {
                    **common,
                    "difficulty": "simplified_fit",
                    "original_serialized_tokens": serialized_tokens,
                }
                candidate.pop("serialized_token_count", None)
                candidate_handle.write(
                    json.dumps(candidate, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
                over_context += 1

    fit_tmp.replace(fit_path)
    candidates_tmp.replace(candidates_path)
    manifest = {
        "schema": CANDIDATE_SCHEMA,
        "scaling_schema": SCALING_SCHEMA,
        "dataset_id": superior.DATASET_ID,
        "dataset_config": STAGE2_CONFIG,
        "dataset_split": STAGE2_SPLIT,
        "dataset_revision": superior.DATASET_REVISION,
        "dataset_filename": STAGE2_FILENAME,
        "filter_policy": superior.PRODUCTION_FILTER_VERSION,
        "context_length": superior.PRODUCTION_CONTEXT_LENGTH,
        "prompt_sha256": _sha256_text(superior.SIMPLIFICATION_SYSTEM_PROMPT),
        "batch_size_max": superior.SIMPLIFICATION_MAX_BATCH_SIZE,
        "base_jsonl_sha256": _sha256_path(base_jsonl),
        "base_rows": base_rows,
        "source_rows": source_rows,
        "fit_unchanged_rows": accepted_fit,
        "fit_target_tokens_before_partition": fit_target_tokens,
        "over_context_rows": over_context,
        "duplicate_or_base_collision_count": duplicate_count,
        "rejected_output_count": rejected_output_count,
        "exclusion_counts": dict(sorted(exclusion_counts.items())),
        "fit_jsonl": {
            "path": fit_path.name,
            "sha256": _sha256_path(fit_path),
            "byte_size": fit_path.stat().st_size,
            "records": accepted_fit,
        },
        "candidates_jsonl": {
            "path": candidates_path.name,
            "sha256": _sha256_path(candidates_path),
            "byte_size": candidates_path.stat().st_size,
            "records": over_context,
        },
    }
    _atomic_json(manifest_path, manifest)
    return {**manifest, "manifest": str(manifest_path)}


def _rsft_record(raw: Mapping[str, Any]) -> dict[str, str]:
    record = superior.to_rsft_mapping(raw)
    return {field: record[field] for field in FIVE_FIELDS}


def _stable_conversation_id(record: Mapping[str, str]) -> str:
    payload = "\x1f".join(record[field] for field in FIVE_FIELDS).encode("utf-8")
    return f"rsft-{hashlib.sha256(payload).hexdigest()[:20]}"


def _stable_key(seed: int, label: str, identity: str) -> bytes:
    return hashlib.sha256(f"{seed}:{label}:{identity}".encode("utf-8")).digest()


def _record_target_tokens(record: Mapping[str, str], token_counter) -> int:
    return _assistant_target_tokens(
        reasoning=record["reasoning"], answer=record["answer"], token_counter=token_counter
    )


def projected_train_reasoning_targets(
    records: Sequence[Mapping[str, str]], *, seed: int = SEED
) -> int:
    """Mirror build_atomic.py's heterogeneous 1%/1% held-out partition exactly."""
    token_counter = superior._default_token_counter()
    grouped: dict[tuple[str, str], list[tuple[bytes, int]]] = defaultdict(list)
    identities: set[str] = set()
    for record in records:
        identity = _stable_conversation_id(record)
        if identity in identities:
            raise RuntimeError(f"duplicate R-SFT training identity: {identity}")
        identities.add(identity)
        group = (record["skill"], record["difficulty"])
        grouped[group].append(
            (
                _stable_key(seed, "production-reasoning-partition", identity),
                _record_target_tokens(record, token_counter),
            )
        )

    total = 0
    for group, values in grouped.items():
        if len(values) < 3:
            raise RuntimeError(f"reasoning group {group!r} needs at least three records")
        heldout = max(1, round(len(values) * HELDOUT_FRACTION))
        if 2 * heldout >= len(values):
            raise RuntimeError(f"reasoning group {group!r} is too small for heldout")
        ordered = sorted(values, key=lambda item: item[0])
        total += sum(targets for _, targets in ordered[2 * heldout :])
    return total


def _load_base_records(path: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    prompts: set[str] = set()
    for line_number, raw in enumerate(_read_jsonl(path), start=1):
        if set(raw) != set(FIVE_FIELDS):
            raise RuntimeError(f"base row {line_number} has the wrong schema")
        record = {field: str(raw[field]).strip() for field in FIVE_FIELDS}
        normalized = superior._normalized_input_hash(record["problem"])
        if normalized in prompts:
            raise RuntimeError("base reasoning corpus has duplicate normalized prompts")
        prompts.add(normalized)
        records.append(record)
    return records


def _load_stage2_pool(work_dir: Path, *, include_adapted: bool) -> list[tuple[str, dict[str, str]]]:
    pool: list[tuple[str, dict[str, str]]] = []
    for raw in _read_jsonl(work_dir / "fit.jsonl"):
        pool.append((str(raw["id"]), _rsft_record(raw)))
    adapted_path = work_dir / "adapted.jsonl"
    if include_adapted and adapted_path.is_file():
        for raw in _read_jsonl(adapted_path):
            pool.append((str(raw["id"]), _rsft_record(raw)))
    return pool


def _requested_targets(percent: float) -> tuple[int, int]:
    if percent <= 0:
        raise ValueError("percent must be positive")
    total = int(PARENT_TRAIN_TARGETS * (percent / 100.0))
    reasoning = round(total * REASONING_SHARE)
    return total, reasoning


def build_percent_corpus(
    *,
    base_jsonl: Path,
    work_dir: Path,
    output_jsonl: Path,
    percent: float,
    include_adapted: bool = False,
    seed: int = SEED,
) -> dict[str, object]:
    base = _load_base_records(base_jsonl)
    base_prompts = {superior._normalized_input_hash(row["problem"]) for row in base}
    raw_pool = _load_stage2_pool(work_dir, include_adapted=include_adapted)

    seen = set(base_prompts)
    deduped: list[tuple[str, dict[str, str]]] = []
    collision_ids: list[str] = []
    for source_id, record in raw_pool:
        normalized = superior._normalized_input_hash(record["problem"])
        if normalized in seen:
            collision_ids.append(source_id)
            continue
        seen.add(normalized)
        deduped.append((source_id, record))
    deduped.sort(
        key=lambda item: hashlib.sha256(f"{seed}\x1f{item[0]}".encode("utf-8")).digest()
    )

    requested_total, requested_reasoning = _requested_targets(percent)
    base_reasoning = projected_train_reasoning_targets(base, seed=seed)
    if base_reasoning >= requested_reasoning:
        raise RuntimeError(
            f"base corpus already projects {base_reasoning} train reasoning targets, "
            f"which reaches/exceeds the requested {requested_reasoning}"
        )
    if not deduped:
        raise RuntimeError("Stage-2 pool is empty")

    def measure(prefix: int) -> int:
        return projected_train_reasoning_targets(
            [*base, *(record for _, record in deduped[:prefix])], seed=seed
        )

    high_value = measure(len(deduped))
    if high_value < requested_reasoning:
        raise RuntimeError(
            f"prepared Stage-2 pool only reaches {high_value:,} reasoning train targets; "
            f"need {requested_reasoning:,}. Run keeper curation + Variant-D adaptation "
            "for Stage-2 over-context candidates and retry with --include-adapted."
        )

    # Binary-search a near-minimal deterministic prefix. Held-out-count changes can
    # cause tiny local non-monotonicities every ~100 rows, so scan a bounded window
    # behind the crossing and choose the smallest prefix that actually reaches it.
    lo, hi = 0, len(deduped)
    while lo < hi:
        mid = (lo + hi) // 2
        if measure(mid) >= requested_reasoning:
            hi = mid
        else:
            lo = mid + 1
    start = max(0, lo - 128)
    chosen = lo
    chosen_targets = measure(chosen)
    for prefix in range(start, min(len(deduped), lo + 8) + 1):
        value = measure(prefix)
        if value >= requested_reasoning:
            chosen = prefix
            chosen_targets = value
            break

    combined = [*base, *(record for _, record in deduped[:chosen])]
    random.Random(seed).shuffle(combined)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_jsonl.with_suffix(output_jsonl.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in combined:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(output_jsonl)

    retention = max(1, round(chosen_targets * RETENTION_SHARE / REASONING_SHARE))
    projected_total = chosen_targets + retention
    manifest = {
        "schema": SCALING_SCHEMA,
        "source": "Superior-Reasoning Stage1-expanded + Stage2 instruction_following",
        "stage2_policy": "same-R0-filter-and-atomic-context-contract-as-stage1",
        "adaptation_policy": "ADR-0103-variant-D-fidelity-first when needed",
        "percent_of_parent_requested": percent,
        "parent_train_targets": PARENT_TRAIN_TARGETS,
        "requested_total_train_targets": requested_total,
        "requested_reasoning_train_targets": requested_reasoning,
        "projected_reasoning_train_targets": chosen_targets,
        "projected_retention_train_targets": retention,
        "projected_total_train_targets": projected_total,
        "projected_percent_of_parent": projected_total / PARENT_TRAIN_TARGETS * 100.0,
        "heldout_fraction_per_split": HELDOUT_FRACTION,
        "seed": seed,
        "base_rows": len(base),
        "stage2_rows_added": chosen,
        "combined_rows": len(combined),
        "stage2_pool_rows": len(deduped),
        "normalized_prompt_collisions_excluded": len(collision_ids),
        "normalized_prompt_collision_ids": collision_ids,
        "include_adapted": include_adapted,
        "base_jsonl_sha256": _sha256_path(base_jsonl),
        "output_jsonl": str(output_jsonl),
        "output_sha256": _sha256_path(output_jsonl),
        "output_byte_size": output_jsonl.stat().st_size,
    }
    manifest_path = output_jsonl.with_suffix(output_jsonl.suffix + ".manifest.json")
    _atomic_json(manifest_path, manifest)
    return {**manifest, "manifest": str(manifest_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare-stage2")
    prepare.add_argument("--base-jsonl", type=Path, required=True)
    prepare.add_argument("--work-dir", type=Path, required=True)
    prepare.add_argument("--source-jsonl", type=Path)
    prepare.add_argument("--max-source-rows", type=int)
    prepare.add_argument("--progress-every", type=int, default=5_000)

    build = sub.add_parser("build-percent")
    build.add_argument("--base-jsonl", type=Path, required=True)
    build.add_argument("--work-dir", type=Path, required=True)
    build.add_argument("--output-jsonl", type=Path, required=True)
    build.add_argument("--percent", type=float, choices=(1.0, 2.0, 4.0), required=True)
    build.add_argument("--include-adapted", action="store_true")
    build.add_argument("--seed", type=int, default=SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare-stage2":
        result = prepare_stage2(
            base_jsonl=args.base_jsonl,
            work_dir=args.work_dir,
            source_jsonl=args.source_jsonl,
            max_source_rows=args.max_source_rows,
            progress_every=args.progress_every,
        )
    else:
        result = build_percent_corpus(
            base_jsonl=args.base_jsonl,
            work_dir=args.work_dir,
            output_jsonl=args.output_jsonl,
            percent=args.percent,
            include_adapted=args.include_adapted,
            seed=args.seed,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
