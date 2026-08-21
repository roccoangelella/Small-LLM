#!/usr/bin/env python3
"""Main entry point for R-SFT reasoning-dataset creation.

Architecture:
  source adapters -> normalized fit/over-context streams
  generic over-context adapter -> optional GemRouter-compressed fit rows
  assembler -> one frozen reasoning JSONL at a loss-bearing token budget

Superior Reasoning is the only active source today. Adding another source should
require one new source adapter plus one registry entry here; GemRouter adaptation
and final assembly remain source-agnostic.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Mapping, Sequence

HERE = Path(__file__).resolve().parent
RSFT_DIR = HERE.parent
REPO = RSFT_DIR.parents[1]

PARENT_TRAIN_TARGETS = 2_001_000_448
REASONING_SHARE = 0.90
RETENTION_SHARE = 0.10
HELDOUT_FRACTION = 0.01
DEFAULT_SEED = 17
DEFAULT_BASE_JSONL = REPO / "artifacts/rsft-superior-instruction-r0-expanded/reasoning.jsonl"
BUILD_MANIFEST_SCHEMA = "small-llm-rsft-dataset-build-v1"


def _load(name: str, path: Path) -> ModuleType:
    module_name = f"small_llm_rsft_dataset_build_{name}"
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
over_context = _load("over_context", HERE / "over_context.py")
superior = _load("superior_reasoning", HERE / "sources" / "superior_reasoning.py")

SOURCE_REGISTRY: dict[str, ModuleType] = {
    "superior_reasoning": superior,
}
SOURCE_PRIORITY = {
    "superior_reasoning": 10,
}


def _base_records(path: Path | None) -> list[dict[str, str]]:
    if path is None:
        return []
    records: list[dict[str, str]] = []
    prompts: set[str] = set()
    for line_number, row in enumerate(common.read_jsonl(path), start=1):
        record = common.canonical_rsft_record(row)
        prompt_hash = common.normalized_prompt_hash(record["problem"])
        if prompt_hash in prompts:
            raise RuntimeError(f"base reasoning corpus has duplicate prompt at line {line_number}")
        prompts.add(prompt_hash)
        records.append(record)
    return records


def _base_prompt_hashes(path: Path | None) -> set[str]:
    return {
        common.normalized_prompt_hash(row["problem"])
        for row in _base_records(path)
    }


def _aggregate_source_rows(
    work_dir: Path,
    *,
    base_jsonl: Path | None,
) -> dict[str, object]:
    fit: list[dict[str, Any]] = []
    over: list[dict[str, Any]] = []
    seen = _base_prompt_hashes(base_jsonl)
    duplicate_count = 0
    source_manifests: dict[str, object] = {}

    sources_root = work_dir / "sources"
    for source_name in SOURCE_REGISTRY:
        source_dir = sources_root / source_name
        manifest_path = source_dir / "manifest.json"
        if not manifest_path.is_file():
            continue
        source_manifests[source_name] = json.loads(manifest_path.read_text(encoding="utf-8"))
        priority = SOURCE_PRIORITY[source_name]
        for kind, destination in (("fit.jsonl", fit), ("candidates.jsonl", over)):
            for row in common.read_jsonl(source_dir / kind):
                prompt_hash = common.normalized_prompt_hash(str(row["problem"]))
                if prompt_hash in seen:
                    duplicate_count += 1
                    continue
                seen.add(prompt_hash)
                copied = dict(row)
                copied["global_source_order"] = (
                    priority * 10**15 + int(copied.get("source_order", 0))
                )
                destination.append(copied)

    fit.sort(key=lambda row: (int(row["global_source_order"]), str(row["id"])))
    over.sort(key=lambda row: (int(row["global_source_order"]), str(row["id"])))
    fit_path = work_dir / "fit.jsonl"
    candidates_path = work_dir / "over_context" / "candidates.jsonl"
    common.write_jsonl(fit_path, fit)
    common.write_jsonl(candidates_path, over)
    return {
        "fit_records": len(fit),
        "over_context_records": len(over),
        "cross_source_or_base_duplicates": duplicate_count,
        "fit_sha256": common.sha256_path(fit_path),
        "over_context_sha256": common.sha256_path(candidates_path),
        "source_manifests": source_manifests,
    }


def prepare(
    *,
    work_dir: Path,
    base_jsonl: Path | None,
    superior_stages: Sequence[str],
    superior_stage1_jsonl: Path | None = None,
    superior_stage2_jsonl: Path | None = None,
    progress_every: int = 5_000,
) -> dict[str, object]:
    root = Path(work_dir)
    manifest_path = root / "build.manifest.json"
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"refusing to replace non-empty R-SFT build directory {root}")
    root.mkdir(parents=True, exist_ok=True)

    base_hashes = _base_prompt_hashes(base_jsonl)
    source_paths: dict[str, Path] = {}
    if superior_stage1_jsonl is not None:
        source_paths["stage1"] = superior_stage1_jsonl
    if superior_stage2_jsonl is not None:
        source_paths["stage2"] = superior_stage2_jsonl

    source_result = superior.prepare(
        output_dir=root / "sources" / "superior_reasoning",
        stages=superior_stages,
        base_prompt_hashes=base_hashes,
        source_jsonl_by_stage=source_paths,
        progress_every=progress_every,
    )
    aggregate = _aggregate_source_rows(root, base_jsonl=base_jsonl)
    manifest = {
        "schema": BUILD_MANIFEST_SCHEMA,
        "base_jsonl": str(base_jsonl) if base_jsonl is not None else None,
        "base_sha256": common.sha256_path(base_jsonl) if base_jsonl is not None else None,
        "sources": ["superior_reasoning"],
        "superior_stages": list(superior_stages),
        "source_results": {"superior_reasoning": source_result},
        "aggregate": aggregate,
    }
    common.atomic_json(manifest_path, manifest)
    return {**manifest, "manifest": str(manifest_path)}


def _record_target_tokens(record: Mapping[str, Any]) -> int:
    return common.assistant_target_tokens(
        reasoning=str(record["reasoning"]),
        answer=str(record["answer"]),
    )


def projected_train_reasoning_targets(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int = DEFAULT_SEED,
) -> int:
    """Mirror build_atomic.py's per-group stable 1% validation + 1% test split."""
    grouped: dict[tuple[str, str], list[tuple[bytes, int]]] = defaultdict(list)
    identities: set[str] = set()
    for record in records:
        canonical = common.canonical_rsft_record(record)
        identity = common.stable_conversation_id(canonical)
        if identity in identities:
            raise RuntimeError(f"duplicate reasoning training identity: {identity}")
        identities.add(identity)
        group = (canonical["skill"], canonical["difficulty"])
        grouped[group].append(
            (
                common.stable_key(seed, "production-reasoning-partition", identity),
                _record_target_tokens(canonical),
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
        total += sum(tokens for _, tokens in ordered[2 * heldout :])
    return total


def _requested_targets(percent: float) -> tuple[int, int]:
    if percent <= 0:
        raise ValueError("percent must be positive")
    total = int(PARENT_TRAIN_TARGETS * percent / 100.0)
    return total, round(total * REASONING_SHARE)


def _load_extra_pool(work_dir: Path, *, include_adapted: bool) -> list[dict[str, Any]]:
    rows = [dict(row) for row in common.read_jsonl(work_dir / "fit.jsonl")]
    adapted_path = work_dir / "over_context" / "adapted.jsonl"
    if include_adapted:
        if not adapted_path.is_file():
            raise RuntimeError(
                "--include-adapted requested but over_context/adapted.jsonl is missing"
            )
        rows.extend(dict(row) for row in common.read_jsonl(adapted_path))
    return rows


def assemble(
    *,
    work_dir: Path,
    base_jsonl: Path | None,
    output_jsonl: Path,
    percent: float,
    include_adapted: bool = False,
    seed: int = DEFAULT_SEED,
) -> dict[str, object]:
    base = _base_records(base_jsonl)
    seen = {common.normalized_prompt_hash(row["problem"]) for row in base}
    pool = _load_extra_pool(work_dir, include_adapted=include_adapted)
    pool.sort(
        key=lambda row: (
            int(row.get("global_source_order", row.get("source_order", 0))),
            str(row.get("id", "")),
        )
    )
    extras: list[dict[str, Any]] = []
    collisions = 0
    for row in pool:
        prompt_hash = common.normalized_prompt_hash(str(row["problem"]))
        if prompt_hash in seen:
            collisions += 1
            continue
        seen.add(prompt_hash)
        extras.append(row)

    requested_total, requested_reasoning = _requested_targets(percent)
    if not base:
        raise RuntimeError(
            "current percentage assembly requires a frozen reasoning base; "
            "use --base-jsonl (the 16,716-row Stage-1-expanded corpus for the 1% run)"
        )
    base_targets = projected_train_reasoning_targets(base, seed=seed)
    if base_targets >= requested_reasoning:
        raise RuntimeError(
            f"base already projects {base_targets:,} reasoning train targets, "
            f"meeting/exceeding requested {requested_reasoning:,}"
        )
    if not extras:
        raise RuntimeError("prepared source pool contains no additional fit rows")

    def measure(prefix: int) -> int:
        return projected_train_reasoning_targets(
            [*base, *(common.canonical_rsft_record(row) for row in extras[:prefix])],
            seed=seed,
        )

    maximum = measure(len(extras))
    if maximum < requested_reasoning:
        pending = work_dir / "over_context" / "candidates.jsonl"
        raise RuntimeError(
            f"context-fit pool reaches only {maximum:,} reasoning train targets; "
            f"need {requested_reasoning:,}. Curate {pending}, run the generic "
            "over-context GemRouter adapter, finalize adapted.jsonl, then rebuild "
            "with --include-adapted."
        )

    lo, hi = 0, len(extras)
    while lo < hi:
        mid = (lo + hi) // 2
        if measure(mid) >= requested_reasoning:
            hi = mid
        else:
            lo = mid + 1
    chosen = lo
    for prefix in range(max(0, lo - 256), lo + 1):
        if measure(prefix) >= requested_reasoning:
            chosen = prefix
            break
    chosen_targets = measure(chosen)
    final_records = [
        *base,
        *(common.canonical_rsft_record(row) for row in extras[:chosen]),
    ]

    final_records.sort(
        key=lambda row: common.stable_key(
            seed, "final-reasoning-jsonl", common.stable_conversation_id(row)
        )
    )
    common.write_jsonl(output_jsonl, final_records)
    retention = max(1, round(chosen_targets * RETENTION_SHARE / REASONING_SHARE))
    projected_total = chosen_targets + retention
    manifest = {
        "schema": "small-llm-rsft-reasoning-corpus-v1",
        "percent_of_parent_requested": percent,
        "parent_train_targets": PARENT_TRAIN_TARGETS,
        "requested_total_train_targets": requested_total,
        "requested_reasoning_train_targets": requested_reasoning,
        "base_projected_reasoning_train_targets": base_targets,
        "projected_reasoning_train_targets": chosen_targets,
        "projected_retention_train_targets": retention,
        "projected_total_train_targets": projected_total,
        "projected_percent_of_parent": projected_total / PARENT_TRAIN_TARGETS * 100.0,
        "base_rows": len(base),
        "additional_rows": chosen,
        "combined_rows": len(final_records),
        "prepared_pool_rows": len(extras),
        "prompt_collisions_excluded": collisions,
        "include_adapted": include_adapted,
        "seed": seed,
        "output_jsonl": str(output_jsonl),
        "output_sha256": common.sha256_path(output_jsonl),
        "output_byte_size": output_jsonl.stat().st_size,
    }
    common.atomic_json(output_jsonl.with_suffix(output_jsonl.suffix + ".manifest.json"), manifest)
    return manifest


def build(
    *,
    work_dir: Path,
    base_jsonl: Path,
    output_jsonl: Path,
    percent: float,
    superior_stages: Sequence[str],
    superior_stage1_jsonl: Path | None = None,
    superior_stage2_jsonl: Path | None = None,
    include_adapted: bool = False,
    progress_every: int = 5_000,
) -> dict[str, object]:
    manifest = work_dir / "build.manifest.json"
    if not manifest.is_file():
        prepare(
            work_dir=work_dir,
            base_jsonl=base_jsonl,
            superior_stages=superior_stages,
            superior_stage1_jsonl=superior_stage1_jsonl,
            superior_stage2_jsonl=superior_stage2_jsonl,
            progress_every=progress_every,
        )
    return assemble(
        work_dir=work_dir,
        base_jsonl=base_jsonl,
        output_jsonl=output_jsonl,
        percent=percent,
        include_adapted=include_adapted,
    )


def _stages(value: str) -> tuple[str, ...]:
    stages = tuple(part.strip() for part in value.split(",") if part.strip())
    if not stages:
        raise argparse.ArgumentTypeError("at least one stage is required")
    unknown = [stage for stage in stages if stage not in superior.STAGES]
    if unknown:
        raise argparse.ArgumentTypeError(f"unsupported Superior stages: {unknown}")
    return stages


def _add_source_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--superior-stages",
        type=_stages,
        default=("stage2",),
        help="comma-separated Superior stages; current 1%% default is stage2 because Stage1 is frozen in --base-jsonl",
    )
    parser.add_argument("--superior-stage1-jsonl", type=Path)
    parser.add_argument("--superior-stage2-jsonl", type=Path)
    parser.add_argument("--progress-every", type=int, default=5_000)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--work-dir", type=Path, required=True)
    prepare_parser.add_argument("--base-jsonl", type=Path, default=DEFAULT_BASE_JSONL)
    _add_source_args(prepare_parser)

    assemble_parser = sub.add_parser("assemble")
    assemble_parser.add_argument("--work-dir", type=Path, required=True)
    assemble_parser.add_argument("--base-jsonl", type=Path, default=DEFAULT_BASE_JSONL)
    assemble_parser.add_argument("--output-jsonl", type=Path, required=True)
    assemble_parser.add_argument("--percent", type=float, choices=(1.0, 2.0, 4.0), required=True)
    assemble_parser.add_argument("--include-adapted", action="store_true")

    build_parser_ = sub.add_parser("build")
    build_parser_.add_argument("--work-dir", type=Path, required=True)
    build_parser_.add_argument("--base-jsonl", type=Path, default=DEFAULT_BASE_JSONL)
    build_parser_.add_argument("--output-jsonl", type=Path, required=True)
    build_parser_.add_argument("--percent", type=float, choices=(1.0, 2.0, 4.0), required=True)
    build_parser_.add_argument("--include-adapted", action="store_true")
    _add_source_args(build_parser_)

    adapt_prepare = sub.add_parser("adapt-prepare")
    adapt_prepare.add_argument("--work-dir", type=Path, required=True)
    adapt_prepare.add_argument("--curation-jsonl", type=Path, required=True)
    adapt_prepare.add_argument("--batch-size", type=int, default=over_context.MAX_BATCH_SIZE)

    adapt_wave = sub.add_parser("adapt-wave")
    adapt_wave.add_argument("--work-dir", type=Path, required=True)
    adapt_wave.add_argument("--first-batch", type=int, required=True)
    adapt_wave.add_argument("--batch-count", type=int, required=True)
    adapt_wave.add_argument("--workers", type=int, default=4)
    adapt_wave.add_argument("--max-attempts", type=int, default=over_context.DEFAULT_MAX_ATTEMPTS)

    adapt_status = sub.add_parser("adapt-status")
    adapt_status.add_argument("--work-dir", type=Path, required=True)

    adapt_finalize = sub.add_parser("adapt-finalize")
    adapt_finalize.add_argument("--work-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare(
            work_dir=args.work_dir,
            base_jsonl=args.base_jsonl,
            superior_stages=args.superior_stages,
            superior_stage1_jsonl=args.superior_stage1_jsonl,
            superior_stage2_jsonl=args.superior_stage2_jsonl,
            progress_every=args.progress_every,
        )
    elif args.command == "assemble":
        result = assemble(
            work_dir=args.work_dir,
            base_jsonl=args.base_jsonl,
            output_jsonl=args.output_jsonl,
            percent=args.percent,
            include_adapted=args.include_adapted,
        )
    elif args.command == "build":
        result = build(
            work_dir=args.work_dir,
            base_jsonl=args.base_jsonl,
            output_jsonl=args.output_jsonl,
            percent=args.percent,
            superior_stages=args.superior_stages,
            superior_stage1_jsonl=args.superior_stage1_jsonl,
            superior_stage2_jsonl=args.superior_stage2_jsonl,
            include_adapted=args.include_adapted,
            progress_every=args.progress_every,
        )
    elif args.command == "adapt-prepare":
        adaptation_root = args.work_dir / "over_context" / "adaptation"
        result = over_context.prepare_keep(
            candidates_jsonl=args.work_dir / "over_context" / "candidates.jsonl",
            curation_jsonl=args.curation_jsonl,
            work_dir=adaptation_root,
            batch_size=args.batch_size,
        )
    elif args.command == "adapt-wave":
        result = over_context.adapt_wave(
            args.work_dir / "over_context" / "adaptation",
            first_batch=args.first_batch,
            batch_count=args.batch_count,
            workers=args.workers,
            max_attempts=args.max_attempts,
        )
    elif args.command == "adapt-status":
        result = over_context.status(args.work_dir / "over_context" / "adaptation")
    else:
        output = args.work_dir / "over_context" / "adapted.jsonl"
        result = over_context.finalize(
            args.work_dir / "over_context" / "adaptation",
            output_jsonl=output,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
