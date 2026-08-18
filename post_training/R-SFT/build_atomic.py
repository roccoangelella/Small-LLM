#!/usr/bin/env python3
"""Build one canonical atomic-only R-SFT bundle for production training.

This is intentionally separate from the historical matched delimiter-ablation
builder. Production R-SFT has one serialization contract: <think>, </think>,
<answer> mapped atomically to IDs 50257, 50258, and 50259.
"""
from __future__ import annotations

from collections import Counter
import argparse
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Mapping, Sequence

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

CANONICAL_TOKEN_SPEC = Path(__file__).with_name("reasoning-tokens.json")
CANONICAL_MARKERS = ("<think>", "</think>", "<answer>")
DEFAULT_OPTIMIZER_TARGET_TOKENS = 32_768
DEFAULT_CONTEXT_LENGTH = 2_048
DEFAULT_SEED = 17


def _load_bundle() -> ModuleType:
    path = Path(__file__).with_name("bundle.py")
    spec = importlib.util.spec_from_file_location("small_llm_rsft_atomic_bundle", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load R-SFT bundle module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bundle = _load_bundle()


def _canonical_token_spec():
    spec = bundle.load_reasoning_token_spec(CANONICAL_TOKEN_SPEC)
    actual = (spec.reasoning_start, spec.reasoning_end, spec.answer_start)
    if actual != CANONICAL_MARKERS:
        raise RuntimeError(
            f"canonical production token spec drifted: expected {CANONICAL_MARKERS}, got {actual}"
        )
    if spec.special_tokens != {"<think>": 50_257, "</think>": 50_258, "<answer>": 50_259}:
        raise RuntimeError("canonical production reasoning-token IDs drifted")
    return spec


def _infer_examples_per_cell(records: Sequence[object]) -> int:
    counts = Counter((record.skill, record.difficulty) for record in records)
    expected = {
        (skill, difficulty)
        for skill in bundle.prompts.R0_SKILLS
        for difficulty in bundle.generation.R0_DIFFICULTIES
    }
    if set(counts) != expected:
        missing = sorted(expected - set(counts))
        extra = sorted(set(counts) - expected)
        raise ValueError(f"production R0 matrix is incomplete: missing={missing}, extra={extra}")
    sizes = set(counts.values())
    if len(sizes) != 1:
        raise ValueError(f"production R0 matrix must remain uniform by record count: {dict(counts)}")
    return next(iter(sizes))


def _retention_target(reasoning_targets: int) -> int:
    if reasoning_targets <= 0:
        raise ValueError("reasoning_targets must be positive")
    return max(
        1,
        round(
            reasoning_targets
            * bundle.mixture.RETENTION_SHARE
            / bundle.mixture.REASONING_SHARE
        ),
    )


def build_atomic_production_bundle(
    reasoning_jsonl: Path | str,
    *,
    s0_bundle: Path | str,
    output_dir: Path | str,
    heldout_per_cell: int,
    optimizer_target_tokens: int = DEFAULT_OPTIMIZER_TARGET_TOKENS,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    seed: int = DEFAULT_SEED,
) -> dict[str, object]:
    reasoning_path = Path(reasoning_jsonl).expanduser().resolve()
    source_bundle = Path(s0_bundle).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if not reasoning_path.is_file() or reasoning_path.is_symlink():
        raise RuntimeError(f"reasoning JSONL is missing or unsafe: {reasoning_path}")
    if not source_bundle.is_dir():
        raise RuntimeError(f"S0 bundle directory is missing: {source_bundle}")
    if output.exists():
        raise FileExistsError(f"refusing to replace existing production R-SFT bundle: {output}")
    if heldout_per_cell <= 0:
        raise ValueError("heldout_per_cell must be positive")
    if optimizer_target_tokens <= 0 or context_length <= 0:
        raise ValueError("optimizer_target_tokens and context_length must be positive")

    records = bundle.schema.read_jsonl(reasoning_path)
    examples_per_cell = _infer_examples_per_cell(records)
    cells = bundle.validate_reasoning_matrix(records, examples_per_cell=examples_per_cell)
    partition = bundle.partition_reasoning_records(
        records,
        heldout_per_cell=heldout_per_cell,
        seed=seed,
    )
    token_spec = _canonical_token_spec()
    bundle._assert_no_atomic_marker_collision(records, token_spec)

    s0_verification = bundle.verify_bundle(source_bundle)
    s0_manifest = bundle._read_bundle_manifest(source_bundle)
    retention_source_shares = bundle._instruction_source_shares(s0_manifest)

    atomic_train = bundle._tokenize_reasoning_split(
        partition["train"],
        split="train",
        arm="atomic",
        token_spec=token_spec,
        context_length=context_length,
    )
    reasoning_targets = sum(record.target_token_count for record in atomic_train)
    retention_requested = _retention_target(reasoning_targets)
    retention_records = bundle.select_s0_retention_records(
        bundle.iter_tokenized_bundle_records(source_bundle, split="train"),
        source_shares=retention_source_shares,
        target_tokens=retention_requested,
    )
    requested_train_source_shares = bundle.mixture.build_rsft_source_shares(
        retention_source_shares
    )
    source_manifest = bundle._source_manifest(
        reasoning_path=reasoning_path,
        reasoning_records=records,
        cells=cells,
        partition=partition,
        s0_verification=s0_verification,
        s0_source_shares=retention_source_shares,
        retention_records=retention_records,
        retention_target_requested=retention_requested,
        heldout_per_cell=heldout_per_cell,
        seed=seed,
    )

    result = bundle._build_arm(
        output,
        arm="atomic",
        partition=partition,
        token_spec=token_spec,
        retention_records=retention_records,
        requested_train_source_shares=requested_train_source_shares,
        source_manifest=source_manifest,
        optimizer_target_tokens=optimizer_target_tokens,
        context_length=context_length,
        seed=seed,
    )

    # _build_arm is shared with the historical pilot and therefore labels its
    # prepared source as a pilot. Rewrite only production metadata, recompute the
    # canonical manifest identity, then verify again before returning.
    manifest_path = output / "bundle-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise RuntimeError("production R-SFT bundle manifest is malformed")
    prepared = manifest.get("prepared_source")
    rsft = manifest.get("rsft")
    if not isinstance(prepared, dict) or not isinstance(rsft, dict):
        raise RuntimeError("production R-SFT bundle metadata is malformed")
    prepared["dataset_name"] = "small-llm-rsft-r0"
    rsft["contract"] = "atomic-production-v1"
    rsft["delimiter_format"] = "atomic"
    without_hash = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    manifest["manifest_sha256"] = bundle.canonical_hash(without_hash)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    verification = bundle.verify_bundle(output)

    retention_targets = sum(record.target_token_count for record in retention_records)
    train_targets = int(manifest["train_target_tokens_requested"])
    return {
        "schema": "small-llm-rsft-atomic-production-build-v1",
        "bundle": str(output),
        "bundle_manifest_sha256": verification["bundle_manifest_sha256"],
        "reasoning_jsonl": str(reasoning_path),
        "examples_per_cell": examples_per_cell,
        "records": len(records),
        "partition_records": {name: len(values) for name, values in partition.items()},
        "reasoning_train_target_tokens": reasoning_targets,
        "retention_train_target_tokens": retention_targets,
        "realized_retention_share": retention_targets / train_targets,
        "train_target_tokens": train_targets,
        "optimizer_target_tokens": optimizer_target_tokens,
        "passes": 1,
        "delimiter_format": "atomic",
        "reasoning_tokens": token_spec.to_metadata(),
    }


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reasoning-jsonl", type=Path, required=True)
    parser.add_argument("--s0-bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--heldout-per-cell",
        type=_positive_int,
        required=True,
        help="explicit production validation/test count per skill x difficulty cell",
    )
    parser.add_argument(
        "--optimizer-target-tokens",
        type=_positive_int,
        default=DEFAULT_OPTIMIZER_TARGET_TOKENS,
    )
    parser.add_argument("--context-length", type=_positive_int, default=DEFAULT_CONTEXT_LENGTH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_atomic_production_bundle(
        args.reasoning_jsonl,
        s0_bundle=args.s0_bundle,
        output_dir=args.output_dir,
        heldout_per_cell=args.heldout_per_cell,
        optimizer_target_tokens=args.optimizer_target_tokens,
        context_length=args.context_length,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CANONICAL_MARKERS",
    "CANONICAL_TOKEN_SPEC",
    "DEFAULT_CONTEXT_LENGTH",
    "DEFAULT_OPTIMIZER_TARGET_TOKENS",
    "DEFAULT_SEED",
    "build_atomic_production_bundle",
    "build_parser",
    "main",
]
