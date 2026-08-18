"""Production entry point for the first R0 reasoning-SFT pilot dataset.

The default ``pilot`` command performs the whole data-side experiment:

1. generate 30 schema-valid Gemini examples in every R0 skill x difficulty cell;
2. checkpoint each teacher batch so an interrupted 63-call run resumes safely;
3. freeze the globally shuffled 630-example reasoning JSONL;
4. sample the frozen 10% retention lane from the completed S0 instruction bundle;
5. emit matched atomic and textual native SFT bundles for Kaggle training.

No semantic teacher re-judging is added here. Generation acceptance remains the
strict JSON/schema boundary already frozen for R0.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
from types import ModuleType
from typing import Any, Mapping, Sequence


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from trainer.identity import canonical_hash
from post_training.sft.bundle import verify_bundle


def _load_sibling(name: str) -> ModuleType:
    module_name = f"small_llm_rsft_{name}"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load R-SFT sibling module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


bundle = _load_sibling("bundle")
dataset = _load_sibling("dataset")
generation = _load_sibling("generate")
prompts = _load_sibling("prompts")
schema = _load_sibling("schema")

GENERATION_BATCH_SCHEMA = "small-llm-rsft-generation-batch-v1"
GENERATION_MANIFEST_SCHEMA = "small-llm-rsft-generation-manifest-v1"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _batch_path(root: Path, index: int, request: Any) -> Path:
    return root / "batches" / f"{index:03d}-{request.skill}-{request.difficulty}.json"


def _read_generation_batch(path: Path, *, index: int, request: Any) -> tuple[Any, ...]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as error:
        raise RuntimeError(f"saved Gemini batch is missing or invalid: {path}") from error
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"saved Gemini batch must be an object: {path}")
    expected = {
        "schema": GENERATION_BATCH_SCHEMA,
        "request_index": index,
        "skill": request.skill,
        "difficulty": request.difficulty,
        "count": request.count,
        "prompt_sha256": _sha256_text(request.prompt),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(
                f"saved Gemini batch no longer matches the generation plan at {path}: {key}"
            )
    response_text = payload.get("response")
    if not isinstance(response_text, str) or not response_text.strip():
        raise RuntimeError(f"saved Gemini batch has no response text: {path}")
    teacher_records = schema.parse_teacher_batch(response_text, expected_count=request.count)
    return tuple(
        schema.ReasoningExample.from_teacher(
            teacher_record,
            skill=request.skill,
            difficulty=request.difficulty,
        )
        for teacher_record in teacher_records
    )


def _write_generation_batch(
    path: Path,
    *,
    index: int,
    request: Any,
    response_text: str,
) -> None:
    # Malformed provider output never becomes resumable state.
    schema.parse_teacher_batch(response_text, expected_count=request.count)
    _atomic_json(
        path,
        {
            "schema": GENERATION_BATCH_SCHEMA,
            "request_index": index,
            "skill": request.skill,
            "difficulty": request.difficulty,
            "count": request.count,
            "prompt_sha256": _sha256_text(request.prompt),
            "response": response_text,
        },
    )


def _validate_frozen_reasoning(
    path: Path,
    *,
    examples_per_cell: int,
) -> tuple[Any, ...]:
    records = schema.read_jsonl(path)
    bundle.validate_reasoning_matrix(records, examples_per_cell=examples_per_cell)
    return records


def _validate_completed_generation(
    root: Path,
    *,
    final_path: Path,
    examples_per_cell: int,
    batch_size: int,
    seed: int,
    total_calls: int,
) -> dict[str, object]:
    manifest_path = root / "generation-manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as error:
        raise RuntimeError("frozen reasoning JSONL exists without a valid generation manifest") from error
    if not isinstance(manifest, Mapping):
        raise RuntimeError("generation manifest must be a JSON object")
    supplied = manifest.get("manifest_sha256")
    without_hash = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if supplied != canonical_hash(without_hash):
        raise RuntimeError("generation manifest self-hash mismatch")
    expected = {
        "schema": GENERATION_MANIFEST_SCHEMA,
        "examples_per_cell": examples_per_cell,
        "batch_size": batch_size,
        "total_calls": total_calls,
        "seed": seed,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"completed generation configuration drifted at {key}")
    reasoning = manifest.get("reasoning_jsonl")
    if not isinstance(reasoning, Mapping):
        raise RuntimeError("generation manifest has no reasoning_jsonl identity")
    if reasoning.get("path") != final_path.name:
        raise RuntimeError("generation manifest reasoning path drifted")
    if reasoning.get("sha256") != _sha256_path(final_path):
        raise RuntimeError("frozen reasoning JSONL hash drifted")
    if reasoning.get("byte_size") != final_path.stat().st_size:
        raise RuntimeError("frozen reasoning JSONL byte size drifted")
    return dict(manifest)


def generate_resumable(
    output_dir: Path | str,
    *,
    examples_per_cell: int = bundle.DEFAULT_EXAMPLES_PER_CELL,
    batch_size: int = prompts.DEFAULT_BATCH_SIZE,
    seed: int = bundle.DEFAULT_SEED,
    client: Any | None = None,
) -> dict[str, object]:
    """Generate/freeze a uniform R0 corpus, resuming from schema-valid batch files."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    final_path = root / "reasoning.jsonl"
    plan = generation.build_uniform_generation_plan(
        examples_per_cell=examples_per_cell,
        batch_size=batch_size,
    )

    if final_path.is_file():
        manifest = _validate_completed_generation(
            root,
            final_path=final_path,
            examples_per_cell=examples_per_cell,
            batch_size=batch_size,
            seed=seed,
            total_calls=len(plan),
        )
        records = _validate_frozen_reasoning(final_path, examples_per_cell=examples_per_cell)
        reasoning_identity = manifest.get("reasoning_jsonl")
        if not isinstance(reasoning_identity, Mapping) or reasoning_identity.get("records") != len(records):
            raise RuntimeError("generation manifest reasoning record count drifted")
        return {
            "reasoning_jsonl": str(final_path),
            "records": len(records),
            "calls": len(plan),
            "resumed_complete": True,
            "manifest_sha256": manifest["manifest_sha256"],
        }

    records: list[Any] = []
    batch_files: list[dict[str, object]] = []
    live_client = client
    for index, request in enumerate(plan, start=1):
        path = _batch_path(root, index, request)
        if path.is_file():
            batch_records = _read_generation_batch(path, index=index, request=request)
        else:
            if live_client is None:
                live_client = dataset.GeminiDistillationClient()
            response = live_client.complete_text(request.prompt)
            response_text = getattr(response, "content", None)
            if not isinstance(response_text, str) or not response_text.strip():
                raise RuntimeError(f"Gemini request {index} returned no textual content")
            _write_generation_batch(
                path,
                index=index,
                request=request,
                response_text=response_text,
            )
            batch_records = _read_generation_batch(path, index=index, request=request)
        records.extend(batch_records)
        batch_files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": _sha256_path(path),
                "byte_size": path.stat().st_size,
                "request_index": index,
                "skill": request.skill,
                "difficulty": request.difficulty,
                "records": request.count,
            }
        )

    expected = len(prompts.R0_SKILLS) * len(generation.R0_DIFFICULTIES) * examples_per_cell
    if len(records) != expected:
        raise RuntimeError(f"generation assembled {len(records)} records; expected {expected}")
    bundle.validate_reasoning_matrix(records, examples_per_cell=examples_per_cell)
    random.Random(seed).shuffle(records)

    temporary = final_path.with_suffix(".jsonl.tmp")
    schema.write_jsonl(records, temporary)
    temporary.replace(final_path)
    frozen = _validate_frozen_reasoning(final_path, examples_per_cell=examples_per_cell)
    manifest_without_hash: dict[str, object] = {
        "schema": GENERATION_MANIFEST_SCHEMA,
        "reasoning_jsonl": {
            "path": final_path.name,
            "sha256": _sha256_path(final_path),
            "byte_size": final_path.stat().st_size,
            "records": len(frozen),
        },
        "examples_per_cell": examples_per_cell,
        "batch_size": batch_size,
        "total_calls": len(plan),
        "seed": seed,
        "batches": batch_files,
    }
    manifest = {
        **manifest_without_hash,
        "manifest_sha256": canonical_hash(manifest_without_hash),
    }
    _atomic_json(root / "generation-manifest.json", manifest)
    return {
        "reasoning_jsonl": str(final_path),
        "records": len(frozen),
        "calls": len(plan),
        "resumed_complete": False,
        "manifest_sha256": manifest["manifest_sha256"],
    }


def verify_matched_bundles(root: Path | str) -> dict[str, object]:
    """Verify both native arm bundles and their shared pilot identity."""

    directory = Path(root).expanduser().resolve()
    try:
        pilot = json.loads((directory / "pilot-manifest.json").read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as error:
        raise RuntimeError(f"R-SFT pilot manifest is missing or invalid: {directory}") from error
    if not isinstance(pilot, Mapping):
        raise RuntimeError("R-SFT pilot manifest must be an object")
    supplied = pilot.get("manifest_sha256")
    without_hash = {key: value for key, value in pilot.items() if key != "manifest_sha256"}
    if supplied != canonical_hash(without_hash):
        raise RuntimeError("R-SFT pilot manifest self-hash mismatch")

    arms = pilot.get("arms")
    if not isinstance(arms, Mapping):
        raise RuntimeError("R-SFT pilot manifest has no arm identities")
    verified: dict[str, object] = {}
    for arm in ("atomic", "textual"):
        row = arms.get(arm)
        if not isinstance(row, Mapping):
            raise RuntimeError(f"R-SFT pilot manifest has no {arm} arm")
        arm_root = directory / arm
        result = verify_bundle(arm_root)
        if result["bundle_manifest_sha256"] != row.get("bundle_manifest_sha256"):
            raise RuntimeError(f"R-SFT {arm} bundle identity mismatch")
        bundle.load_reasoning_token_spec(arm_root / "reasoning-tokens.json")
        verified[arm] = result
    return {
        "status": "verified",
        "pilot_manifest_sha256": supplied,
        "arms": verified,
    }


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    generate_cmd = sub.add_parser("generate", help="resumably generate the frozen R0 reasoning JSONL")
    generate_cmd.add_argument("--output-dir", type=Path, required=True)
    generate_cmd.add_argument("--examples-per-cell", type=_positive_int, default=bundle.DEFAULT_EXAMPLES_PER_CELL)
    generate_cmd.add_argument("--batch-size", type=_positive_int, default=prompts.DEFAULT_BATCH_SIZE)
    generate_cmd.add_argument("--seed", type=int, default=bundle.DEFAULT_SEED)
    generate_cmd.add_argument("--dry-run", action="store_true")

    build_cmd = sub.add_parser("build", help="build matched atomic/textual native SFT bundles")
    build_cmd.add_argument("--reasoning-jsonl", type=Path, required=True)
    build_cmd.add_argument("--s0-bundle", type=Path, required=True)
    build_cmd.add_argument("--token-spec", type=Path, required=True)
    build_cmd.add_argument("--output-dir", type=Path, required=True)
    build_cmd.add_argument("--examples-per-cell", type=_positive_int, default=bundle.DEFAULT_EXAMPLES_PER_CELL)
    build_cmd.add_argument("--heldout-per-cell", type=_positive_int, default=bundle.DEFAULT_HELDOUT_PER_CELL)
    build_cmd.add_argument("--optimizer-target-tokens", type=_positive_int, default=bundle.DEFAULT_OPTIMIZER_TARGET_TOKENS)
    build_cmd.add_argument("--context-length", type=_positive_int, default=bundle.DEFAULT_CONTEXT_LENGTH)
    build_cmd.add_argument("--seed", type=int, default=bundle.DEFAULT_SEED)

    pilot_cmd = sub.add_parser("pilot", help="generate the 630-example pilot and build both ablation bundles")
    pilot_cmd.add_argument("--s0-bundle", type=Path, required=True)
    pilot_cmd.add_argument("--token-spec", type=Path, required=True)
    pilot_cmd.add_argument("--output-dir", type=Path, required=True)
    pilot_cmd.add_argument("--examples-per-cell", type=_positive_int, default=bundle.DEFAULT_EXAMPLES_PER_CELL)
    pilot_cmd.add_argument("--batch-size", type=_positive_int, default=prompts.DEFAULT_BATCH_SIZE)
    pilot_cmd.add_argument("--heldout-per-cell", type=_positive_int, default=bundle.DEFAULT_HELDOUT_PER_CELL)
    pilot_cmd.add_argument("--optimizer-target-tokens", type=_positive_int, default=bundle.DEFAULT_OPTIMIZER_TARGET_TOKENS)
    pilot_cmd.add_argument("--context-length", type=_positive_int, default=bundle.DEFAULT_CONTEXT_LENGTH)
    pilot_cmd.add_argument("--seed", type=int, default=bundle.DEFAULT_SEED)
    pilot_cmd.add_argument("--dry-run", action="store_true")

    verify_cmd = sub.add_parser("verify", help="verify a matched bundle root")
    verify_cmd.add_argument("--dataset-dir", type=Path, required=True)
    return parser


def _dry_run_payload(args: argparse.Namespace) -> dict[str, object]:
    plan = generation.build_uniform_generation_plan(
        examples_per_cell=args.examples_per_cell,
        batch_size=args.batch_size,
    )
    root = args.output_dir.expanduser().resolve()
    return {
        "schema": "small-llm-rsft-pilot-dry-run-v1",
        "examples_per_cell": args.examples_per_cell,
        "total_examples": len(prompts.R0_SKILLS) * len(generation.R0_DIFFICULTIES) * args.examples_per_cell,
        "total_api_calls": len(plan),
        "batch_size": args.batch_size,
        "generation_root": str(root / "generation"),
        "reasoning_jsonl": str(root / "generation" / "reasoning.jsonl"),
        "matched_bundles": str(root / "bundles"),
        "atomic_bundle": str(root / "bundles" / "atomic"),
        "textual_bundle": str(root / "bundles" / "textual"),
        "retention_share": bundle.mixture.RETENTION_SHARE,
        "retention_source": "completed S0 instruction records only",
        "heldout_per_cell_per_split": args.heldout_per_cell,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "verify":
        result = verify_matched_bundles(args.dataset_dir)
    elif args.command == "generate":
        if args.dry_run:
            plan = generation.build_uniform_generation_plan(
                examples_per_cell=args.examples_per_cell,
                batch_size=args.batch_size,
            )
            result = generation.plan_summary(plan)
        else:
            result = generate_resumable(
                args.output_dir,
                examples_per_cell=args.examples_per_cell,
                batch_size=args.batch_size,
                seed=args.seed,
            )
    elif args.command == "build":
        result = bundle.build_matched_pilot_bundles(
            args.reasoning_jsonl,
            s0_bundle=args.s0_bundle,
            token_spec_path=args.token_spec,
            output_dir=args.output_dir,
            examples_per_cell=args.examples_per_cell,
            heldout_per_cell=args.heldout_per_cell,
            optimizer_target_tokens=args.optimizer_target_tokens,
            context_length=args.context_length,
            seed=args.seed,
        )
    else:
        if args.dry_run:
            result = _dry_run_payload(args)
        else:
            root = args.output_dir.expanduser().resolve()
            root.mkdir(parents=True, exist_ok=True)
            generation_result = generate_resumable(
                root / "generation",
                examples_per_cell=args.examples_per_cell,
                batch_size=args.batch_size,
                seed=args.seed,
            )
            bundles_root = root / "bundles"
            if bundles_root.exists():
                verified = verify_matched_bundles(bundles_root)
                result = {
                    "generation": generation_result,
                    "bundles": verified,
                    "resumed_complete": True,
                }
            else:
                built = bundle.build_matched_pilot_bundles(
                    Path(generation_result["reasoning_jsonl"]),
                    s0_bundle=args.s0_bundle,
                    token_spec_path=args.token_spec,
                    output_dir=bundles_root,
                    examples_per_cell=args.examples_per_cell,
                    heldout_per_cell=args.heldout_per_cell,
                    optimizer_target_tokens=args.optimizer_target_tokens,
                    context_length=args.context_length,
                    seed=args.seed,
                )
                result = {
                    "generation": generation_result,
                    "bundles": built,
                    "resumed_complete": False,
                }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "GENERATION_BATCH_SCHEMA",
    "GENERATION_MANIFEST_SCHEMA",
    "build_parser",
    "generate_resumable",
    "main",
    "verify_matched_bundles",
]
