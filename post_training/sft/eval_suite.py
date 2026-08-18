"""Comprehensive parent-versus-SFT checkpoint qualification."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
from typing import Mapping, Sequence

import torch

from trainer.eval_entrypoint import ensure_eval_core
from trainer.eval_suite import evaluate_split, run_prompt_cases
from trainer.identity import canonical_hash
from trainer.evaluation import evaluate_batches

from .behavior_eval import evaluate_behavior
from .bundle import verify_bundle
from .checkpoints import download_parent_checkpoint, load_verified_native_checkpoint
from .storage import SFTShardReader


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed
def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _resolve_precision(precision: str, device: torch.device) -> str:
    if precision == "auto":
        return "fp16" if device.type == "cuda" else "fp32"
    return precision


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--suite", choices=("fast", "full"), default="full")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("auto", "fp32", "fp16", "bf16"), default="auto")
    parser.add_argument("--batch-size", type=_positive_int, default=1)
    parser.add_argument("--validation-blocks", type=_positive_int, default=32)
    parser.add_argument("--test-blocks", type=_positive_int, default=32)
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-env", default="HF_TOKEN")

    parent = parser.add_argument_group("parent checkpoint")
    parent.add_argument("--parent-checkpoint-dir", type=Path)
    parent.add_argument("--parent-repo-id")
    parent.add_argument("--parent-run-id")
    parent.add_argument("--parent-pointer", choices=("best", "latest"), default="best")
    parent.add_argument("--parent-revision")

    tuned = parser.add_argument_group("SFT checkpoint")
    tuned.add_argument("--sft-checkpoint-dir", type=Path)
    tuned.add_argument("--sft-repo-id")
    tuned.add_argument("--sft-run-id")
    tuned.add_argument("--sft-pointer", choices=("best", "latest"), default="latest")
    tuned.add_argument("--sft-revision")
    return parser


def _resolve(
    *,
    local: Path | None,
    repo_id: str | None,
    run_id: str | None,
    pointer: str,
    revision: str | None,
    token: str | None,
    label: str,
) -> tuple[Path, dict[str, object]]:
    if local is not None:
        return local.resolve(), {"transport": "local", "path": str(local.resolve())}
    if not repo_id or not run_id:
        raise RuntimeError(f"{label}: pass a local checkpoint directory or both repo/run IDs")
    destination = Path(tempfile.mkdtemp(prefix=f"small-llm-{label}-"))
    root, remote = download_parent_checkpoint(
        repo_id=repo_id,
        run_id=run_id,
        pointer=pointer,
        token=token,
        revision=revision,
        destination=destination,
    )
    return root, {"transport": "remote", **remote}


def _masked_split(
    model,
    *,
    precision: str,
    device: torch.device,
    bundle_root: Path,
    split: str,
    maximum_blocks: int,
    microbatch_size: int = 4,
) -> dict[str, object]:
    if split not in {"validation", "test"}:
        raise ValueError("masked SFT evaluation split must be validation or test")
    engine = SimpleNamespace(
        model=model,
        device=device,
        config=SimpleNamespace(precision=precision),
        optimizer=None,
        best_validation_loss=None,
    )
    reader = SFTShardReader(bundle_root / split, split=split)
    return evaluate_batches(
        engine,
        reader.iter_from_start(),
        maximum_batches=maximum_blocks,
        microbatch_size=microbatch_size,
    )


def _base_prompts_greedy_32(
    model, *, max_seq_len: int, precision: str, suite: str
) -> list[dict[str, object]]:
    return run_prompt_cases(
        model,
        model_max_seq_len=max_seq_len,
        precision=precision,
        questions_only=False,
        max_cases=8 if suite == "fast" else None,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        seed=17,
        samples_per_prompt=1,
        max_new_tokens=32,
    )


def _base_prompts_sampled(
    model, *, max_seq_len: int, precision: str, suite: str
) -> list[dict[str, object]]:
    return run_prompt_cases(
        model,
        model_max_seq_len=max_seq_len,
        precision=precision,
        questions_only=False,
        max_cases=8 if suite == "fast" else None,
        temperature=1.0,
        top_p=0.9,
        top_k=20,
        seed=17,
        samples_per_prompt=1,
        max_new_tokens=None,
    )


def _base_prompts(
    model, *, max_seq_len: int, precision: str, suite: str
) -> list[dict[str, object]]:
    return _base_prompts_greedy_32(
        model, max_seq_len=max_seq_len, precision=precision, suite=suite
    )


def _score_one(
    model,
    *,
    model_config,
    bundle_root: Path,
    eval_dir: Path,
    suite: str,
    precision: str,
    batch_size: int,
    validation_blocks: int,
    test_blocks: int,
    bootstrap_samples: int,
) -> dict[str, object]:
    intrinsic = evaluate_split(
        model,
        eval_dir=eval_dir,
        suite=suite,
        precision=precision,
        batch_size=batch_size,
        bootstrap_samples=bootstrap_samples,
    )
    validation = _masked_split(
        model,
        precision=precision,
        device=next(model.parameters()).device,
        bundle_root=bundle_root,
        split="validation",
        maximum_blocks=validation_blocks,
        microbatch_size=max(1, batch_size),
    )
    test = (
        _masked_split(
            model,
            precision=precision,
            device=next(model.parameters()).device,
            bundle_root=bundle_root,
            split="test",
            maximum_blocks=test_blocks,
            microbatch_size=max(1, batch_size),
        )
        if suite == "full"
        else None
    )
    behavior = evaluate_behavior(
        model,
        precision=precision,
        max_seq_len=model_config.max_seq_len,
        max_cases=16 if suite == "fast" else None,
    )
    prompts_greedy_32 = _base_prompts_greedy_32(
        model,
        max_seq_len=model_config.max_seq_len,
        precision=precision,
        suite=suite,
    )
    prompts_sampled = _base_prompts_sampled(
        model,
        max_seq_len=model_config.max_seq_len,
        precision=precision,
        suite=suite,
    )
    return {
        "eval_core_v1": intrinsic,
        "sft_validation": validation,
        "sft_test": test,
        "instruction_behavior": behavior,
        "base_qualitative_prompts": prompts_greedy_32,
        "qualitative_greedy_32": prompts_greedy_32,
        "qualitative_sampled": prompts_sampled,
    }


def _delta_number(parent: object, tuned: object) -> float | None:
    if (
        isinstance(parent, bool)
        or isinstance(tuned, bool)
        or not isinstance(parent, (int, float))
        or not isinstance(tuned, (int, float))
    ):
        return None
    return float(tuned) - float(parent)


def _mapping_numeric_deltas(
    parent: object,
    tuned: object,
    *,
    keys: Sequence[str] | None = None,
) -> dict[str, float | None]:
    if not isinstance(parent, Mapping) or not isinstance(tuned, Mapping):
        return {}
    selected = list(keys) if keys is not None else sorted(set(parent) & set(tuned))
    return {str(key): _delta_number(parent.get(key), tuned.get(key)) for key in selected}


def _cluster_deltas(parent: object, tuned: object) -> dict[str, object]:
    if not isinstance(parent, Mapping) or not isinstance(tuned, Mapping):
        return {}
    result: dict[str, object] = {}
    for cluster in sorted(set(parent) & set(tuned), key=str):
        p_row, t_row = parent[cluster], tuned[cluster]
        if isinstance(p_row, Mapping) and isinstance(t_row, Mapping):
            result[str(cluster)] = _mapping_numeric_deltas(
                p_row,
                t_row,
                keys=("loss", "perplexity"),
            )
    return result


def _position_deltas(parent: object, tuned: object) -> list[dict[str, object]]:
    if not isinstance(parent, list) or not isinstance(tuned, list):
        return []
    p_rows = {
        int(row["index"]): row
        for row in parent
        if isinstance(row, Mapping) and isinstance(row.get("index"), int)
    }
    t_rows = {
        int(row["index"]): row
        for row in tuned
        if isinstance(row, Mapping) and isinstance(row.get("index"), int)
    }
    return [
        {
            "index": index,
            "loss": _delta_number(p_rows[index].get("loss"), t_rows[index].get("loss")),
        }
        for index in sorted(set(p_rows) & set(t_rows))
    ]


def _behavior_category_deltas(parent: object, tuned: object) -> dict[str, object]:
    if not isinstance(parent, Mapping) or not isinstance(tuned, Mapping):
        return {}
    result: dict[str, object] = {}
    for category in sorted(set(parent) & set(tuned)):
        p_row, t_row = parent[category], tuned[category]
        if isinstance(p_row, Mapping) and isinstance(t_row, Mapping):
            result[str(category)] = {
                "pass_rate": _delta_number(p_row.get("pass_rate"), t_row.get("pass_rate")),
                "passed": _delta_number(p_row.get("passed"), t_row.get("passed")),
            }
    return result


def _masked_loss_deltas(parent: object, tuned: object) -> dict[str, object] | None:
    if parent is None or tuned is None:
        return None
    if not isinstance(parent, Mapping) or not isinstance(tuned, Mapping):
        return None
    return {
        "loss": _delta_number(parent.get("loss"), tuned.get("loss")),
        "perplexity": _delta_number(parent.get("perplexity"), tuned.get("perplexity")),
    }


def _deltas(parent: Mapping[str, object], tuned: Mapping[str, object]) -> dict[str, object]:
    p_intrinsic = parent["eval_core_v1"]
    t_intrinsic = tuned["eval_core_v1"]
    p_behavior = parent["instruction_behavior"]
    t_behavior = tuned["instruction_behavior"]
    assert isinstance(p_intrinsic, Mapping) and isinstance(t_intrinsic, Mapping)
    assert isinstance(p_behavior, Mapping) and isinstance(t_behavior, Mapping)
    p_summary = p_behavior["summary"]
    t_summary = t_behavior["summary"]
    assert isinstance(p_summary, Mapping) and isinstance(t_summary, Mapping)

    p_calibration = p_intrinsic.get("calibration")
    t_calibration = t_intrinsic.get("calibration")
    calibration_delta = None
    if isinstance(p_calibration, Mapping) and isinstance(t_calibration, Mapping):
        calibration_delta = _delta_number(p_calibration.get("ece"), t_calibration.get("ece"))

    return {
        "eval_core_v1": {
            **{
                name: _delta_number(p_intrinsic.get(name), t_intrinsic.get(name))
                for name in (
                    "loss",
                    "perplexity",
                    "bits_per_byte",
                    "cluster_macro_loss",
                    "cluster_mixture_weighted_loss",
                    "worst_cluster_loss",
                    "tokens_per_second",
                    "peak_allocated_vram_bytes",
                )
            },
            "top_k_accuracy": _mapping_numeric_deltas(
                p_intrinsic.get("top_k_accuracy"),
                t_intrinsic.get("top_k_accuracy"),
            ),
            "calibration_ece": calibration_delta,
            "per_cluster": _cluster_deltas(
                p_intrinsic.get("per_cluster"),
                t_intrinsic.get("per_cluster"),
            ),
            "position_buckets": _position_deltas(
                p_intrinsic.get("position_buckets"),
                t_intrinsic.get("position_buckets"),
            ),
        },
        "sft_validation": _masked_loss_deltas(
            parent.get("sft_validation"), tuned.get("sft_validation")
        ),
        "sft_test": _masked_loss_deltas(parent.get("sft_test"), tuned.get("sft_test")),
        "instruction_behavior": {
            **{
                name: _delta_number(p_summary.get(name), t_summary.get(name))
                for name in (
                    "pass_rate",
                    "eos_termination_rate",
                    "runaway_rate",
                    "empty_rate",
                    "role_leak_rate",
                    "mean_response_tokens",
                    "mean_trigram_repetition",
                )
            },
            "per_category": _behavior_category_deltas(
                p_behavior.get("per_category"),
                t_behavior.get("per_category"),
            ),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bundle_root = args.dataset_dir.resolve()
    bundle_verification = verify_bundle(bundle_root)
    eval_dir = ensure_eval_core(args.eval_dir)
    token = os.environ.get(args.token_env)
    device = _resolve_device(args.device)
    precision = _resolve_precision(args.precision, device)
    if precision == "fp16" and device.type != "cuda":
        raise RuntimeError("fp16 comprehensive evaluation requires CUDA")

    parent_root, parent_transport = _resolve(
        local=args.parent_checkpoint_dir,
        repo_id=args.parent_repo_id,
        run_id=args.parent_run_id,
        pointer=args.parent_pointer,
        revision=args.parent_revision,
        token=token,
        label="parent",
    )
    sft_root, sft_transport = _resolve(
        local=args.sft_checkpoint_dir,
        repo_id=args.sft_repo_id,
        run_id=args.sft_run_id,
        pointer=args.sft_pointer,
        revision=args.sft_revision,
        token=token,
        label="sft",
    )

    print("Scoring immutable parent checkpoint...", flush=True)
    parent_model, parent_config, parent_identity = load_verified_native_checkpoint(
        parent_root, device=device
    )
    parent_result = _score_one(
        parent_model,
        model_config=parent_config,
        bundle_root=bundle_root,
        eval_dir=eval_dir,
        suite=args.suite,
        precision=precision,
        batch_size=args.batch_size,
        validation_blocks=args.validation_blocks,
        test_blocks=args.test_blocks,
        bootstrap_samples=args.bootstrap_samples,
    )
    del parent_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("Scoring SFT checkpoint...", flush=True)
    sft_model, sft_config, sft_identity = load_verified_native_checkpoint(sft_root, device=device)
    if parent_config != sft_config:
        raise RuntimeError("parent and SFT checkpoints have different model geometry")
    sft_result = _score_one(
        sft_model,
        model_config=sft_config,
        bundle_root=bundle_root,
        eval_dir=eval_dir,
        suite=args.suite,
        precision=precision,
        batch_size=args.batch_size,
        validation_blocks=args.validation_blocks,
        test_blocks=args.test_blocks,
        bootstrap_samples=args.bootstrap_samples,
    )

    report_without_hash: dict[str, object] = {
        "schema": "small-llm-post-sft-qualification",
        "schema_version": 1,
        "suite": args.suite,
        "bundle": bundle_verification,
        "parent": {
            "checkpoint": parent_identity,
            "transport": parent_transport,
            "scorecard": parent_result,
        },
        "sft": {
            "checkpoint": sft_identity,
            "transport": sft_transport,
            "scorecard": sft_result,
        },
        "deltas_sft_minus_parent": _deltas(parent_result, sft_result),
        "selection_policy": {
            "single_master_score": False,
            "interpretation": "inspect instruction acquisition and base-capability retention together",
        },
    }
    report = {**report_without_hash, "report_sha256": canonical_hash(report_without_hash)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "_deltas",
    "_mapping_numeric_deltas",
    "build_parser",
    "main",
]
