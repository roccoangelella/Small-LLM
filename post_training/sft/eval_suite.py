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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--suite", choices=("fast", "full"), default="full")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp16")
    parser.add_argument("--batch-size", type=_positive_int, default=1)
    parser.add_argument("--validation-blocks", type=_positive_int, default=32)
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


def _masked_validation(
    model,
    *,
    precision: str,
    device: torch.device,
    bundle_root: Path,
    maximum_blocks: int,
) -> dict[str, object]:
    engine = SimpleNamespace(
        model=model,
        device=device,
        config=SimpleNamespace(precision=precision),
        optimizer=None,
        best_validation_loss=None,
    )
    reader = SFTShardReader(bundle_root / "validation", split="validation")
    return evaluate_batches(
        engine,
        reader.iter_from_start(),
        maximum_batches=maximum_blocks,
        microbatch_size=1,
    )


def _base_prompts(model, *, max_seq_len: int, precision: str, suite: str) -> list[dict[str, object]]:
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
    validation = _masked_validation(
        model,
        precision=precision,
        device=next(model.parameters()).device,
        bundle_root=bundle_root,
        maximum_blocks=validation_blocks,
    )
    behavior = evaluate_behavior(
        model,
        precision=precision,
        max_seq_len=model_config.max_seq_len,
        max_cases=16 if suite == "fast" else None,
    )
    qualitative = _base_prompts(
        model,
        max_seq_len=model_config.max_seq_len,
        precision=precision,
        suite=suite,
    )
    return {
        "eval_core_v1": intrinsic,
        "sft_validation": validation,
        "instruction_behavior": behavior,
        "base_qualitative_prompts": qualitative,
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


def _deltas(parent: Mapping[str, object], tuned: Mapping[str, object]) -> dict[str, object]:
    p_intrinsic = parent["eval_core_v1"]
    t_intrinsic = tuned["eval_core_v1"]
    p_validation = parent["sft_validation"]
    t_validation = tuned["sft_validation"]
    p_behavior = parent["instruction_behavior"]
    t_behavior = tuned["instruction_behavior"]
    assert isinstance(p_intrinsic, Mapping) and isinstance(t_intrinsic, Mapping)
    assert isinstance(p_validation, Mapping) and isinstance(t_validation, Mapping)
    assert isinstance(p_behavior, Mapping) and isinstance(t_behavior, Mapping)
    p_summary = p_behavior["summary"]
    t_summary = t_behavior["summary"]
    assert isinstance(p_summary, Mapping) and isinstance(t_summary, Mapping)
    return {
        "eval_core_v1": {
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
        "sft_validation": {
            "loss": _delta_number(p_validation.get("loss"), t_validation.get("loss")),
            "perplexity": _delta_number(p_validation.get("perplexity"), t_validation.get("perplexity")),
        },
        "instruction_behavior": {
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
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bundle_root = args.dataset_dir.resolve()
    bundle_verification = verify_bundle(bundle_root)
    eval_dir = ensure_eval_core(args.eval_dir)
    token = os.environ.get(args.token_env)
    device = torch.device(args.device)
    if args.precision == "fp16" and device.type != "cuda":
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

    parent_model, parent_config, parent_identity = load_verified_native_checkpoint(parent_root, device=device)
    sft_model, sft_config, sft_identity = load_verified_native_checkpoint(sft_root, device=device)
    if parent_config != sft_config:
        raise RuntimeError("parent and SFT checkpoints have different model geometry")

    print("Scoring immutable parent checkpoint...", flush=True)
    parent_result = _score_one(
        parent_model,
        model_config=parent_config,
        bundle_root=bundle_root,
        eval_dir=eval_dir,
        suite=args.suite,
        precision=args.precision,
        batch_size=args.batch_size,
        validation_blocks=args.validation_blocks,
        bootstrap_samples=args.bootstrap_samples,
    )
    del parent_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("Scoring SFT checkpoint...", flush=True)
    sft_result = _score_one(
        sft_model,
        model_config=sft_config,
        bundle_root=bundle_root,
        eval_dir=eval_dir,
        suite=args.suite,
        precision=args.precision,
        batch_size=args.batch_size,
        validation_blocks=args.validation_blocks,
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


__all__ = ["build_parser", "main"]
