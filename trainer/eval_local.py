"""Evaluate a verified local joint checkpoint on eval_core_v1 only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

from dataset.eval_core import CONTEXT_LENGTH, canonical_json_bytes
from dataset.src.joint_checkpoint import verify_local_manifest
from dataset.src.remote import sha256_path
from MOE_model.evaluation import load_moe_model
from trainer.eval_suite import evaluate_split
from trainer.observation import _source_identity
from trainer.post_pretraining_prompt_suite import (
    _load_model as load_dense_model,
    _resolve_device,
    _resolve_precision,
)


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a local Small-LLM checkpoint on eval_core_v1.")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("fast", "full"), default="fast")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("auto", "fp32", "fp16", "bf16"), default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--bootstrap-samples", type=int, default=1_000)
    parser.add_argument("--moe", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.bootstrap_samples < 0:
        parser.error("--batch-size must be positive and --bootstrap-samples non-negative")
    return args


def _json_safe(value):
    # Undefined empty-bucket metrics are null in the artifact; scoring is unchanged.
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    # Keep the final path component visible to the joint verifier, which rejects
    # symlinked checkpoint roots before reading opaque trainer state.
    checkpoint_dir = args.checkpoint_dir.absolute()
    eval_dir = args.eval_dir.resolve()
    verify_local_manifest(checkpoint_dir)
    checkpoint_json = checkpoint_dir / "checkpoint.json"
    checkpoint_metadata = json.loads(checkpoint_json.read_text(encoding="utf-8"))
    if (not isinstance(checkpoint_metadata, dict) or checkpoint_metadata.get("version") != 1
            or checkpoint_metadata.get("optimizer_step_complete") is not True):
        raise RuntimeError("checkpoint.json is not a version-1 joint checkpoint")
    device = _resolve_device(args.device)
    precision = _resolve_precision(args.precision, device)
    loader = load_moe_model if args.moe else load_dense_model
    model, model_config, trainer_state = loader(
        checkpoint_dir, device=device, model_config_json=None
    )
    if model_config.max_seq_len != CONTEXT_LENGTH:
        raise RuntimeError(
            f"eval_core_v1 requires context {CONTEXT_LENGTH}, "
            f"checkpoint uses {model_config.max_seq_len}"
        )
    metrics = evaluate_split(
        model,
        eval_dir=eval_dir,
        suite=args.split,
        precision=precision,
        batch_size=args.batch_size,
        bootstrap_samples=args.bootstrap_samples,
    )
    result_without_hash = {
        "schema": "small-llm-local-evaluation-v1",
        "evaluation_source": _source_identity(None),
        "evaluation_config": {"split": args.split, "device": str(device),
                              "precision": precision, "batch_size": args.batch_size,
                              "bootstrap_samples": args.bootstrap_samples},
        "checkpoint": {
            "path": str(checkpoint_dir),
            "checkpoint_manifest_sha256": sha256_path(checkpoint_json),
            "local_manifest_sha256": sha256_path(checkpoint_dir / "local_manifest.json"),
            "metadata": checkpoint_metadata,
        },
        "checkpoint_state": {
            "global_step": trainer_state.get("global_step"),
            "consumed_tokens": trainer_state.get("consumed_tokens"),
        },
        "model_config": model_config.as_dict(),
        "eval_core_v1": _json_safe(metrics),
    }
    result = {
        **result_without_hash,
        "result_sha256": hashlib.sha256(canonical_json_bytes(result_without_hash)).hexdigest(),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
