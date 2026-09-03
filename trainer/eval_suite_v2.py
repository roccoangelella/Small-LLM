"""Active evaluation-v2 CLI for pretrained Small-LLM checkpoints.

This module keeps the frozen eval_core_v1 intrinsic measurement and wires the
new ADR-0140 evaluation layers into the normal pretrained evaluation entrypoint:
L20-style conditional likelihood and the 100-scored/20-qualitative base prompt
suite.  The legacy prompt runner remains available in trainer.eval_suite for
historical replays, but this CLI does not emit the retired fixed-length
qualitative benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Sequence

from dataset.eval_core import CONTEXT_LENGTH, canonical_json_bytes, verify_eval_core
from trainer.eval_suite import evaluate_split
from trainer.post_pretraining_prompt_suite import (
    _load_model,
    _resolve_device,
    _resolve_precision,
    download_verified_checkpoint,
)
from trainer.pretraining_eval_v2 import (
    run_base_prompt_suite_v2,
    run_l20_conditional_likelihood,
)

RESULT_SCHEMA_VERSION = 2


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pretrained checkpoint evaluation v2: eval_core_v1, L20 conditional likelihood, and base prompt v2."
    )
    parser.add_argument("suite", choices=("fast", "full"))
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--repo-id", default=os.environ.get("SMALL_LLM_HF_REPO_ID"))
    parser.add_argument("--run-id", default=os.environ.get("SMALL_LLM_RUN_ID"))
    parser.add_argument("--token-env", default="HF_TOKEN")
    parser.add_argument("--revision")
    parser.add_argument("--pointer", choices=("best", "latest"), default="best")
    parser.add_argument("--model-config-json", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--precision",
        choices=("auto", "fp32", "fp16", "bf16"),
        default="auto",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--bootstrap-samples", type=int, default=1_000)
    parser.add_argument(
        "--skip-l20",
        action="store_true",
        help="debug only; production evaluation v2 includes the L20 conditional-likelihood layer",
    )
    parser.add_argument(
        "--skip-base-prompts-v2",
        action="store_true",
        help="debug only; production evaluation v2 includes the 100 scored + 20 qualitative base prompt suite",
    )

    # Retired prompt-runner flags are accepted for command compatibility, but
    # deliberately ignored by evaluation v2. The v2 prompt layer has fixed
    # greedy and sampled contracts from ADR 0140.
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--samples-per-prompt", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--questions-only", action="store_true")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--skip-prompts", action="store_true")

    args = parser.parse_args(argv)
    if not args.repo_id:
        parser.error("set --repo-id or SMALL_LLM_HF_REPO_ID")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.bootstrap_samples < 0:
        parser.error("--bootstrap-samples cannot be negative")
    if args.max_new_tokens is not None and args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    return args


def _metric_mean(section: object) -> float | None:
    if not isinstance(section, dict):
        return None
    value = section.get("mean_6")
    return float(value) if isinstance(value, (int, float)) else None


def _prompt_accuracy(prompt_suite: object, view: str) -> float | None:
    if not isinstance(prompt_suite, dict):
        return None
    selected = prompt_suite.get(view)
    if not isinstance(selected, dict):
        return None
    summary = selected.get("summary")
    if not isinstance(summary, dict):
        return None
    value = summary.get("accuracy")
    return float(value) if isinstance(value, (int, float)) else None


def _read_me_first(*, l20_enabled: bool, prompt_v2_enabled: bool) -> dict[str, object]:
    return {
        "schema": "small-llm-pretraining-evaluation-v2",
        "how_to_read": [
            "metrics is the frozen in-domain eval_core_v1 next-token benchmark.",
            "l20_conditional_likelihood is the external six-task base-model capability layer; higher mean_6 is better.",
            "base_prompt_suite_v2.greedy is the deterministic human-readable prompt view with 100 scored cases and 20 qualitative continuations.",
            "base_prompt_suite_v2.sampled uses the project-standard sampled contract: temperature=1, top_p=1, top_k=0.",
            "EOS termination is intentionally not a pretraining metric.",
            "Teacher-forced confidence diagnostics are intentionally not part of this headline JSON; run them separately when debugging token probabilities.",
        ],
        "metric_direction": {
            "eval_core_v1.loss": "lower_is_better",
            "eval_core_v1.perplexity": "lower_is_better",
            "eval_core_v1.bits_per_byte": "lower_is_better",
            "eval_core_v1.top_k_accuracy": "higher_is_better",
            "l20_conditional_likelihood.mean_6": "higher_is_better",
            "base_prompt_suite_v2.greedy.summary.accuracy": "higher_is_better",
        },
        "enabled_layers": {
            "eval_core_v1": True,
            "l20_conditional_likelihood": l20_enabled,
            "base_prompt_suite_v2": prompt_v2_enabled,
        },
        "retired_protocol": "The old global fixed-length qualitative cap is not emitted by this v2 entrypoint.",
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    eval_dir = args.eval_dir.resolve()
    verify_eval_core(eval_dir)
    device = _resolve_device(args.device)
    precision = _resolve_precision(args.precision, device)
    token = os.environ.get(args.token_env)

    with tempfile.TemporaryDirectory(prefix="small-llm-eval-v2-") as temporary:
        checkpoint_root, checkpoint_info = download_verified_checkpoint(
            repo_id=str(args.repo_id),
            run_id=args.run_id,
            token=token,
            revision=args.revision,
            pointer_name=args.pointer,
            destination=Path(temporary),
        )
        model, model_config, trainer_state = _load_model(
            checkpoint_root,
            device=device,
            model_config_json=args.model_config_json,
        )
        if model_config.max_seq_len != CONTEXT_LENGTH:
            raise RuntimeError(
                f"eval_core_v1 requires context {CONTEXT_LENGTH}, "
                f"checkpoint uses {model_config.max_seq_len}"
            )

        print("=" * 80)
        print(f"Small-LLM pretraining evaluation v2 | suite={args.suite}")
        print(
            json.dumps(
                {
                    **checkpoint_info,
                    "device": str(device),
                    "precision": precision,
                    "global_step": trainer_state.get("global_step"),
                    "consumed_tokens": trainer_state.get("consumed_tokens"),
                },
                indent=2,
                sort_keys=True,
            )
        )

        metrics = evaluate_split(
            model,
            eval_dir=eval_dir,
            suite=args.suite,
            precision=precision,
            batch_size=args.batch_size,
            bootstrap_samples=args.bootstrap_samples,
        )
        l20 = None if args.skip_l20 else run_l20_conditional_likelihood(
            model,
            model_max_seq_len=model_config.max_seq_len,
            precision=precision,
            suite=args.suite,
        )
        prompt_suite = None if (args.skip_base_prompts_v2 or args.skip_prompts) else run_base_prompt_suite_v2(
            model,
            model_max_seq_len=model_config.max_seq_len,
            precision=precision,
            suite=args.suite,
        )

        headline_summary = {
            "eval_core_v1": {
                "loss": metrics.get("loss"),
                "perplexity": metrics.get("perplexity"),
                "bits_per_byte": metrics.get("bits_per_byte"),
                "top_1_accuracy": metrics.get("top_k_accuracy", {}).get("1")
                if isinstance(metrics.get("top_k_accuracy"), dict)
                else None,
                "cluster_macro_loss": metrics.get("cluster_macro_loss"),
                "cluster_mixture_weighted_loss": metrics.get("cluster_mixture_weighted_loss"),
                "worst_cluster_loss": metrics.get("worst_cluster_loss"),
            },
            "l20_conditional_likelihood_mean_6": _metric_mean(l20),
            "base_prompt_v2_greedy_accuracy": _prompt_accuracy(prompt_suite, "greedy"),
            "base_prompt_v2_sampled_accuracy": _prompt_accuracy(prompt_suite, "sampled"),
            "interpretation": "Use eval_core_v1 for in-domain LM fit, L20 mean_6 for external base capability, and prompt v2 cases for readable generation evidence.",
        }

        result_without_hash: dict[str, object] = {
            "schema": "small-llm-pretraining-evaluation-v2",
            "schema_version": RESULT_SCHEMA_VERSION,
            "read_me_first": _read_me_first(
                l20_enabled=l20 is not None,
                prompt_v2_enabled=prompt_suite is not None,
            ),
            "headline_summary": headline_summary,
            "checkpoint": checkpoint_info,
            "checkpoint_state": {
                "global_step": trainer_state.get("global_step"),
                "consumed_tokens": trainer_state.get("consumed_tokens"),
            },
            "model_config": {
                "architecture": model_config.architecture,
                "d_model": model_config.d_model,
                "n_layers": model_config.n_layers,
                "d_ff": model_config.d_ff,
                "max_seq_len": model_config.max_seq_len,
            },
            "metrics": metrics,
            "eval_core_v1": metrics,
            "l20_conditional_likelihood": l20,
            "base_prompt_suite_v2": prompt_suite,
            "sampling_contracts": {
                "base_prompt_v2_greedy": {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "seed": 17},
                "base_prompt_v2_sampled": {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "seed": 17},
            },
            "compatibility_notes": {
                "accepted_legacy_prompt_flags": [
                    "--temperature",
                    "--top-p",
                    "--top-k",
                    "--seed",
                    "--samples-per-prompt",
                    "--max-new-tokens",
                    "--questions-only",
                    "--max-cases",
                ],
                "legacy_prompt_flags_used": False,
            },
        }
        result = {
            **result_without_hash,
            "result_sha256": hashlib.sha256(
                canonical_json_bytes(result_without_hash)
            ).hexdigest(),
        }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"\nSaved pretraining evaluation-v2 bundle to {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
