"""Run the complete Small-LLM checkpoint evaluation suite.

One command verifies and downloads a native checkpoint, evaluates either the
nested ``fast`` or ``full`` ``eval_core_v1`` split, prints the existing
qualitative prompt answers, and writes one versioned JSON result bundle.
"""
from __future__ import annotations

import argparse
from array import array
from collections import defaultdict
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import time
from typing import Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from dataset.eval_core import (
    ACCEPTED_CLUSTERS,
    CONTEXT_LENGTH,
    MIXTURE_SOURCE_TOKENS,
    STORED_TOKENS,
    canonical_json_bytes,
    verify_eval_core,
)
from trainer.post_pretraining_prompt_suite import (
    _load_model,
    _resolve_device,
    _resolve_precision,
    _selected_cases,
    download_verified_checkpoint,
    sample_token_ids,
)

RESULT_SCHEMA_VERSION = 1
ECE_BINS = 15
POSITION_BUCKETS = 8


def _autocast(device: torch.device, precision: str):
    if precision == "fp32":
        return nullcontext()
    if device.type != "cuda":
        raise RuntimeError(f"{precision} evaluation requires CUDA")
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _records(
    eval_dir: Path, suite: str, manifest: Mapping[str, object]
) -> list[dict[str, object]]:
    suites = manifest["suites"]
    assert isinstance(suites, Mapping)
    suite_manifest = suites[suite]
    assert isinstance(suite_manifest, Mapping)
    path = eval_dir / str(suite_manifest["records_file"])
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"non-object row at {path}:{line_number}")
            rows.append(row)
    return rows


def _binary_batches(
    eval_dir: Path,
    suite: str,
    manifest: Mapping[str, object],
    records: Sequence[Mapping[str, object]],
    *,
    batch_size: int,
) -> Iterable[tuple[Tensor, Tensor, Tensor, tuple[Mapping[str, object], ...]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    suites = manifest["suites"]
    assert isinstance(suites, Mapping)
    suite_manifest = suites[suite]
    assert isinstance(suite_manifest, Mapping)
    path = eval_dir / str(suite_manifest["data_file"])
    row_bytes = STORED_TOKENS * 2
    with path.open("rb") as handle:
        for start in range(0, len(records), batch_size):
            rows = tuple(records[start : start + batch_size])
            raw = handle.read(len(rows) * row_bytes)
            if len(raw) != len(rows) * row_bytes:
                raise RuntimeError(f"short eval read from {path}")
            values = array("H")
            values.frombytes(raw)
            if sys.byteorder != "little":
                values.byteswap()
            tokens = torch.tensor(values, dtype=torch.long).view(
                len(rows), STORED_TOKENS
            )
            valid_targets = torch.tensor(
                [int(row["valid_targets"]) for row in rows], dtype=torch.long
            )
            positions = torch.arange(CONTEXT_LENGTH).unsqueeze(0)
            mask = positions < valid_targets.unsqueeze(1)
            yield tokens[:, :-1], tokens[:, 1:], mask, rows
        if handle.read(1):
            raise RuntimeError(f"eval binary has trailing bytes: {path}")


def _bootstrap_interval(
    values: Sequence[tuple[float, int]],
    *,
    samples: int,
    seed: int,
) -> dict[str, float | int]:
    if samples <= 0:
        return {"samples": 0, "low": math.nan, "high": math.nan}
    if not values:
        return {"samples": samples, "low": math.nan, "high": math.nan}
    rng = random.Random(seed)
    size = len(values)
    estimates: list[float] = []
    for _ in range(samples):
        loss_sum = 0.0
        token_count = 0
        for _ in range(size):
            selected_loss, selected_tokens = values[rng.randrange(size)]
            loss_sum += selected_loss
            token_count += selected_tokens
        estimates.append(loss_sum / token_count)
    estimates.sort()
    low_index = max(0, round(0.025 * (samples - 1)))
    high_index = min(samples - 1, round(0.975 * (samples - 1)))
    return {
        "samples": samples,
        "low": estimates[low_index],
        "high": estimates[high_index],
    }


def _gpt2_token_byte_lengths() -> list[int]:
    try:
        import tiktoken
    except ImportError as error:
        raise RuntimeError(
            "evaluation requires tiktoken; install the project with .[post-training]"
        ) from error
    encoding = tiktoken.get_encoding("gpt2")
    return [
        len(encoding.decode_single_token_bytes(token_id))
        for token_id in range(encoding.n_vocab)
    ]


@torch.inference_mode()
def evaluate_split(
    model: nn.Module,
    *,
    eval_dir: Path,
    suite: str,
    precision: str,
    batch_size: int,
    bootstrap_samples: int,
    token_byte_lengths: Sequence[int] | None = None,
    enforce_frozen_minimums: bool = True,
) -> dict[str, object]:
    """Evaluate one verified eval_core split with streaming accumulators."""
    if suite not in {"fast", "full"}:
        raise ValueError("suite must be fast or full")
    manifest = verify_eval_core(
        eval_dir, enforce_frozen_minimums=enforce_frozen_minimums
    )
    records = _records(eval_dir, suite, manifest)
    device = next(model.parameters()).device
    byte_lengths = list(token_byte_lengths or _gpt2_token_byte_lengths())
    if len(byte_lengths) < int(manifest["semantic_vocab_size"]):
        raise ValueError("token byte-length table is smaller than the vocabulary")
    byte_length_tensor = torch.tensor(byte_lengths, dtype=torch.long, device=device)

    total_loss = 0.0
    total_tokens = 0
    total_bytes = 0
    top_correct = {1: 0, 5: 0, 10: 0}
    per_cluster_loss = defaultdict(float)
    per_cluster_tokens = defaultdict(int)
    per_document_loss = defaultdict(float)
    per_document_tokens = defaultdict(int)
    per_document_cluster: dict[str, int] = {}
    position_loss = [0.0] * POSITION_BUCKETS
    position_tokens = [0] * POSITION_BUCKETS
    ece_count = [0] * ECE_BINS
    ece_confidence = [0.0] * ECE_BINS
    ece_correct = [0] * ECE_BINS

    was_training = model.training
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    try:
        for input_ids, labels, mask, rows in _binary_batches(
            eval_dir, suite, manifest, records, batch_size=batch_size
        ):
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            with _autocast(device, precision):
                logits = model(input_ids)
            if logits.shape[:2] != labels.shape:
                raise RuntimeError("model logits do not match eval sequence geometry")
            losses = F.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                reduction="none",
            ).view_as(labels)
            if not torch.isfinite(losses[mask]).all():
                raise FloatingPointError("evaluation produced non-finite token loss")

            masked_losses = losses[mask]
            total_loss += float(masked_losses.sum())
            batch_tokens = int(mask.sum())
            total_tokens += batch_tokens
            total_bytes += int(byte_length_tensor[labels[mask]].sum())

            maximum_k = min(10, logits.shape[-1])
            ranked = logits.topk(maximum_k, dim=-1).indices
            for k in top_correct:
                effective_k = min(k, maximum_k)
                matches = (ranked[..., :effective_k] == labels.unsqueeze(-1)).any(-1)
                top_correct[k] += int(matches[mask].sum())

            float_logits = logits.float()
            max_logits, predictions = float_logits.max(dim=-1)
            confidences = torch.exp(max_logits - torch.logsumexp(float_logits, dim=-1))
            correctness = predictions.eq(labels)
            selected_confidence = confidences[mask]
            selected_correctness = correctness[mask]
            bins = torch.clamp(
                (selected_confidence * ECE_BINS).long(), max=ECE_BINS - 1
            )
            for bin_index in range(ECE_BINS):
                selected = bins == bin_index
                count = int(selected.sum())
                if count:
                    ece_count[bin_index] += count
                    ece_confidence[bin_index] += float(
                        selected_confidence[selected].sum()
                    )
                    ece_correct[bin_index] += int(
                        selected_correctness[selected].sum()
                    )

            for batch_index, row in enumerate(rows):
                valid_targets = int(row["valid_targets"])
                cluster = int(row["cluster_id"])
                document_id = str(row["document_id"])
                sequence_losses = losses[batch_index, :valid_targets]
                loss_sum = float(sequence_losses.sum())
                per_cluster_loss[cluster] += loss_sum
                per_cluster_tokens[cluster] += valid_targets
                per_document_loss[document_id] += loss_sum
                per_document_tokens[document_id] += valid_targets
                existing_cluster = per_document_cluster.setdefault(document_id, cluster)
                if existing_cluster != cluster:
                    raise RuntimeError("one eval document appears in multiple clusters")
                for bucket_index in range(POSITION_BUCKETS):
                    start = bucket_index * CONTEXT_LENGTH // POSITION_BUCKETS
                    end = (bucket_index + 1) * CONTEXT_LENGTH // POSITION_BUCKETS
                    bounded_end = min(end, valid_targets)
                    if bounded_end > start:
                        bucket_losses = losses[batch_index, start:bounded_end]
                        position_loss[bucket_index] += float(bucket_losses.sum())
                        position_tokens[bucket_index] += bounded_end - start
    finally:
        model.train(was_training)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    if total_tokens <= 0:
        raise RuntimeError("evaluation scored zero targets")
    if total_bytes <= 0:
        raise RuntimeError("evaluation decoded zero target bytes")

    cluster_metrics: dict[str, object] = {}
    cluster_losses: list[float] = []
    mixture_numerator = 0.0
    mixture_denominator = 0
    for cluster in ACCEPTED_CLUSTERS:
        token_count = per_cluster_tokens[cluster]
        if token_count <= 0:
            raise RuntimeError(f"eval suite has no targets for cluster {cluster}")
        mean_loss = per_cluster_loss[cluster] / token_count
        cluster_losses.append(mean_loss)
        weight = MIXTURE_SOURCE_TOKENS[cluster]
        mixture_numerator += mean_loss * weight
        mixture_denominator += weight
        documents = [
            (per_document_loss[document_id], per_document_tokens[document_id])
            for document_id, document_cluster in per_document_cluster.items()
            if document_cluster == cluster
        ]
        cluster_metrics[str(cluster)] = {
            "loss": mean_loss,
            "perplexity": math.exp(min(mean_loss, 80.0)),
            "target_tokens": token_count,
            "documents": len(documents),
            "bootstrap_95": _bootstrap_interval(
                documents,
                samples=bootstrap_samples,
                seed=17_000 + cluster,
            ),
        }

    document_values = [
        (per_document_loss[document_id], per_document_tokens[document_id])
        for document_id in sorted(per_document_loss)
    ]
    ece = 0.0
    ece_rows: list[dict[str, object]] = []
    for index in range(ECE_BINS):
        count = ece_count[index]
        mean_confidence = ece_confidence[index] / count if count else 0.0
        accuracy = ece_correct[index] / count if count else 0.0
        ece += count / total_tokens * abs(mean_confidence - accuracy)
        ece_rows.append(
            {
                "bin": index,
                "count": count,
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
            }
        )

    mean_loss = total_loss / total_tokens
    peak_vram = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0
    )
    return {
        "suite": suite,
        "eval_manifest_sha256": manifest["manifest_sha256"],
        "loss": mean_loss,
        "perplexity": math.exp(min(mean_loss, 80.0)),
        "bits_per_byte": total_loss / math.log(2.0) / total_bytes,
        "target_tokens": total_tokens,
        "decoded_target_bytes": total_bytes,
        "documents": len(per_document_loss),
        "sequences": len(records),
        "top_k_accuracy": {
            str(k): top_correct[k] / total_tokens for k in sorted(top_correct)
        },
        "calibration": {"ece": ece, "bins": ece_rows},
        "cluster_macro_loss": sum(cluster_losses) / len(cluster_losses),
        "cluster_mixture_weighted_loss": mixture_numerator / mixture_denominator,
        "worst_cluster_loss": max(cluster_losses),
        "per_cluster": cluster_metrics,
        "position_buckets": [
            {
                "index": index,
                "start": index * CONTEXT_LENGTH // POSITION_BUCKETS,
                "end": (index + 1) * CONTEXT_LENGTH // POSITION_BUCKETS,
                "target_tokens": position_tokens[index],
                "loss": (
                    position_loss[index] / position_tokens[index]
                    if position_tokens[index]
                    else math.nan
                ),
            }
            for index in range(POSITION_BUCKETS)
        ],
        "bootstrap_95": _bootstrap_interval(
            document_values, samples=bootstrap_samples, seed=17
        ),
        "wall_seconds": elapsed,
        "tokens_per_second": total_tokens / elapsed,
        "peak_allocated_vram_bytes": peak_vram,
        "precision": precision,
        "batch_size": batch_size,
    }


def run_prompt_cases(
    model: nn.Module,
    *,
    model_max_seq_len: int,
    precision: str,
    questions_only: bool,
    max_cases: int | None,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
    samples_per_prompt: int,
    max_new_tokens: int | None = None,
) -> list[dict[str, object]]:
    """Run the existing prompt definitions and print every answer."""
    try:
        import tiktoken
    except ImportError as error:
        raise RuntimeError(
            "prompt evaluation requires tiktoken; install .[post-training]"
        ) from error
    encoding = tiktoken.get_encoding("gpt2")
    cases = _selected_cases(questions_only=questions_only, max_cases=max_cases)
    results: list[dict[str, object]] = []
    print("\n" + "=" * 80)
    print("Qualitative prompt answers")
    for case_index, case in enumerate(cases):
        prompt_ids = encoding.encode(case.prompt, disallowed_special=())
        if len(prompt_ids) > model_max_seq_len:
            raise RuntimeError(f"prompt {case.name!r} exceeds model context")
        for sample_index in range(samples_per_prompt):
            sample_seed = seed + case_index * 1_000 + sample_index
            budget = (
                case.max_new_tokens
                if max_new_tokens is None
                else min(case.max_new_tokens, max_new_tokens)
            )
            generated_ids = sample_token_ids(
                model,
                prompt_ids,
                max_new_tokens=budget,
                max_seq_len=model_max_seq_len,
                eos_token_id=encoding.eot_token,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                seed=sample_seed,
                precision=precision,
            )
            continuation = encoding.decode(
                [
                    token_id
                    for token_id in generated_ids
                    if token_id != encoding.eot_token
                ]
            )
            print("\n" + "-" * 80)
            print(
                f"[{case.category}] {case.name} | sample={sample_index + 1} "
                f"| seed={sample_seed}"
            )
            print("PROMPT:")
            print(case.prompt)
            print("\nCONTINUATION:")
            print(continuation)
            results.append(
                {
                    "name": case.name,
                    "category": case.category,
                    "sample": sample_index + 1,
                    "seed": sample_seed,
                    "prompt": case.prompt,
                    "prompt_token_ids": prompt_ids,
                    "generated_token_ids": generated_ids,
                    "continuation": continuation,
                }
            )
    return results


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run eval_core metrics and the fixed qualitative prompts"
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
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--samples-per-prompt", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--questions-only", action="store_true")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument(
        "--skip-prompts",
        action="store_true",
        help="diagnostic only; the complete suite includes prompt answers",
    )
    args = parser.parse_args(argv)
    if not args.repo_id:
        parser.error("set --repo-id or SMALL_LLM_HF_REPO_ID")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.bootstrap_samples < 0:
        parser.error("--bootstrap-samples cannot be negative")
    if args.samples_per_prompt <= 0:
        parser.error("--samples-per-prompt must be positive")
    if args.max_new_tokens is not None and args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    eval_dir = args.eval_dir.resolve()
    verify_eval_core(eval_dir)
    device = _resolve_device(args.device)
    precision = _resolve_precision(args.precision, device)
    token = os.environ.get(args.token_env)

    with tempfile.TemporaryDirectory(prefix="small-llm-eval-suite-") as temporary:
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
        print(f"Small-LLM complete evaluation | suite={args.suite}")
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
        print("\n" + "=" * 80)
        print("Intrinsic evaluation summary")
        print(
            json.dumps(
                {
                    "loss": metrics["loss"],
                    "perplexity": metrics["perplexity"],
                    "bits_per_byte": metrics["bits_per_byte"],
                    "top_k_accuracy": metrics["top_k_accuracy"],
                    "calibration": metrics["calibration"],
                    "cluster_macro_loss": metrics["cluster_macro_loss"],
                    "cluster_mixture_weighted_loss": metrics[
                        "cluster_mixture_weighted_loss"
                    ],
                    "worst_cluster_loss": metrics["worst_cluster_loss"],
                    "bootstrap_95": metrics["bootstrap_95"],
                    "tokens_per_second": metrics["tokens_per_second"],
                    "peak_allocated_vram_bytes": metrics[
                        "peak_allocated_vram_bytes"
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        prompts = (
            []
            if args.skip_prompts
            else run_prompt_cases(
                model,
                model_max_seq_len=model_config.max_seq_len,
                precision=precision,
                questions_only=args.questions_only,
                max_cases=args.max_cases,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                seed=args.seed,
                samples_per_prompt=args.samples_per_prompt,
                max_new_tokens=args.max_new_tokens,
            )
        )
        result: dict[str, object] = {
            "schema_version": RESULT_SCHEMA_VERSION,
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
            "prompts": prompts,
            "sampling": {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "seed": args.seed,
                "samples_per_prompt": args.samples_per_prompt,
            },
        }
        result["result_sha256"] = hashlib.sha256(
            canonical_json_bytes(result)
        ).hexdigest()

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"\nSaved complete evaluation bundle to {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
