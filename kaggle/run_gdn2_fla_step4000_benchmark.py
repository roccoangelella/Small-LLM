#!/usr/bin/env python3
"""Warmed real-block throughput benchmark for qualified GDN-2 backends.

Uses the verified step-4000 model and true next block 4000 with the trainer's
FP16 autocast, microbatching, loss normalization, and checkpoint GradScaler
scale. It performs forward/backward only: no clipping, optimizer/scheduler
step, data acknowledgement, W&B, or checkpoint mutation.
"""
from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
KAGGLE_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(KAGGLE_DIR) not in sys.path:
    sys.path.insert(0, str(KAGGLE_DIR))

from run_gdn2_fla_step4000_parity import (
    DEFAULT_CHECKPOINT,
    DEFAULT_CHECKPOINT_ROOT,
    EXPECTED_STEP,
    _discover_dataset,
    _load_next_batch,
    _replace_gdn_backends,
)

DEFAULT_REPORT = Path("/kaggle/working/gdn2_step4000_fla_benchmark.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--measure-repeats", type=int, default=2)
    return parser.parse_args()


def _one_block(model: Any, raw: dict[str, Any], batch: Any, scaler_scale: float) -> tuple[float, bool]:
    model.zero_grad(set_to_none=True)
    inputs = batch.input_ids.cuda(non_blocking=True)
    labels = batch.labels.cuda(non_blocking=True)
    microbatch = int(raw["config"]["microbatch_size"])
    targets = int(batch.target_token_count)
    torch.cuda.synchronize()
    started = time.perf_counter()
    for start in range(0, batch.sequence_count, microbatch):
        stop = min(batch.sequence_count, start + microbatch)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(inputs[start:stop])
            loss_sum = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels[start:stop].reshape(-1),
                reduction="sum",
            )
        (loss_sum / targets * scaler_scale).backward()
        del logits, loss_sum
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    finite = True
    for parameter in model.parameters():
        if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
            finite = False
            break
    del inputs, labels
    return elapsed, finite


def benchmark_mode(raw: dict[str, Any], config: Any, batch: Any, mode: str, repeats: int) -> dict[str, object]:
    from model.model import SmallLLM

    model = SmallLLM(config)
    model.load_state_dict(raw["model"], strict=True)
    gdn_layers = _replace_gdn_backends(model, mode=mode)
    model = model.cuda().train()
    scaler_scale = float(raw["scaler"]["scale"])

    # One real microbatch-sized block pass as warmup is intentionally not timed.
    # It makes Triton/JIT/cache effects explicit rather than charging them to
    # steady-state training throughput. Warmup still performs no optimizer step.
    warmup_batch = type(
        "WarmupBatch",
        (),
        {
            "input_ids": batch.input_ids[: int(raw["config"]["microbatch_size"])],
            "labels": batch.labels[: int(raw["config"]["microbatch_size"])],
            "sequence_count": int(raw["config"]["microbatch_size"]),
            "target_token_count": int(
                batch.labels[: int(raw["config"]["microbatch_size"])].ne(-100).sum().item()
            ),
        },
    )()
    _, warmup_finite = _one_block(model, raw, warmup_batch, scaler_scale)
    if not warmup_finite:
        raise RuntimeError(f"{mode} warmup produced non-finite gradients")

    timings: list[float] = []
    peaks: list[int] = []
    finite_runs: list[bool] = []
    for index in range(repeats):
        torch.cuda.reset_peak_memory_stats()
        elapsed, finite = _one_block(model, raw, batch, scaler_scale)
        timings.append(elapsed)
        peaks.append(int(torch.cuda.max_memory_allocated()))
        finite_runs.append(finite)
        print(
            f"[{mode}] repeat={index + 1} elapsed={elapsed:.3f}s "
            f"tps={batch.target_token_count / elapsed:.1f} finite={finite}",
            flush=True,
        )

    median_elapsed = statistics.median(timings)
    result = {
        "mode": mode,
        "gdn_layers": gdn_layers,
        "elapsed_seconds": timings,
        "median_elapsed_seconds": median_elapsed,
        "median_tokens_per_second": int(batch.target_token_count) / median_elapsed,
        "peak_allocated_bytes": max(peaks),
        "all_gradients_finite": all(finite_runs),
        "warmup_gradients_finite": warmup_finite,
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> int:
    args = parse_args()
    if args.measure_repeats < 1:
        raise SystemExit("--measure-repeats must be >= 1")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required")

    raw = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if raw.get("global_step") != EXPECTED_STEP:
        raise SystemExit("expected verified step-4000 checkpoint")
    from model.config import ModelConfig

    config = ModelConfig(**raw["model_config"])
    if config.gdn_chunk_size != 32:
        raise SystemExit("benchmark refuses changed checkpoint GDN chunk geometry")
    dataset_root = _discover_dataset(args.dataset_dir)
    batch = _load_next_batch(dataset_root, args.checkpoint_root, config)

    print("=" * 88)
    print("Small-LLM warmed step-4000 / block-4000 backend benchmark")
    print("=" * 88)
    print(
        f"block={batch.block_id} targets={batch.target_token_count} "
        f"microbatch={raw['config']['microbatch_size']} repeats={args.measure_repeats}"
    )
    print("[safety] forward/backward only; no clipping or optimizer/scheduler/data mutation")

    results: dict[str, object] = {}
    for mode in ("reference", "mixed", "fp32"):
        results[mode] = benchmark_mode(raw, config, batch, mode, args.measure_repeats)

    adaptive_tps = float(results["reference"]["median_tokens_per_second"])
    mixed_tps = float(results["mixed"]["median_tokens_per_second"])
    fp32_tps = float(results["fp32"]["median_tokens_per_second"])
    report = {
        "experiment": "gdn2_fla_step4000_benchmark_v1",
        "checkpoint_step": EXPECTED_STEP,
        "block": batch.block_id,
        "target_tokens": batch.target_token_count,
        "microbatch_size": raw["config"]["microbatch_size"],
        "saved_gdn_chunk_size": config.gdn_chunk_size,
        "fla_runtime_chunk_size": 64,
        "results": results,
        "speedup_mixed_vs_adaptive": mixed_tps / adaptive_tps,
        "speedup_fp32_vs_adaptive": fp32_tps / adaptive_tps,
        "optimizer_step_executed": False,
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"speedup mixed/adaptive={report['speedup_mixed_vs_adaptive']:.3f}x "
        f"fp32/adaptive={report['speedup_fp32_vs_adaptive']:.3f}x"
    )
    print(f"REPORT={args.report}")
    return 0 if all(bool(row["all_gradients_finite"]) for row in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
