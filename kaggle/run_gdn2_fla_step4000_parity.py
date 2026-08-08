#!/usr/bin/env python3
"""Real step-4000/next-block GDN-2 FLA forward/backward parity gate.

This diagnostic restores no state and performs no optimizer/scheduler/data
mutation. It loads the already-restored verified step-4000 trainer state, reads
exactly the next unconsumed train block (4000 after cursor 3999), and compares
one full accumulated training backward against a finite FP32 adaptive GDN-2
recurrence oracle.

The outer contract matches the trainer: FP32 master parameters, CUDA FP16
autocast, checkpoint GradScaler scale, microbatching, summed cross entropy, and
normalization by the full block target-token count. Gradients are unscaled for
comparison. No gradient clipping or optimizer step is executed.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
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

EXPECTED_STEP = 4000
EXPECTED_CURSOR = 3999
EXPECTED_BLOCK = 4000
DEFAULT_CHECKPOINT = Path(
    "/kaggle/working/gdn2-step4000-decay-telemetry/step-00004000-trainer-state.pt"
)
DEFAULT_CHECKPOINT_ROOT = Path(
    "/kaggle/working/gdn2-step4000-decay-telemetry/restore/checkpoints/step-00004000"
)
DEFAULT_REPORT = Path("/kaggle/working/gdn2_step4000_fla_parity.json")
OUTPUT_ATOL = 5e-2
OUTPUT_RTOL = 5e-2
GRAD_ATOL = 1e-1
GRAD_RTOL = 1e-1
OUTPUT_TIME_CHUNK = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--candidate-modes",
        default="mixed,fp32",
        help="comma-separated subset of mixed,fp32",
    )
    return parser.parse_args()


def _candidate_modes(raw: str) -> tuple[str, ...]:
    modes = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not modes or any(mode not in {"mixed", "fp32"} for mode in modes):
        raise SystemExit("--candidate-modes must contain only mixed and/or fp32")
    return modes


def _replace_gdn_backends(model: Any, *, mode: str) -> int:
    from model.gdn2_fla import FLAPreferredGDN2Backend
    from model.gdn2_stable import AdaptiveChunkwiseGDN2Backend, StableGatedDeltaNet2
    from run_gdn2_fla_fp32_qualification import _FP32AdaptiveReferenceBackend

    count = 0
    for kind, block in zip(model.layer_kinds, model.blocks, strict=True):
        if kind not in {"gdn", "gdn-2"}:
            continue
        if not isinstance(block.mixer, StableGatedDeltaNet2):
            raise RuntimeError(f"unexpected GDN mixer type: {type(block.mixer).__name__}")
        if mode == "reference":
            block.mixer.backend = _FP32AdaptiveReferenceBackend(
                chunk_size=block.mixer.config.gdn_chunk_size
            )
        else:
            adaptive = AdaptiveChunkwiseGDN2Backend(block.mixer.config.gdn_chunk_size)
            block.mixer.backend = FLAPreferredGDN2Backend(
                chunk_size=block.mixer.config.gdn_chunk_size,
                fallback_backend=adaptive,
                force_fp32=(mode == "fp32"),
            )
        count += 1
    if count == 0:
        raise RuntimeError("model contains no GDN-2 layers")
    return count


def _discover_dataset(explicit: Path | None) -> Path:
    import run_gdn2_step4000_decay_telemetry as telemetry

    root, _ = telemetry.discover_dataset(explicit, EXPECTED_BLOCK)
    return root


def _load_next_batch(dataset_root: Path, checkpoint_root: Path, config: Any):
    from trainer.shards import SchemaV2ShardReader

    checkpoint_meta = json.loads((checkpoint_root / "checkpoint.json").read_text())
    pipeline_state = checkpoint_meta.get("pipeline_state")
    if not isinstance(pipeline_state, dict):
        raise RuntimeError("checkpoint metadata has no pipeline_state")
    if pipeline_state.get("last_consumed_block_id") != EXPECTED_CURSOR:
        raise RuntimeError(
            f"checkpoint cursor={pipeline_state.get('last_consumed_block_id')!r}, "
            f"expected {EXPECTED_CURSOR}"
        )

    reader = SchemaV2ShardReader(
        dataset_root,
        split="train",
        semantic_vocab_size=config.semantic_vocab_size,
        verify_checksums=True,
        context_length=config.max_seq_len,
    )
    reader.load_pipeline_state(pipeline_state)
    batch = reader.next_batch()
    if batch.block_id != EXPECTED_BLOCK:
        raise RuntimeError(f"next batch is block {batch.block_id}, expected {EXPECTED_BLOCK}")
    return batch


def _finite_count(tensor: torch.Tensor) -> int:
    return int((~torch.isfinite(tensor)).sum().item())


def _compare_logits(
    reference: torch.Tensor,
    candidate: torch.Tensor,
) -> dict[str, object]:
    if reference.shape != candidate.shape:
        return {
            "passed": False,
            "reason": f"shape mismatch {tuple(reference.shape)} != {tuple(candidate.shape)}",
        }
    ref_bad = _finite_count(reference)
    cand_bad = _finite_count(candidate)
    mismatch_count = 0
    max_abs = 0.0
    for start in range(0, reference.shape[1], OUTPUT_TIME_CHUNK):
        ref = reference[:, start : start + OUTPUT_TIME_CHUNK].float()
        cand = candidate[:, start : start + OUTPUT_TIME_CHUNK].float()
        diff = (cand - ref).abs()
        max_abs = max(max_abs, float(diff.max().item()))
        tolerance = OUTPUT_ATOL + OUTPUT_RTOL * ref.abs()
        mismatch_count += int((diff > tolerance).sum().item())
    return {
        "passed": ref_bad == 0 and cand_bad == 0 and mismatch_count == 0,
        "reference_nonfinite": ref_bad,
        "candidate_nonfinite": cand_bad,
        "mismatch_count": mismatch_count,
        "max_abs": max_abs,
    }


def _run_model(
    *,
    raw: dict[str, Any],
    config: Any,
    batch: Any,
    mode: str,
    scaler_scale: float,
    reference_logits: list[torch.Tensor] | None,
    reference_grads: dict[str, torch.Tensor] | None,
) -> tuple[dict[str, object], list[torch.Tensor] | None, dict[str, torch.Tensor] | None]:
    from model.model import SmallLLM

    model = SmallLLM(config)
    model.load_state_dict(raw["model"], strict=True)
    gdn_layers = _replace_gdn_backends(model, mode=mode)
    model = model.cuda().train()
    model.zero_grad(set_to_none=True)

    input_ids = batch.input_ids.cuda(non_blocking=True)
    labels = batch.labels.cuda(non_blocking=True)
    microbatch_size = int(raw["config"]["microbatch_size"])
    total_targets = int(batch.target_token_count)
    total_loss = 0.0
    elapsed_forward_backward = 0.0
    peak_allocated = 0
    mode_logits: list[torch.Tensor] | None = [] if mode == "reference" else None
    forward_rows: list[dict[str, object]] = []

    torch.cuda.reset_peak_memory_stats()
    for microbatch_index, start in enumerate(range(0, batch.sequence_count, microbatch_size)):
        stop = min(batch.sequence_count, start + microbatch_size)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(input_ids[start:stop])
            loss_sum = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels[start:stop].reshape(-1),
                reduction="sum",
            )
        if not bool(torch.isfinite(loss_sum)):
            raise RuntimeError(f"{mode} produced non-finite loss at microbatch {microbatch_index}")
        total_loss += float(loss_sum.detach().float())

        cpu_logits = logits.detach().cpu()
        if mode == "reference":
            assert mode_logits is not None
            mode_logits.append(cpu_logits)
            forward_rows.append({"passed": _finite_count(cpu_logits) == 0})
        else:
            assert reference_logits is not None
            forward_rows.append(_compare_logits(reference_logits[microbatch_index], cpu_logits))

        # Match checkpoint GradScaler semantics: scale the normalized loss before
        # backward so the FLA backward receives the same upstream magnitude as
        # the real trainer. Unscale all FP32 parameter gradients below.
        (loss_sum / total_targets * scaler_scale).backward()
        torch.cuda.synchronize()
        elapsed_forward_backward += time.perf_counter() - started
        del logits, loss_sum, cpu_logits
        peak_allocated = max(peak_allocated, int(torch.cuda.max_memory_allocated()))

    gradient_rows: dict[str, object] = {}
    current_grads: dict[str, torch.Tensor] = {}
    all_finite = True
    all_close = True
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            raise RuntimeError(f"{mode} parameter has no gradient: {name}")
        grad = parameter.grad.detach().float() / scaler_scale
        bad = _finite_count(grad)
        all_finite &= bad == 0
        cpu_grad = grad.cpu()
        if mode == "reference":
            current_grads[name] = cpu_grad
            gradient_rows[name] = {
                "passed": bad == 0,
                "reference_nonfinite": bad,
                "candidate_nonfinite": 0,
                "max_abs": 0.0,
            }
        else:
            assert reference_grads is not None
            ref = reference_grads[name]
            ref_bad = _finite_count(ref)
            diff = (cpu_grad - ref).abs()
            max_abs = float(diff.max().item())
            passed = ref_bad == 0 and bad == 0
            if passed:
                tolerance = GRAD_ATOL + GRAD_RTOL * ref.abs()
                passed = bool((diff <= tolerance).all())
            all_close &= passed
            gradient_rows[name] = {
                "passed": passed,
                "reference_nonfinite": ref_bad,
                "candidate_nonfinite": bad,
                "max_abs": max_abs,
            }
        del grad

    forward_pass = all(bool(row.get("passed")) for row in forward_rows)
    gradient_pass = all(bool(row.get("passed")) for row in gradient_rows.values())
    report = {
        "mode": mode,
        "gdn_layers": gdn_layers,
        "loss": total_loss / total_targets,
        "forward_pass": forward_pass,
        "forward_rows": forward_rows,
        "gradients_finite": all_finite,
        "gradient_parity_pass": gradient_pass if mode != "reference" else all_finite,
        "gradient_failures": [
            name for name, row in gradient_rows.items() if not bool(row.get("passed"))
        ],
        "gradient_rows": gradient_rows,
        "elapsed_forward_backward_seconds": elapsed_forward_backward,
        "target_tokens": total_targets,
        "tokens_per_second": total_targets / max(elapsed_forward_backward, 1e-12),
        "peak_allocated_bytes": peak_allocated,
    }

    del model, input_ids, labels
    gc.collect()
    torch.cuda.empty_cache()
    return (
        report,
        mode_logits if mode == "reference" else None,
        current_grads if mode == "reference" else None,
    )


def main() -> int:
    args = parse_args()
    modes = _candidate_modes(args.candidate_modes)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required")

    checkpoint = args.checkpoint.resolve()
    checkpoint_root = args.checkpoint_root.resolve()
    if not checkpoint.is_file() or not checkpoint_root.is_dir():
        raise SystemExit("restored step-4000 checkpoint is missing")
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if raw.get("global_step") != EXPECTED_STEP:
        raise SystemExit(f"checkpoint global_step={raw.get('global_step')!r}, expected {EXPECTED_STEP}")
    if not isinstance(raw.get("model"), dict) or not isinstance(raw.get("model_config"), dict):
        raise SystemExit("checkpoint is missing model/model_config")
    scaler = raw.get("scaler")
    if not isinstance(scaler, dict) or not math.isfinite(float(scaler.get("scale", math.nan))):
        raise SystemExit("checkpoint is missing a finite GradScaler scale")
    scaler_scale = float(scaler["scale"])

    from model.config import ModelConfig

    config = ModelConfig(**raw["model_config"])
    if config.gdn_chunk_size != 32:
        raise SystemExit(f"saved gdn_chunk_size={config.gdn_chunk_size}, expected 32")
    dataset_root = _discover_dataset(args.dataset_dir)
    batch = _load_next_batch(dataset_root, checkpoint_root, config)

    print("=" * 88)
    print("Small-LLM real step-4000 / block-4000 FLA GDN-2 parity gate")
    print("=" * 88)
    print(f"checkpoint_step={raw['global_step']} cursor={EXPECTED_CURSOR} next_block={batch.block_id}")
    print(
        f"batch={batch.sequence_count}x{config.max_seq_len} microbatch={raw['config']['microbatch_size']} "
        f"targets={batch.target_token_count} scaler_scale={scaler_scale:g}"
    )
    print(f"candidate_modes={list(modes)} saved_gdn_chunk_size={config.gdn_chunk_size}")
    print("[safety] no optimizer, scheduler, clipping, acknowledgement, W&B, or checkpoint mutation")

    reference, reference_logits, reference_grads = _run_model(
        raw=raw,
        config=config,
        batch=batch,
        mode="reference",
        scaler_scale=scaler_scale,
        reference_logits=None,
        reference_grads=None,
    )
    if not reference["forward_pass"] or not reference["gradients_finite"]:
        raise SystemExit("FP32 adaptive reference failed; parity gate is invalid")

    candidate_reports: dict[str, object] = {}
    assert reference_logits is not None and reference_grads is not None
    for mode in modes:
        report, _, _ = _run_model(
            raw=raw,
            config=config,
            batch=batch,
            mode=mode,
            scaler_scale=scaler_scale,
            reference_logits=reference_logits,
            reference_grads=reference_grads,
        )
        report["loss_abs_diff"] = abs(float(report["loss"]) - float(reference["loss"]))
        report["passed"] = bool(
            report["forward_pass"]
            and report["gradients_finite"]
            and report["gradient_parity_pass"]
        )
        candidate_reports[mode] = report
        print(
            f"[{mode}] forward={'PASS' if report['forward_pass'] else 'FAIL'} "
            f"grad_finite={'PASS' if report['gradients_finite'] else 'FAIL'} "
            f"grad_parity={'PASS' if report['gradient_parity_pass'] else 'FAIL'} "
            f"loss={float(report['loss']):.6f} tps={float(report['tokens_per_second']):.1f}",
            flush=True,
        )
        if report["gradient_failures"]:
            print(f"[{mode}] first gradient failures: {report['gradient_failures'][:12]}", flush=True)

    result = {
        "experiment": "gdn2_fla_real_step4000_parity_v1",
        "checkpoint_step": EXPECTED_STEP,
        "last_consumed_block_id": EXPECTED_CURSOR,
        "next_block": batch.block_id,
        "dataset_root": str(dataset_root),
        "saved_gdn_chunk_size": config.gdn_chunk_size,
        "fla_runtime_chunk_size": 64,
        "trainer_precision": raw["config"]["precision"],
        "microbatch_size": raw["config"]["microbatch_size"],
        "scaler_scale": scaler_scale,
        "reference": reference,
        "candidates": candidate_reports,
        "optimizer_step_executed": False,
        "production_authorized": False,
    }
    args.report.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    overall = all(bool(row.get("passed")) for row in candidate_reports.values())
    print(f"REPORT={args.report}")
    print(f"REAL_STEP_4000_PARITY={'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
