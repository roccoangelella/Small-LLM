"""Kaggle/T4 qualification CLI for the Small LLM model path.

This is intentionally a hardware acceptance harness rather than an ordinary
unit test. It checks GDN-2 mathematical parity on small tensors, benchmarks
full smoke-model training steps at the frozen 2,048-token context, probes both
candidate initializers, and writes a machine-readable JSON report.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from model.config import ModelConfig
from model.gdn2 import PyTorchChunkwiseGDN2Backend, assert_gdn2_backend_parity
from model.initialization import initialize_model
from model.model import SmallLLM

REPORT_SCHEMA_VERSION = 1
DEFAULT_CHUNK_SIZES = (16, 32, 64)
DEFAULT_PRECISIONS = ("fp32", "fp16")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qualify the GDN-2 smoke model on a CUDA GPU, normally a Kaggle T4."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.environ.get("T4_QUALIFICATION_OUTPUT", "/tmp/t4_qualification.json")),
        help="JSON report path.",
    )
    parser.add_argument("--device", default="cuda:0", help="CUDA device, default: cuda:0.")
    parser.add_argument(
        "--require-t4",
        action="store_true",
        help="Fail unless the selected device name contains 'T4'.",
    )
    parser.add_argument(
        "--chunk-sizes",
        nargs="+",
        type=_positive_int,
        default=list(DEFAULT_CHUNK_SIZES),
        metavar="N",
        help="Chunk sizes to qualify and benchmark.",
    )
    parser.add_argument(
        "--precisions",
        nargs="+",
        choices=DEFAULT_PRECISIONS,
        default=list(DEFAULT_PRECISIONS),
        help="Precisions to test.",
    )
    parser.add_argument(
        "--sequence-length",
        type=_positive_int,
        default=2_048,
        help="Full-model benchmark context length.",
    )
    parser.add_argument("--batch-size", type=_positive_int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--measure-steps", type=_positive_int, default=3)
    parser.add_argument(
        "--parity-sequence-length",
        type=_positive_int,
        default=129,
        help="Small recurrence-parity length; 129 exercises a partial chunk for C=64.",
    )
    parser.add_argument(
        "--initialization-sequence-length",
        type=_positive_int,
        default=256,
        help="Context length for the short normal-versus-Xavier FP16 probe.",
    )
    parser.add_argument("--initialization-steps", type=_positive_int, default=3)
    parser.add_argument(
        "--skip-initialization-probe",
        action="store_true",
        help="Skip the short normal-versus-Xavier training probe.",
    )
    parser.add_argument(
        "--include-plan-b",
        action="store_true",
        help="Also benchmark the SWA-512 Plan-B fallback in FP16.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if len(set(args.chunk_sizes)) != len(args.chunk_sizes):
        raise ValueError("chunk_sizes must not contain duplicates")
    if args.sequence_length < 2:
        raise ValueError("sequence_length must be at least 2")
    if args.parity_sequence_length < 2:
        raise ValueError("parity_sequence_length must be at least 2")


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def collect_environment(device: torch.device) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": str(device),
        "device_name": properties.name,
        "compute_capability": [properties.major, properties.minor],
        "device_total_memory_mib": total_bytes / 2**20,
        "device_free_memory_mib_at_start": free_bytes / 2**20,
    }


def _autocast(precision: str):
    if precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def _precision_dtype(precision: str) -> torch.dtype:
    if precision == "fp16":
        return torch.float16
    if precision == "fp32":
        return torch.float32
    raise ValueError(f"unsupported precision: {precision}")


def _parity_tolerances(precision: str) -> dict[str, float]:
    if precision == "fp16":
        return {
            "atol": 2e-3,
            "rtol": 2e-3,
            "gradient_atol": 6e-3,
            "gradient_rtol": 6e-3,
        }
    return {
        "atol": 3e-5,
        "rtol": 3e-5,
        "gradient_atol": 8e-5,
        "gradient_rtol": 8e-5,
    }


def _make_parity_inputs(
    *, device: torch.device, precision: str, sequence_length: int, seed: int
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    dtype = _precision_dtype(precision)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    shape = (1, sequence_length, 2, 16)
    q = torch.randn(shape, generator=generator, device=device, dtype=torch.float32).to(dtype)
    k = torch.randn(shape, generator=generator, device=device, dtype=torch.float32).to(dtype)
    v = torch.randn(shape, generator=generator, device=device, dtype=torch.float32).to(dtype)
    log_decay = (-0.05 * torch.rand(shape, generator=generator, device=device)).to(dtype)
    erase = torch.sigmoid(
        torch.randn(shape, generator=generator, device=device, dtype=torch.float32)
    ).to(dtype)
    write = torch.sigmoid(
        torch.randn(shape, generator=generator, device=device, dtype=torch.float32)
    ).to(dtype)
    initial_state = torch.randn(
        (1, 2, 16, 16), generator=generator, device=device, dtype=torch.float32
    )
    return q, k, v, log_decay, erase, write, initial_state


def run_parity_case(
    *,
    device: torch.device,
    precision: str,
    chunk_size: int,
    sequence_length: int,
    seed: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    tolerances = _parity_tolerances(precision)
    try:
        tensors = _make_parity_inputs(
            device=device,
            precision=precision,
            sequence_length=sequence_length,
            seed=seed,
        )
        assert_gdn2_backend_parity(
            PyTorchChunkwiseGDN2Backend(chunk_size),
            *tensors,
            check_gradients=True,
            **tolerances,
        )
        torch.cuda.synchronize(device)
        return {
            "status": "pass",
            "precision": precision,
            "chunk_size": chunk_size,
            "sequence_length": sequence_length,
            "tolerances": tolerances,
            "elapsed_seconds": time.perf_counter() - started,
        }
    except Exception as exc:
        torch.cuda.synchronize(device)
        return {
            "status": "fail",
            "precision": precision,
            "chunk_size": chunk_size,
            "sequence_length": sequence_length,
            "tolerances": tolerances,
            "elapsed_seconds": time.perf_counter() - started,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def _global_gradient_norm(model: torch.nn.Module) -> Tensor:
    squared = torch.zeros((), device=next(model.parameters()).device, dtype=torch.float32)
    for parameter in model.parameters():
        if parameter.grad is not None:
            squared = squared + parameter.grad.detach().float().square().sum()
    return torch.sqrt(squared)


def _make_training_batch(
    config: ModelConfig,
    *,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
    seed: int,
) -> tuple[Tensor, Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    tokens = torch.randint(
        0,
        config.semantic_vocab_size,
        (batch_size, sequence_length + 1),
        generator=generator,
        device=device,
    )
    return tokens[:, :-1], tokens[:, 1:]


def _build_model(
    *,
    architecture: str,
    chunk_size: int,
    sequence_length: int,
    initializer: str,
    device: torch.device,
    seed: int,
) -> SmallLLM:
    overrides: dict[str, Any] = {
        "architecture": architecture,
        "gdn_chunk_size": chunk_size,
        "max_seq_len": sequence_length,
    }
    if architecture == "swa_hybrid":
        overrides["attention_window"] = 512
    torch.manual_seed(seed)
    model = SmallLLM(ModelConfig.smoke(**overrides)).to(device)
    initialize_model(model, initializer)
    model.train()
    return model


def _run_training_step(
    *,
    model: SmallLLM,
    input_ids: Tensor,
    targets: Tensor,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    precision: str,
) -> dict[str, Any]:
    optimizer.zero_grad(set_to_none=True)
    old_scale = float(scaler.get_scale())
    with _autocast(precision):
        logits = model(input_ids)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    gradient_norm = _global_gradient_norm(model)
    finite = bool(torch.isfinite(loss).item() and torch.isfinite(gradient_norm).item())
    scaler.step(optimizer)
    scaler.update()
    new_scale = float(scaler.get_scale())
    overflow = precision == "fp16" and new_scale < old_scale
    return {
        "loss": float(loss.detach()),
        "gradient_norm": float(gradient_norm.detach()),
        "finite": finite,
        "overflow": overflow,
        "scale_before": old_scale,
        "scale_after": new_scale,
    }


def _is_cuda_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def benchmark_model(
    *,
    device: torch.device,
    architecture: str,
    chunk_size: int,
    precision: str,
    initializer: str,
    sequence_length: int,
    batch_size: int,
    warmup_steps: int,
    measure_steps: int,
    seed: int,
) -> dict[str, Any]:
    model: SmallLLM | None = None
    optimizer: torch.optim.Optimizer | None = None
    input_ids: Tensor | None = None
    targets: Tensor | None = None
    scaler: torch.amp.GradScaler | None = None
    try:
        torch.cuda.empty_cache()
        model = _build_model(
            architecture=architecture,
            chunk_size=chunk_size,
            sequence_length=sequence_length,
            initializer=initializer,
            device=device,
            seed=seed,
        )
        input_ids, targets = _make_training_batch(
            model.config,
            batch_size=batch_size,
            sequence_length=sequence_length,
            device=device,
            seed=seed + 1,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
        scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")

        warmup_results = []
        for _ in range(warmup_steps):
            warmup_results.append(
                _run_training_step(
                    model=model,
                    input_ids=input_ids,
                    targets=targets,
                    optimizer=optimizer,
                    scaler=scaler,
                    precision=precision,
                )
            )
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        measured = []
        for _ in range(measure_steps):
            measured.append(
                _run_training_step(
                    model=model,
                    input_ids=input_ids,
                    targets=targets,
                    optimizer=optimizer,
                    scaler=scaler,
                    precision=precision,
                )
            )
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        finite = all(step["finite"] for step in measured)
        overflow_count = sum(bool(step["overflow"]) for step in measured)
        tokens = batch_size * sequence_length * measure_steps
        return {
            "status": "pass" if finite and overflow_count == 0 else "fail",
            "architecture": architecture,
            "backend": "pytorch_chunkwise" if architecture == "gdn2_hybrid" else "pytorch_attention",
            "chunk_size": chunk_size if architecture == "gdn2_hybrid" else None,
            "precision": precision,
            "initializer": initializer,
            "sequence_length": sequence_length,
            "batch_size": batch_size,
            "warmup_steps": warmup_steps,
            "measure_steps": measure_steps,
            "elapsed_seconds": elapsed,
            "mean_step_ms": 1_000.0 * elapsed / measure_steps,
            "tokens_per_second": tokens / elapsed,
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
            "overflow_count": overflow_count,
            "losses": [step["loss"] for step in measured],
            "gradient_norms": [step["gradient_norm"] for step in measured],
            "warmup": warmup_results,
        }
    except Exception as exc:
        return {
            "status": "oom" if _is_cuda_oom(exc) else "fail",
            "architecture": architecture,
            "backend": "pytorch_chunkwise" if architecture == "gdn2_hybrid" else "pytorch_attention",
            "chunk_size": chunk_size if architecture == "gdn2_hybrid" else None,
            "precision": precision,
            "initializer": initializer,
            "sequence_length": sequence_length,
            "batch_size": batch_size,
            "warmup_steps": warmup_steps,
            "measure_steps": measure_steps,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    finally:
        del scaler
        del targets
        del input_ids
        del optimizer
        del model
        gc.collect()
        torch.cuda.empty_cache()


def choose_recommendation(
    parity_results: Sequence[dict[str, Any]],
    benchmarks: Sequence[dict[str, Any]],
    requested_precisions: Iterable[str],
) -> dict[str, Any]:
    requested = tuple(requested_precisions)
    precision_order = [p for p in ("fp16", "fp32") if p in set(requested)]
    parity_pass = {
        (result.get("chunk_size"), result.get("precision"))
        for result in parity_results
        if result.get("status") == "pass"
    }
    fully_qualified_chunks = {
        result.get("chunk_size")
        for result in parity_results
        if all((result.get("chunk_size"), precision) in parity_pass for precision in requested)
    }
    candidates = []
    for result in benchmarks:
        if result.get("architecture") != "gdn2_hybrid" or result.get("status") != "pass":
            continue
        chunk_size = result.get("chunk_size")
        precision = result.get("precision")
        if chunk_size not in fully_qualified_chunks or (chunk_size, precision) not in parity_pass:
            continue
        candidates.append(result)

    for precision in precision_order:
        matching = [result for result in candidates if result.get("precision") == precision]
        if matching:
            best = max(matching, key=lambda item: float(item["tokens_per_second"]))
            return {
                "status": "candidate",
                "architecture": "gdn2_hybrid",
                "backend": "pytorch_chunkwise",
                "chunk_size": best["chunk_size"],
                "precision": precision,
                "tokens_per_second": best["tokens_per_second"],
                "reason": "Fastest parity-qualified finite smoke-model result in the preferred precision.",
            }

    plan_b = [
        result
        for result in benchmarks
        if result.get("architecture") == "swa_hybrid" and result.get("status") == "pass"
    ]
    if plan_b:
        best = max(plan_b, key=lambda item: float(item["tokens_per_second"]))
        return {
            "status": "fallback_candidate",
            "architecture": "swa_hybrid",
            "backend": "pytorch_attention",
            "chunk_size": None,
            "precision": best["precision"],
            "tokens_per_second": best["tokens_per_second"],
            "reason": "No GDN-2 chunk candidate passed both parity and training; Plan B ran successfully.",
        }
    return {
        "status": "blocked",
        "architecture": None,
        "backend": None,
        "chunk_size": None,
        "precision": None,
        "reason": "No parity-qualified finite training candidate completed.",
    }


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; enable a GPU accelerator in the Kaggle notebook")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    environment = collect_environment(device)
    if args.require_t4 and "T4" not in environment["device_name"].upper():
        raise RuntimeError(
            f"--require-t4 was set, but selected device is {environment['device_name']!r}"
        )

    parity_results = []
    for precision in args.precisions:
        for chunk_size in args.chunk_sizes:
            parity_results.append(
                run_parity_case(
                    device=device,
                    precision=precision,
                    chunk_size=chunk_size,
                    sequence_length=args.parity_sequence_length,
                    seed=args.seed,
                )
            )

    benchmarks = []
    for precision in args.precisions:
        for chunk_size in args.chunk_sizes:
            benchmarks.append(
                benchmark_model(
                    device=device,
                    architecture="gdn2_hybrid",
                    chunk_size=chunk_size,
                    precision=precision,
                    initializer="normal",
                    sequence_length=args.sequence_length,
                    batch_size=args.batch_size,
                    warmup_steps=args.warmup_steps,
                    measure_steps=args.measure_steps,
                    seed=args.seed,
                )
            )

    if args.include_plan_b:
        benchmarks.append(
            benchmark_model(
                device=device,
                architecture="swa_hybrid",
                chunk_size=max(args.chunk_sizes),
                precision="fp16",
                initializer="normal",
                sequence_length=args.sequence_length,
                batch_size=args.batch_size,
                warmup_steps=args.warmup_steps,
                measure_steps=args.measure_steps,
                seed=args.seed,
            )
        )

    initialization_probe = []
    if not args.skip_initialization_probe:
        probe_length = min(args.initialization_sequence_length, args.sequence_length)
        for initializer in ("normal", "xavier"):
            initialization_probe.append(
                benchmark_model(
                    device=device,
                    architecture="gdn2_hybrid",
                    chunk_size=max(args.chunk_sizes),
                    precision="fp16",
                    initializer=initializer,
                    sequence_length=probe_length,
                    batch_size=args.batch_size,
                    warmup_steps=0,
                    measure_steps=args.initialization_steps,
                    seed=args.seed,
                )
            )

    recommendation = choose_recommendation(parity_results, benchmarks, args.precisions)
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "environment": environment,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
        },
        "parity": parity_results,
        "benchmarks": benchmarks,
        "initialization_probe": initialization_probe,
        "recommendation": recommendation,
        "notes": [
            "A candidate result is not an automatic architecture or configuration change.",
            "Review repeatability, losses, gradient norms, overflow counts, memory, and throughput before freezing a choice.",
        ],
    }
    parity_ok = all(result["status"] == "pass" for result in parity_results)
    exit_code = 0 if parity_ok and recommendation["status"] != "blocked" else 1
    return report, exit_code


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report, exit_code = run(args)
        _write_report(args.output, report)
        print(json.dumps(report["recommendation"], indent=2, sort_keys=True))
        print(f"Full report: {args.output}")
        return exit_code
    except Exception as exc:
        failure = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        _write_report(args.output, failure)
        print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr)
        print(f"Failure report: {args.output}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
