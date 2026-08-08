#!/usr/bin/env python3
"""One-click Kaggle/T4 qualification probe for the FLA GDN-2 training kernel.

Run with:
    python kaggle/run_gdn2_fla_t4_probe.py

The probe does not modify Small-LLM training code or upgrade PyTorch. It checks
whether Flash Linear Attention's optimized GDN-2 kernel preserves our recurrence
and avoids the strong-decay slowdown of AdaptiveChunkwiseGDN2Backend.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import importlib.metadata
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FLA_VERSION = "0.5.1"
DEFAULT_REPORT = (
    Path("/kaggle/working/gdn2_fla_t4_probe.json")
    if Path("/kaggle/working").is_dir()
    else ROOT / "gdn2_fla_t4_probe.json"
)

# 20M smoke-model GDN geometry.
HEADS = 4
KEY_DIM = 64
VALUE_DIM = 64
CURRENT_MAX_CHUNK = 64
STRESS_LOG_DECAY = -6.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qualify FLA GDN-2 against Small-LLM on a CUDA GPU."
    )
    parser.add_argument(
        "--no-install",
        action="store_true",
        help="Do not auto-install flash-linear-attention==0.5.1 if missing.",
    )
    parser.add_argument(
        "--benchmark-batch",
        type=int,
        default=4,
        help="Benchmark batch size (default: 4, matching the 500M microbatch).",
    )
    parser.add_argument(
        "--benchmark-seq",
        type=int,
        default=2048,
        help="Benchmark sequence length (default: 2048).",
    )
    parser.add_argument(
        "--forward-iters", type=int, default=3, help="Measured forward iterations."
    )
    parser.add_argument(
        "--train-iters", type=int, default=1, help="Measured forward+backward iterations."
    )
    parser.add_argument(
        "--skip-train-benchmark",
        action="store_true",
        help="Run correctness and forward benchmarks only.",
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _ensure_fla(*, allow_install: bool) -> str:
    try:
        import fla  # noqa: F401
    except Exception as first_error:
        if not allow_install:
            raise RuntimeError("FLA is not importable and --no-install was requested.") from first_error
        print(
            f"[setup] installing flash-linear-attention=={FLA_VERSION} --no-deps "
            "(PyTorch will NOT be changed)"
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--quiet",
                "--no-deps",
                f"flash-linear-attention=={FLA_VERSION}",
            ],
            check=True,
        )
        importlib.invalidate_caches()
        try:
            import fla  # noqa: F401
        except Exception as second_error:
            raise RuntimeError(
                "FLA still cannot import after the no-deps install. The existing Kaggle "
                "Torch/Triton/dependency stack is incompatible. This probe intentionally "
                "does not upgrade PyTorch automatically."
            ) from second_error

    version = _package_version("flash-linear-attention") or "unknown"
    if version != FLA_VERSION:
        print(
            f"[setup] warning: installed FLA is {version}; probe was authored against {FLA_VERSION}."
        )
    return version


def _environment(torch: Any, fla_version: str) -> dict[str, Any]:
    try:
        import triton

        triton_version: str | None = getattr(triton, "__version__", "unknown")
    except Exception:
        triton_version = None
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this probe.")
    index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    capability = torch.cuda.get_device_capability(index)
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "triton": triton_version,
        "flash_linear_attention": fla_version,
        "cuda_runtime": torch.version.cuda,
        "device": props.name,
        "compute_capability": f"{capability[0]}.{capability[1]}",
        "vram_gib": props.total_memory / (1024**3),
    }


def _make_inputs(
    torch: Any,
    F: Any,
    *,
    batch: int,
    sequence: int,
    decay_case: str,
    seed: int,
) -> tuple[Any, ...]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    shape_k = (batch, sequence, HEADS, KEY_DIM)
    shape_v = (batch, sequence, HEADS, VALUE_DIM)

    q = F.normalize(
        torch.randn(*shape_k, device="cuda", dtype=torch.float32, generator=generator),
        dim=-1,
        eps=1e-6,
    ).half()
    k = F.normalize(
        torch.randn(*shape_k, device="cuda", dtype=torch.float32, generator=generator),
        dim=-1,
        eps=1e-6,
    ).half()
    v = (
        torch.randn(*shape_v, device="cuda", dtype=torch.float32, generator=generator) * 0.5
    ).half()

    if decay_case == "normal":
        # Mild forgetting; our 64-token adaptive chunk should stay whole.
        g = torch.empty(*shape_k, device="cuda", dtype=torch.float32).uniform_(
            -0.20, -0.01, generator=generator
        )
    elif decay_case == "stress_-6":
        # Same magnitude as the Small-LLM strong-decay regression test. A 64-token
        # proposal has span 378, so our adaptive backend must subdivide repeatedly.
        g = torch.full(shape_k, STRESS_LOG_DECAY, device="cuda", dtype=torch.float32)
    elif decay_case == "extreme_-10":
        g = torch.full(shape_k, -10.0, device="cuda", dtype=torch.float32)
    else:
        raise ValueError(f"unknown decay case: {decay_case}")

    erase = torch.sigmoid(
        torch.randn(*shape_k, device="cuda", dtype=torch.float32, generator=generator)
    ).half()
    write = torch.sigmoid(
        torch.randn(*shape_v, device="cuda", dtype=torch.float32, generator=generator)
    ).half()
    h0 = (
        torch.randn(
            batch,
            HEADS,
            KEY_DIM,
            VALUE_DIM,
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        )
        * 0.1
    )
    return q, k, v, g, erase, write, h0


def _clone_grad_inputs(tensors: tuple[Any, ...]) -> list[Any]:
    return [tensor.detach().clone().requires_grad_(True) for tensor in tensors]


def _error_metrics(reference: Any, candidate: Any) -> dict[str, float]:
    ref = reference.detach().float()
    got = candidate.detach().float()
    diff = (ref - got).abs()
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
    }


def _assert_close(
    torch: Any,
    label: str,
    reference: Any,
    candidate: Any,
    *,
    atol: float,
    rtol: float,
) -> dict[str, float]:
    metrics = _error_metrics(reference, candidate)
    torch.testing.assert_close(candidate, reference, atol=atol, rtol=rtol)
    print(
        f"    {label:<8} PASS  max_abs={metrics['max_abs']:.3e} "
        f"mean_abs={metrics['mean_abs']:.3e}"
    )
    return metrics


def _correctness_case(
    torch: Any,
    *,
    label: str,
    inputs: tuple[Any, ...],
    recurrent_fn: Callable[..., tuple[Any, Any]],
    fla_fn: Callable[..., tuple[Any, Any]],
    check_gradients: bool,
) -> dict[str, Any]:
    print(f"[correctness] {label}")
    result: dict[str, Any] = {"label": label, "passed": False}
    try:
        ref_inputs = _clone_grad_inputs(inputs)
        got_inputs = _clone_grad_inputs(inputs)
        ref_o, ref_h = recurrent_fn(*ref_inputs)
        got_o, got_h = fla_fn(*got_inputs)
        if not bool(torch.isfinite(got_o).all()) or not bool(torch.isfinite(got_h).all()):
            raise AssertionError("FLA produced non-finite output or state")

        errors: dict[str, Any] = {
            "output": _assert_close(torch, "output", ref_o, got_o, atol=2e-2, rtol=2e-2),
            "state": _assert_close(torch, "state", ref_h, got_h, atol=2e-2, rtol=2e-2),
        }
        if check_gradients:
            generator = torch.Generator(device="cuda")
            generator.manual_seed(991)
            do = torch.randn(
                ref_o.shape, device="cuda", dtype=torch.float32, generator=generator
            ).to(ref_o.dtype)
            dh = torch.randn(
                ref_h.shape, device="cuda", dtype=torch.float32, generator=generator
            )
            ref_loss = (ref_o.float() * do.float()).sum() + (ref_h * dh).sum()
            got_loss = (got_o.float() * do.float()).sum() + (got_h * dh).sum()
            ref_grads = torch.autograd.grad(ref_loss, ref_inputs)
            got_grads = torch.autograd.grad(got_loss, got_inputs)
            names = ("q", "k", "v", "g", "erase", "write", "h0")
            errors["gradients"] = {}
            for name, ref_grad, got_grad in zip(names, ref_grads, got_grads, strict=True):
                errors["gradients"][name] = _assert_close(
                    torch, f"d{name}", ref_grad, got_grad, atol=5e-2, rtol=5e-2
                )
        result.update({"passed": True, "errors": errors})
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        result["traceback"] = traceback.format_exc()
        print(f"    FAIL: {result['error']}")
    finally:
        gc.collect()
        torch.cuda.empty_cache()
    return result


def _benchmark_forward(
    torch: Any,
    *,
    label: str,
    fn: Callable[..., tuple[Any, Any]],
    inputs: tuple[Any, ...],
    iterations: int,
) -> dict[str, Any]:
    batch, sequence = inputs[0].shape[:2]
    print(f"[benchmark:fwd] {label}  B={batch} T={sequence}")
    result: dict[str, Any] = {
        "label": label,
        "mode": "forward",
        "batch": batch,
        "sequence": sequence,
        "passed": False,
    }
    try:
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            fn(*inputs)  # compile/autotune warmup
        torch.cuda.synchronize()
        times: list[float] = []
        with torch.no_grad():
            for _ in range(iterations):
                start = time.perf_counter()
                out, state = fn(*inputs)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - start)
                if not bool(torch.isfinite(out).all()) or not bool(torch.isfinite(state).all()):
                    raise AssertionError("non-finite output/state")
        median = statistics.median(times)
        result.update(
            {
                "passed": True,
                "times_seconds": times,
                "median_seconds": median,
                "backend_tokens_per_second": batch * sequence / median,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / (1024**3),
            }
        )
        print(
            f"    median={median:.4f}s  backend_tok/s={result['backend_tokens_per_second']:.1f} "
            f"peak={result['peak_allocated_gib']:.2f}GiB"
        )
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        result["traceback"] = traceback.format_exc()
        print(f"    FAIL: {result['error']}")
    finally:
        gc.collect()
        torch.cuda.empty_cache()
    return result


def _benchmark_train(
    torch: Any,
    *,
    label: str,
    fn: Callable[..., tuple[Any, Any]],
    template_inputs: tuple[Any, ...],
    iterations: int,
) -> dict[str, Any]:
    batch, sequence = template_inputs[0].shape[:2]
    print(f"[benchmark:fwd+bwd] {label}  B={batch} T={sequence}")
    result: dict[str, Any] = {
        "label": label,
        "mode": "forward_backward",
        "batch": batch,
        "sequence": sequence,
        "passed": False,
    }

    def one_iteration() -> tuple[float, float]:
        leaves = _clone_grad_inputs(template_inputs)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        out, state = fn(*leaves)
        loss = out.float().square().mean() + state.float().square().mean()
        loss.backward()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if not bool(torch.isfinite(out).all()) or not bool(torch.isfinite(state).all()):
            raise AssertionError("non-finite output/state")
        peak = torch.cuda.max_memory_allocated() / (1024**3)
        del leaves, out, state, loss
        gc.collect()
        torch.cuda.empty_cache()
        return elapsed, peak

    try:
        one_iteration()  # compile/autotune warmup
        times: list[float] = []
        peaks: list[float] = []
        for _ in range(iterations):
            elapsed, peak = one_iteration()
            times.append(elapsed)
            peaks.append(peak)
        median = statistics.median(times)
        result.update(
            {
                "passed": True,
                "times_seconds": times,
                "median_seconds": median,
                "backend_tokens_per_second": batch * sequence / median,
                "peak_allocated_gib": max(peaks),
            }
        )
        print(
            f"    median={median:.4f}s  backend_tok/s={result['backend_tokens_per_second']:.1f} "
            f"peak={result['peak_allocated_gib']:.2f}GiB"
        )
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        result["traceback"] = traceback.format_exc()
        print(f"    FAIL: {result['error']}")
    finally:
        gc.collect()
        torch.cuda.empty_cache()
    return result


def _find(
    rows: list[dict[str, Any]], *, backend: str, decay: str, mode: str
) -> dict[str, Any] | None:
    label = f"{backend}:{decay}"
    return next(
        (
            row
            for row in rows
            if row.get("label") == label and row.get("mode") == mode and row.get("passed")
        ),
        None,
    )


def _ratio(num: dict[str, Any] | None, den: dict[str, Any] | None) -> float | None:
    if not num or not den:
        return None
    return float(num["backend_tokens_per_second"]) / float(den["backend_tokens_per_second"])


def main() -> int:
    args = _parse_args()
    if min(args.benchmark_batch, args.benchmark_seq, args.forward_iters, args.train_iters) <= 0:
        raise SystemExit("batch, sequence, and iteration counts must be positive")

    # Import Torch first: the probe must never silently replace the notebook's runtime.
    import torch
    from torch.nn import functional as F

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required. Enable a Kaggle GPU accelerator.")

    print("=" * 78)
    print("Small-LLM FLA GDN-2 T4 qualification probe")
    print("=" * 78)
    print(f"[env] torch={torch.__version__} cuda={torch.version.cuda}")
    print(f"[env] gpu={torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(f"[env] compute_capability={torch.cuda.get_device_capability()}")

    fla_version = _ensure_fla(allow_install=not args.no_install)
    from fla.ops.gdn2 import chunk_gdn2
    from model.gdn2 import gdn2_recurrent_reference
    from model.gdn2_stable import AdaptiveChunkwiseGDN2Backend

    env = _environment(torch, fla_version)
    print(
        f"[env] triton={env['triton']} fla={env['flash_linear_attention']} "
        f"vram={env['vram_gib']:.1f}GiB"
    )

    adaptive = AdaptiveChunkwiseGDN2Backend(chunk_size=CURRENT_MAX_CHUNK)

    def recurrent_fn(q: Any, k: Any, v: Any, g: Any, b: Any, w: Any, h0: Any) -> tuple[Any, Any]:
        return gdn2_recurrent_reference(q, k, v, g, b, w, h0)

    def adaptive_fn(q: Any, k: Any, v: Any, g: Any, b: Any, w: Any, h0: Any) -> tuple[Any, Any]:
        return adaptive(q, k, v, g, b, w, h0)

    def fla_fn(q: Any, k: Any, v: Any, g: Any, b: Any, w: Any, h0: Any) -> tuple[Any, Any]:
        out, final_state = chunk_gdn2(
            q=q,
            k=k,
            v=v,
            g=g,
            b=b,
            w=w,
            scale=1.0 / math.sqrt(KEY_DIM),
            initial_state=h0,
            output_final_state=True,
            use_qk_l2norm_in_kernel=False,
            use_gate_in_kernel=False,
        )
        if final_state is None:
            raise RuntimeError("FLA did not return final state")
        return out, final_state

    report: dict[str, Any] = {
        "probe": "gdn2_fla_t4",
        "fla_target_version": FLA_VERSION,
        "environment": env,
        "geometry": {
            "heads": HEADS,
            "key_dim": KEY_DIM,
            "value_dim": VALUE_DIM,
            "current_max_chunk": CURRENT_MAX_CHUNK,
            "benchmark_batch": args.benchmark_batch,
            "benchmark_sequence": args.benchmark_seq,
        },
        "correctness": [],
        "benchmarks": [],
    }

    for decay, seed in (("normal", 100), ("stress_-6", 200)):
        report["correctness"].append(
            _correctness_case(
                torch,
                label=f"fla_vs_recurrent:{decay}:fp16",
                inputs=_make_inputs(torch, F, batch=1, sequence=64, decay_case=decay, seed=seed),
                recurrent_fn=recurrent_fn,
                fla_fn=fla_fn,
                check_gradients=True,
            )
        )

    report["correctness"].append(
        _correctness_case(
            torch,
            label="fla_vs_recurrent:extreme_-10:fp16",
            inputs=_make_inputs(
                torch, F, batch=1, sequence=64, decay_case="extreme_-10", seed=300
            ),
            recurrent_fn=recurrent_fn,
            fla_fn=fla_fn,
            check_gradients=False,
        )
    )

    for decay, seed in (("normal", 400), ("stress_-6", 500)):
        template = _make_inputs(
            torch,
            F,
            batch=args.benchmark_batch,
            sequence=args.benchmark_seq,
            decay_case=decay,
            seed=seed,
        )
        for backend, fn in (("adaptive", adaptive_fn), ("fla", fla_fn)):
            report["benchmarks"].append(
                _benchmark_forward(
                    torch,
                    label=f"{backend}:{decay}",
                    fn=fn,
                    inputs=template,
                    iterations=args.forward_iters,
                )
            )
        if not args.skip_train_benchmark:
            for backend, fn in (("adaptive", adaptive_fn), ("fla", fla_fn)):
                report["benchmarks"].append(
                    _benchmark_train(
                        torch,
                        label=f"{backend}:{decay}",
                        fn=fn,
                        template_inputs=template,
                        iterations=args.train_iters,
                    )
                )
        del template
        gc.collect()
        torch.cuda.empty_cache()

    correctness_pass = all(row.get("passed", False) for row in report["correctness"])
    rows = report["benchmarks"]
    summary: dict[str, Any] = {"correctness_gate": "PASS" if correctness_pass else "FAIL"}
    for mode in ("forward", "forward_backward"):
        an = _find(rows, backend="adaptive", decay="normal", mode=mode)
        ast = _find(rows, backend="adaptive", decay="stress_-6", mode=mode)
        fn = _find(rows, backend="fla", decay="normal", mode=mode)
        fst = _find(rows, backend="fla", decay="stress_-6", mode=mode)
        summary[f"{mode}_fla_speedup_over_adaptive_normal"] = _ratio(fn, an)
        summary[f"{mode}_fla_speedup_over_adaptive_stress"] = _ratio(fst, ast)
        summary[f"{mode}_adaptive_stress_retention"] = _ratio(ast, an)
        summary[f"{mode}_fla_stress_retention"] = _ratio(fst, fn)

    fla_stress = _find(rows, backend="fla", decay="stress_-6", mode="forward")
    if correctness_pass and fla_stress:
        summary["probe_verdict"] = (
            "PROMISING: FLA preserved the recurrence in this probe and executed the strong-decay "
            "case on this GPU. Review speedup/retention before integrating it."
        )
        exit_code = 0
    else:
        summary["probe_verdict"] = (
            "NOT QUALIFIED: correctness or GPU execution failed. Do not replace the current backend."
        )
        exit_code = 1

    report["summary"] = summary
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"correctness: {summary['correctness_gate']}")
    for key, value in summary.items():
        if key in {"correctness_gate", "probe_verdict"} or value is None:
            continue
        print(f"{key}: {value:.3f}x")
    print(summary["probe_verdict"])
    print(f"JSON report: {args.report}")
    print()
    print("Interpretation:")
    print("  adaptive_stress_retention = strong-decay speed / normal-decay speed")
    print("  fla_stress_retention      = same ratio for FLA; closer to 1.0 is better")
    print("  fla_speedup_over_adaptive_stress = direct speedup in the problematic regime")
    print("  backend_tok/s is one GDN backend call, NOT whole-model training tok/s")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
