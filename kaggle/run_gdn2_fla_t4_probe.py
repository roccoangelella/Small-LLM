#!/usr/bin/env python3
"""One-click Kaggle probe for FLA GDN-2 on the Small-LLM geometry.

Default (fast qualification):
    python kaggle/run_gdn2_fla_t4_probe.py

Explicit training/backward qualification:
    python kaggle/run_gdn2_fla_t4_probe.py --with-backward

The default intentionally avoids FLA's expensive first-time Triton backward
autotuning on GPUs without packaged tuning configs (notably Tesla T4). It still
checks forward numerical parity in normal/strong/extreme decay regimes and
benchmarks the exact problematic strong-decay forward path against Small-LLM's
adaptive PyTorch backend.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import importlib.metadata
import json
import math
import os
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
HEADS = 4
KEY_DIM = 64
VALUE_DIM = 64
CHUNK_SIZE = 64
DEFAULT_REPORT = Path("/kaggle/working/gdn2_fla_t4_probe.json") if Path("/kaggle/working").is_dir() else ROOT / "gdn2_fla_t4_probe.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Qualify FLA GDN-2 on the current CUDA GPU.")
    p.add_argument("--with-backward", action="store_true", help="Also compile/autotune and test the FLA backward path.")
    p.add_argument("--no-install", action="store_true", help="Do not install fla-core if missing.")
    p.add_argument("--benchmark-batch", type=int, default=4)
    p.add_argument("--benchmark-seq", type=int, default=2048)
    p.add_argument("--forward-iters", type=int, default=3)
    p.add_argument("--train-iters", type=int, default=1)
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return p.parse_args()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def clear_fla_modules() -> None:
    for name in tuple(sys.modules):
        if name == "fla" or name.startswith("fla."):
            sys.modules.pop(name, None)
    importlib.invalidate_caches()


def ensure_fla(allow_install: bool) -> dict[str, str | None]:
    try:
        from fla.ops.gdn2 import chunk_gdn2  # noqa: F401
    except Exception as first_error:
        if not allow_install:
            raise RuntimeError(f"fla.ops.gdn2 unavailable: {first_error}") from first_error
        print(f"[setup] installing fla-core=={FLA_VERSION} --no-deps (Torch/Triton will NOT be changed)", flush=True)
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", f"fla-core=={FLA_VERSION}"],
            check=True,
        )
        clear_fla_modules()
        try:
            from fla.ops.gdn2 import chunk_gdn2  # noqa: F401
        except Exception as second_error:
            raise RuntimeError(f"fla-core installed but fla.ops.gdn2 still fails: {second_error}") from second_error
    return {
        "fla_core": package_version("fla-core"),
        "flash_linear_attention": package_version("flash-linear-attention"),
    }


def make_inputs(torch: Any, F: Any, *, batch: int, seq: int, decay: str, seed: int) -> tuple[Any, ...]:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    sk = (batch, seq, HEADS, KEY_DIM)
    sv = (batch, seq, HEADS, VALUE_DIM)
    q = F.normalize(torch.randn(sk, device="cuda", dtype=torch.float32, generator=gen), dim=-1).half()
    k = F.normalize(torch.randn(sk, device="cuda", dtype=torch.float32, generator=gen), dim=-1).half()
    v = (0.5 * torch.randn(sv, device="cuda", dtype=torch.float32, generator=gen)).half()
    if decay == "normal":
        g = torch.empty(sk, device="cuda", dtype=torch.float32).uniform_(-0.20, -0.01, generator=gen)
    elif decay == "stress_-6":
        g = torch.full(sk, -6.0, device="cuda", dtype=torch.float32)
    elif decay == "extreme_-10":
        g = torch.full(sk, -10.0, device="cuda", dtype=torch.float32)
    else:
        raise ValueError(decay)
    b = torch.sigmoid(torch.randn(sk, device="cuda", dtype=torch.float32, generator=gen)).half()
    w = torch.sigmoid(torch.randn(sv, device="cuda", dtype=torch.float32, generator=gen)).half()
    h0 = 0.1 * torch.randn((batch, HEADS, KEY_DIM, VALUE_DIM), device="cuda", dtype=torch.float32, generator=gen)
    return q, k, v, g, b, w, h0


def clone_leaves(xs: tuple[Any, ...]) -> list[Any]:
    return [x.detach().clone().requires_grad_(True) for x in xs]


def error_metrics(ref: Any, got: Any) -> dict[str, float]:
    d = (ref.detach().float() - got.detach().float()).abs()
    return {"max_abs": float(d.max().item()), "mean_abs": float(d.mean().item())}


def assert_close(torch: Any, label: str, ref: Any, got: Any, atol: float, rtol: float) -> dict[str, float]:
    m = error_metrics(ref, got)
    torch.testing.assert_close(got, ref, atol=atol, rtol=rtol)
    print(f"    {label:<8} PASS  max_abs={m['max_abs']:.3e} mean_abs={m['mean_abs']:.3e}", flush=True)
    return m


def forward_correctness(
    torch: Any,
    label: str,
    inputs: tuple[Any, ...],
    recurrent_fn: Callable[..., tuple[Any, Any]],
    fla_fn: Callable[..., tuple[Any, Any]],
) -> dict[str, Any]:
    print(f"[correctness:fwd] {label}", flush=True)
    row: dict[str, Any] = {"label": label, "passed": False}
    try:
        with torch.no_grad():
            ref_o, ref_h = recurrent_fn(*inputs)
            got_o, got_h = fla_fn(*inputs)
        if not bool(torch.isfinite(got_o).all() and torch.isfinite(got_h).all()):
            raise AssertionError("FLA produced non-finite output/state")
        row["output"] = assert_close(torch, "output", ref_o, got_o, 2e-2, 2e-2)
        row["state"] = assert_close(torch, "state", ref_h, got_h, 2e-2, 2e-2)
        row["passed"] = True
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"
        row["traceback"] = traceback.format_exc()
        print(f"    FAIL: {row['error']}", flush=True)
    finally:
        gc.collect(); torch.cuda.empty_cache()
    return row


def gradient_correctness(
    torch: Any,
    inputs: tuple[Any, ...],
    recurrent_fn: Callable[..., tuple[Any, Any]],
    fla_fn: Callable[..., tuple[Any, Any]],
) -> dict[str, Any]:
    print("[correctness:bwd] normal:fp16", flush=True)
    print("    phase 1/2: recurrent-reference backward", flush=True)
    row: dict[str, Any] = {"label": "fla_vs_recurrent:normal:fp16:backward", "passed": False}
    try:
        ref = clone_leaves(inputs)
        got = clone_leaves(inputs)
        ref_o, ref_h = recurrent_fn(*ref)
        gen = torch.Generator(device="cuda").manual_seed(991)
        do = torch.randn(ref_o.shape, device="cuda", dtype=torch.float32, generator=gen).to(ref_o.dtype)
        dh = torch.randn(ref_h.shape, device="cuda", dtype=torch.float32, generator=gen)
        ref_loss = (ref_o.float() * do.float()).sum() + (ref_h * dh).sum()
        ref_grads = torch.autograd.grad(ref_loss, ref)

        print("    phase 2/2: FLA backward (first call may trigger extensive Triton compilation/autotuning)", flush=True)
        got_o, got_h = fla_fn(*got)
        got_loss = (got_o.float() * do.float()).sum() + (got_h * dh).sum()
        got_grads = torch.autograd.grad(got_loss, got)
        names = ("q", "k", "v", "g", "erase", "write", "h0")
        row["gradients"] = {}
        for name, rg, gg in zip(names, ref_grads, got_grads, strict=True):
            row["gradients"][name] = assert_close(torch, f"d{name}", rg, gg, 5e-2, 5e-2)
        row["passed"] = True
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"
        row["traceback"] = traceback.format_exc()
        print(f"    FAIL: {row['error']}", flush=True)
    finally:
        gc.collect(); torch.cuda.empty_cache()
    return row


def benchmark_forward(torch: Any, label: str, fn: Callable[..., tuple[Any, Any]], inputs: tuple[Any, ...], iters: int) -> dict[str, Any]:
    batch, seq = inputs[0].shape[:2]
    print(f"[benchmark:fwd] {label} B={batch} T={seq}", flush=True)
    row: dict[str, Any] = {"label": label, "mode": "forward", "passed": False}
    try:
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            fn(*inputs)
        torch.cuda.synchronize()
        times = []
        with torch.no_grad():
            for _ in range(iters):
                t0 = time.perf_counter(); out, state = fn(*inputs); torch.cuda.synchronize(); times.append(time.perf_counter() - t0)
        med = statistics.median(times)
        row.update({
            "passed": True,
            "median_seconds": med,
            "backend_tokens_per_second": batch * seq / med,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        })
        print(f"    median={med:.4f}s backend_tok/s={row['backend_tokens_per_second']:.1f} peak={row['peak_allocated_gib']:.2f}GiB", flush=True)
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"; row["traceback"] = traceback.format_exc(); print(f"    FAIL: {row['error']}", flush=True)
    finally:
        gc.collect(); torch.cuda.empty_cache()
    return row


def benchmark_train(torch: Any, label: str, fn: Callable[..., tuple[Any, Any]], inputs: tuple[Any, ...], iters: int) -> dict[str, Any]:
    batch, seq = inputs[0].shape[:2]
    print(f"[benchmark:fwd+bwd] {label} B={batch} T={seq}", flush=True)
    row: dict[str, Any] = {"label": label, "mode": "forward_backward", "passed": False}
    try:
        def one() -> tuple[float, float]:
            leaves = clone_leaves(inputs)
            torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0 = time.perf_counter()
            out, state = fn(*leaves)
            (out.float().square().mean() + state.float().square().mean()).backward()
            torch.cuda.synchronize(); dt = time.perf_counter() - t0
            return dt, torch.cuda.max_memory_allocated() / 1024**3
        print("    warmup: may compile/autotune backward kernels", flush=True)
        one()
        vals = [one() for _ in range(iters)]
        med = statistics.median(x[0] for x in vals)
        row.update({"passed": True, "median_seconds": med, "backend_tokens_per_second": batch * seq / med, "peak_allocated_gib": max(x[1] for x in vals)})
        print(f"    median={med:.4f}s backend_tok/s={row['backend_tokens_per_second']:.1f} peak={row['peak_allocated_gib']:.2f}GiB", flush=True)
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"; row["traceback"] = traceback.format_exc(); print(f"    FAIL: {row['error']}", flush=True)
    finally:
        gc.collect(); torch.cuda.empty_cache()
    return row


def ratio(a: dict[str, Any] | None, b: dict[str, Any] | None) -> float | None:
    if not a or not b or not a.get("passed") or not b.get("passed"):
        return None
    return float(a["backend_tokens_per_second"]) / float(b["backend_tokens_per_second"])


def main() -> int:
    args = parse_args()
    if min(args.benchmark_batch, args.benchmark_seq, args.forward_iters, args.train_iters) <= 0:
        raise SystemExit("benchmark dimensions/iterations must be positive")

    if args.with_backward:
        # Make completed autotune choices visible. This does not disable tuning.
        os.environ.setdefault("TRITON_PRINT_AUTOTUNING", "1")

    import torch
    from torch.nn import functional as F
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required")

    print("=" * 78)
    print("Small-LLM FLA GDN-2 GPU qualification probe")
    print("=" * 78)
    print(f"[env] torch={torch.__version__} cuda={torch.version.cuda}")
    print(f"[env] gpu={torch.cuda.get_device_name(0)}")
    print(f"[env] compute_capability={torch.cuda.get_device_capability(0)}")
    packages = ensure_fla(not args.no_install)

    import triton
    from fla.ops.gdn2 import chunk_gdn2
    from model.gdn2 import gdn2_recurrent_reference
    from model.gdn2_stable import AdaptiveChunkwiseGDN2Backend

    print(f"[env] triton={triton.__version__} fla-core={packages['fla_core']} flash-linear-attention={packages['flash_linear_attention']} vram={torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GiB")
    if not args.with_backward:
        print("[mode] forward-first qualification; backward is intentionally skipped. Use --with-backward for the expensive training-kernel qualification.", flush=True)

    adaptive = AdaptiveChunkwiseGDN2Backend(chunk_size=CHUNK_SIZE)

    def recurrent_fn(q: Any, k: Any, v: Any, g: Any, b: Any, w: Any, h0: Any) -> tuple[Any, Any]:
        return gdn2_recurrent_reference(q, k, v, g, b, w, h0)

    def adaptive_fn(q: Any, k: Any, v: Any, g: Any, b: Any, w: Any, h0: Any) -> tuple[Any, Any]:
        return adaptive(q, k, v, g, b, w, h0)

    def fla_fn(q: Any, k: Any, v: Any, g: Any, b: Any, w: Any, h0: Any) -> tuple[Any, Any]:
        out, state = chunk_gdn2(
            q=q, k=k, v=v, g=g, b=b, w=w,
            scale=1.0 / math.sqrt(KEY_DIM), initial_state=h0,
            output_final_state=True, use_qk_l2norm_in_kernel=False,
            use_gate_in_kernel=False, disable_recompute=True,
        )
        if state is None:
            raise RuntimeError("FLA did not return final state")
        return out, state

    report: dict[str, Any] = {
        "environment": {"torch": torch.__version__, "triton": triton.__version__, "gpu": torch.cuda.get_device_name(0), **packages},
        "with_backward": args.with_backward,
        "correctness": [],
        "benchmarks": [],
    }

    for decay, seed in (("normal", 100), ("stress_-6", 200), ("extreme_-10", 300)):
        report["correctness"].append(forward_correctness(
            torch, f"fla_vs_recurrent:{decay}:fp16",
            make_inputs(torch, F, batch=1, seq=64, decay=decay, seed=seed),
            recurrent_fn, fla_fn,
        ))

    if args.with_backward:
        report["correctness"].append(gradient_correctness(
            torch,
            make_inputs(torch, F, batch=1, seq=64, decay="normal", seed=350),
            recurrent_fn, fla_fn,
        ))

    for decay, seed in (("normal", 400), ("stress_-6", 500)):
        template = make_inputs(torch, F, batch=args.benchmark_batch, seq=args.benchmark_seq, decay=decay, seed=seed)
        for name, fn in (("adaptive", adaptive_fn), ("fla", fla_fn)):
            report["benchmarks"].append(benchmark_forward(torch, f"{name}:{decay}", fn, template, args.forward_iters))
        if args.with_backward:
            for name, fn in (("adaptive", adaptive_fn), ("fla", fla_fn)):
                report["benchmarks"].append(benchmark_train(torch, f"{name}:{decay}", fn, template, args.train_iters))
        del template; gc.collect(); torch.cuda.empty_cache()

    fwd = {r["label"]: r for r in report["benchmarks"] if r["mode"] == "forward"}
    train = {r["label"]: r for r in report["benchmarks"] if r["mode"] == "forward_backward"}
    summary = {
        "forward_correctness": all(r.get("passed") for r in report["correctness"] if ":backward" not in r["label"]),
        "backward_correctness": next((r.get("passed") for r in report["correctness"] if ":backward" in r["label"]), None),
        "forward_fla_speedup_over_adaptive_normal": ratio(fwd.get("fla:normal"), fwd.get("adaptive:normal")),
        "forward_fla_speedup_over_adaptive_stress": ratio(fwd.get("fla:stress_-6"), fwd.get("adaptive:stress_-6")),
        "forward_adaptive_stress_retention": ratio(fwd.get("adaptive:stress_-6"), fwd.get("adaptive:normal")),
        "forward_fla_stress_retention": ratio(fwd.get("fla:stress_-6"), fwd.get("fla:normal")),
        "train_fla_speedup_over_adaptive_stress": ratio(train.get("fla:stress_-6"), train.get("adaptive:stress_-6")),
    }
    report["summary"] = summary
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print("=" * 78); print("SUMMARY"); print("=" * 78)
    for k, v in summary.items():
        if isinstance(v, float): print(f"{k}: {v:.3f}x")
        else: print(f"{k}: {v}")
    if not args.with_backward:
        print("verdict: FORWARD QUALIFIED only. This is enough to test the strong-decay runtime hypothesis, but not enough to authorize training integration.")
        print("next training gate: rerun the same entry point with --with-backward (expect explicit phase output before Triton backward autotuning).")
    elif summary["forward_correctness"] and summary["backward_correctness"]:
        print("verdict: FORWARD + BACKWARD correctness qualified; review benchmark ratios before integration.")
    else:
        print("verdict: NOT QUALIFIED for training integration.")
    print(f"JSON report: {args.report}")
    return 0 if summary["forward_correctness"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
