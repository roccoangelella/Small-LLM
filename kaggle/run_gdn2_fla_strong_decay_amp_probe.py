#!/usr/bin/env python3
"""Focused CUDA gate for FLA GDN-2 strong-decay gradients under trainer AMP.

This probe exists because the first integrated layer probe used an all-FP16
model and therefore missed the real trainer dtype contract.  The revised AMP
probe later showed that normal-decay gradients are correct but strong-decay
backward produced NaNs when FLA recomputed forward intermediates during
backward.

This focused test changes one thing only: FLA retains the forward intermediates
(`disable_recompute=True`) and reuses them in backward.  It keeps FP32 model
parameters and executes the layer under CUDA FP16 autocast, matching training.

Run:
    python kaggle/run_gdn2_fla_strong_decay_amp_probe.py
"""

from __future__ import annotations

import gc
import importlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FLA_VERSION = "0.5.1"


def ensure_fla() -> None:
    try:
        from fla.ops.gdn2 import chunk_gdn2  # noqa: F401
        return
    except Exception:
        print(f"[setup] installing fla-core=={FLA_VERSION} --no-deps", flush=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", f"fla-core=={FLA_VERSION}"],
        check=True,
    )
    for name in tuple(sys.modules):
        if name == "fla" or name.startswith("fla."):
            sys.modules.pop(name, None)
    importlib.invalidate_caches()
    from fla.ops.gdn2 import chunk_gdn2  # noqa: F401


def force_log_decay_minus_six(torch, layer) -> None:
    import math

    with torch.no_grad():
        layer.decay_proj[0].weight.zero_()
        layer.decay_proj[1].weight.zero_()
        layer.A_log.fill_(math.log(6.0))
        layer.dt_bias.fill_(math.log(math.expm1(1.0)))


def finite_summary(torch, tensor) -> tuple[bool, int]:
    finite = torch.isfinite(tensor.detach())
    return bool(finite.all()), int((~finite).sum().item())


def main() -> int:
    import torch
    from model.config import ModelConfig
    from model.gdn2_stable import AdaptiveChunkwiseGDN2Backend, StableGatedDeltaNet2

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required")
    ensure_fla()

    print("=" * 78)
    print("Small-LLM FLA GDN-2 strong-decay AMP retained-intermediate probe")
    print("=" * 78)
    print(f"[env] torch={torch.__version__} cuda={torch.version.cuda} gpu={torch.cuda.get_device_name(0)}")
    print("[contract] fp32 parameters + fp16 autocast; saved chunk=32; FLA runtime chunk=64")
    print("[candidate] disable_recompute=True")

    config = ModelConfig.smoke(max_seq_len=64, gdn_chunk_size=32)
    reference = StableGatedDeltaNet2(
        config,
        backend=AdaptiveChunkwiseGDN2Backend(chunk_size=32),
    ).cuda()
    candidate = StableGatedDeltaNet2(config).cuda()
    candidate.load_state_dict(reference.state_dict(), strict=True)
    candidate.backend.fla_backend.disable_recompute = True

    force_log_decay_minus_six(torch, reference)
    candidate.load_state_dict(reference.state_dict(), strict=True)

    generator = torch.Generator(device="cuda").manual_seed(220)
    source = torch.randn(1, 64, config.d_model, device="cuda", dtype=torch.float32, generator=generator)
    upstream = torch.randn(source.shape, device="cuda", dtype=torch.float32, generator=generator)
    ref_x = source.detach().clone().requires_grad_(True)
    fla_x = source.detach().clone().requires_grad_(True)

    names = [name for name, _ in reference.named_parameters()]
    ref_parameters = [parameter for _, parameter in reference.named_parameters()]
    fla_parameters = [parameter for _, parameter in candidate.named_parameters()]

    try:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            ref_output = reference(ref_x)
            fla_output = candidate(fla_x)
            ref_loss = (ref_output.float() * upstream).sum()
            fla_loss = (fla_output.float() * upstream).sum()

        ref_grads = torch.autograd.grad(ref_loss, [ref_x, *ref_parameters], allow_unused=False)
        fla_grads = torch.autograd.grad(fla_loss, [fla_x, *fla_parameters], allow_unused=False)

        torch.testing.assert_close(fla_output, ref_output, atol=3e-2, rtol=3e-2)
        print("[output] PASS", flush=True)

        passed = True
        for name, ref_grad, fla_grad in zip(["x", *names], ref_grads, fla_grads, strict=True):
            ref_finite, ref_bad = finite_summary(torch, ref_grad)
            fla_finite, fla_bad = finite_summary(torch, fla_grad)
            if not ref_finite or not fla_finite:
                passed = False
                print(
                    f"[grad] {name}: NONFINITE reference_bad={ref_bad} fla_bad={fla_bad}",
                    flush=True,
                )
                continue
            diff = (ref_grad.detach().float() - fla_grad.detach().float()).abs()
            try:
                torch.testing.assert_close(fla_grad, ref_grad, atol=1e-1, rtol=1e-1)
                print(
                    f"[grad] {name}: PASS max_abs={float(diff.max().item()):.3e} "
                    f"mean_abs={float(diff.mean().item()):.3e}",
                    flush=True,
                )
            except AssertionError:
                passed = False
                print(
                    f"[grad] {name}: MISMATCH max_abs={float(diff.max().item()):.3e} "
                    f"mean_abs={float(diff.mean().item()):.3e}",
                    flush=True,
                )

        print("=" * 78)
        if passed:
            print("VERDICT: PASS — retained-intermediate FLA strong-decay AMP gradients are finite and match the adaptive reference.")
            return 0
        print("VERDICT: FAIL — retained-intermediate FLA is still not qualified for resumed training.")
        return 1
    finally:
        del reference, candidate, source, upstream, ref_x, fla_x
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
