#!/usr/bin/env python3
"""Test whether released FLA v0.5.2 fixes GDN-2 AMP backward decay failures.

This is the same full Small-LLM GDN-2 layer qualification geometry as
``run_gdn2_fla_amp_decay_sweep.py``, but it *forces* ``fla-core==0.5.2`` even
when a prior Kaggle cell installed v0.5.1.

It does not load a training checkpoint and does not start the trainer.

Run on the Tesla T4:
    python kaggle/run_gdn2_fla_052_amp_decay_sweep.py
"""
from __future__ import annotations

import gc
import importlib
import importlib.metadata
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FLA_VERSION = "0.5.2"
DECAYS = (-0.25, -0.5, -0.75, -1.0, -1.25, -1.5, -2.0, -3.0, -4.0, -5.0, -6.0)


def installed_fla_version() -> str | None:
    try:
        return importlib.metadata.version("fla-core")
    except importlib.metadata.PackageNotFoundError:
        return None


def ensure_exact_fla() -> None:
    current = installed_fla_version()
    if current == FLA_VERSION:
        print(f"[setup] fla-core=={FLA_VERSION} already installed", flush=True)
    else:
        print(
            f"[setup] installing exact fla-core=={FLA_VERSION} --no-deps "
            f"(current={current!r})",
            flush=True,
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--quiet",
                "--upgrade",
                "--force-reinstall",
                "--no-deps",
                f"fla-core=={FLA_VERSION}",
            ],
            check=True,
        )
    for name in tuple(sys.modules):
        if name == "fla" or name.startswith("fla."):
            sys.modules.pop(name, None)
    importlib.invalidate_caches()
    actual = installed_fla_version()
    if actual != FLA_VERSION:
        raise SystemExit(f"expected fla-core=={FLA_VERSION}, found {actual!r}")
    from fla.ops.gdn2 import chunk_gdn2  # noqa: F401


def force_constant_log_decay(torch, layer, value: float) -> None:
    magnitude = abs(float(value))
    with torch.no_grad():
        layer.decay_proj[0].weight.zero_()
        layer.decay_proj[1].weight.zero_()
        layer.A_log.fill_(math.log(magnitude))
        layer.dt_bias.fill_(math.log(math.expm1(1.0)))


def finite_count(torch, tensor) -> int:
    return int((~torch.isfinite(tensor.detach())).sum().item())


def run_case(torch, config, value: float) -> dict[str, object]:
    from model.gdn2_stable import AdaptiveChunkwiseGDN2Backend, StableGatedDeltaNet2

    reference = StableGatedDeltaNet2(
        config,
        backend=AdaptiveChunkwiseGDN2Backend(chunk_size=32),
    ).cuda()
    candidate = StableGatedDeltaNet2(config).cuda()
    candidate.load_state_dict(reference.state_dict(), strict=True)
    force_constant_log_decay(torch, reference, value)
    candidate.load_state_dict(reference.state_dict(), strict=True)

    generator = torch.Generator(device="cuda").manual_seed(12345)
    source = torch.randn(1, 64, config.d_model, device="cuda", dtype=torch.float32, generator=generator)
    upstream = torch.randn(source.shape, device="cuda", dtype=torch.float32, generator=generator)
    ref_x = source.detach().clone().requires_grad_(True)
    fla_x = source.detach().clone().requires_grad_(True)
    names = [name for name, _ in reference.named_parameters()]
    ref_parameters = [parameter for _, parameter in reference.named_parameters()]
    fla_parameters = [parameter for _, parameter in candidate.named_parameters()]

    row: dict[str, object] = {"log_decay": value, "passed": False, "bad_gradients": []}
    try:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            ref_output = reference(ref_x)
            fla_output = candidate(fla_x)
            ref_loss = (ref_output.float() * upstream).sum()
            fla_loss = (fla_output.float() * upstream).sum()

        ref_grads = torch.autograd.grad(ref_loss, [ref_x, *ref_parameters], allow_unused=False)
        fla_grads = torch.autograd.grad(fla_loss, [fla_x, *fla_parameters], allow_unused=False)

        output_ok = bool(torch.isfinite(fla_output).all())
        try:
            torch.testing.assert_close(fla_output, ref_output, atol=3e-2, rtol=3e-2)
        except AssertionError:
            output_ok = False

        bad: list[str] = []
        worst_max_abs = 0.0
        for name, ref_grad, fla_grad in zip(["x", *names], ref_grads, fla_grads, strict=True):
            ref_bad = finite_count(torch, ref_grad)
            fla_bad = finite_count(torch, fla_grad)
            if ref_bad or fla_bad:
                bad.append(f"{name}(ref_nonfinite={ref_bad},fla_nonfinite={fla_bad})")
                continue
            diff = (ref_grad.detach().float() - fla_grad.detach().float()).abs()
            worst_max_abs = max(worst_max_abs, float(diff.max().item()))
            try:
                torch.testing.assert_close(fla_grad, ref_grad, atol=1e-1, rtol=1e-1)
            except AssertionError:
                bad.append(f"{name}(mismatch,max_abs={float(diff.max().item()):.3e})")

        row.update(
            {
                "passed": output_ok and not bad,
                "output_ok": output_ok,
                "bad_gradients": bad,
                "worst_finite_grad_max_abs": worst_max_abs,
            }
        )
        status = "PASS" if row["passed"] else "FAIL"
        details = "" if not bad else " | " + "; ".join(bad[:5])
        print(
            f"g={value:>5.2f} span64={abs(value)*64:>6.1f}  {status}  "
            f"output={'ok' if output_ok else 'BAD'} worst_max_abs={worst_max_abs:.3e}{details}",
            flush=True,
        )
        return row
    finally:
        del reference, candidate, source, upstream, ref_x, fla_x
        gc.collect()
        torch.cuda.empty_cache()


def main() -> int:
    import torch
    from model.config import ModelConfig

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required")
    ensure_exact_fla()

    print("=" * 78)
    print("Small-LLM FLA v0.5.2 GDN-2 trainer-AMP decay sweep")
    print("=" * 78)
    print(
        f"[env] torch={torch.__version__} cuda={torch.version.cuda} "
        f"gpu={torch.cuda.get_device_name(0)} fla-core={installed_fla_version()}"
    )
    print("[contract] fp32 parameters + fp16 autocast; saved chunk=32; FLA runtime chunk=64")
    print("[purpose] determine whether v0.5.2 fixes the v0.5.1 decay-dependent backward failure")

    config = ModelConfig.smoke(max_seq_len=64, gdn_chunk_size=32)
    rows = [run_case(torch, config, value) for value in DECAYS]
    passing = [float(row["log_decay"]) for row in rows if row["passed"]]
    failing = [float(row["log_decay"]) for row in rows if not row["passed"]]

    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"fla_core_version: {installed_fla_version()}")
    print("passing:", passing)
    print("failing:", failing)
    if failing:
        first = failing[0]
        print(
            f"first failing tested point: g={first} "
            f"(64-token cumulative magnitude {abs(first)*64:.1f})"
        )
        print("VERDICT: v0.5.2 still has a tested trainer-AMP backward failure; do not resume training yet.")
        return 1
    print("No failure found in the tested range through g=-6.")
    print("VERDICT: synthetic v0.5.2 sweep passes; real step-4000 forward/backward parity is the next gate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
