#!/usr/bin/env python3
"""One-shot Kaggle qualification for FP32 FLA GDN-2 training execution.

This diagnostic answers one bounded question: does evaluating the complete FLA
GDN-2 chunk kernel in FP32 remove the decay-dependent backward failures seen
under the trainer's FP16 autocast contract?

It deliberately does NOT load or modify the step-4000 checkpoint and does NOT
start training. It first reproduces the released mixed-precision FLA v0.5.2
sweep, then repeats the exact same full-layer forward/backward comparison with
FLA forced to FP32 internally. Every candidate gradient is checked for
finiteness and numerical parity against the adaptive PyTorch reference.

Kaggle entry point:
    python kaggle/run_gdn2_fla_fp32_qualification.py

Copy everything between COPY_PASTE_REPORT_BEGIN and COPY_PASTE_REPORT_END back
into the project chat after the run.
"""
from __future__ import annotations

import gc
import importlib
import importlib.metadata
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FLA_VERSION = "0.5.2"
DECAYS = (-0.25, -0.5, -0.75, -1.0, -1.25, -1.5, -2.0, -3.0, -4.0, -5.0, -6.0)
LAYER_INIT_SEED = 20260808
INPUT_SEED = 12345
OUTPUT_ATOL = 3e-2
OUTPUT_RTOL = 3e-2
GRAD_ATOL = 1e-1
GRAD_RTOL = 1e-1
REPORT_PATH = Path("/kaggle/working/gdn2_fla_fp32_qualification.json")


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
    """Choose layer parameters that make g exactly ``value`` everywhere."""

    magnitude = abs(float(value))
    with torch.no_grad():
        layer.decay_proj[0].weight.zero_()
        layer.decay_proj[1].weight.zero_()
        layer.A_log.fill_(math.log(magnitude))
        # softplus(dt_bias)=1 => g=-exp(A_log)*1=-magnitude.
        layer.dt_bias.fill_(math.log(math.expm1(1.0)))


def nonfinite_count(torch, tensor) -> int:
    return int((~torch.isfinite(tensor.detach())).sum().item())


def make_candidate_backend(config, *, force_fp32: bool):
    from model.gdn2_fla import FLAPreferredGDN2Backend
    from model.gdn2_stable import AdaptiveChunkwiseGDN2Backend

    return FLAPreferredGDN2Backend(
        chunk_size=config.gdn_chunk_size,
        fallback_backend=AdaptiveChunkwiseGDN2Backend(config.gdn_chunk_size),
        force_fp32=force_fp32,
    )


class _FP32AdaptiveReferenceBackend:
    """Run the existing adaptive recurrence with autocast disabled.

    The surrounding layer still executes under the trainer's FP16 autocast
    contract. Only the recurrence oracle is forced to the FP32 arithmetic its
    implementation intends to use. Without this wrapper CUDA autocast can
    downcast eligible ``@``/linear-algebra operations inside the PyTorch
    reference even after its explicit ``.float()`` conversions, making the
    supposed correctness oracle itself produce non-finite gradients.
    """

    def __init__(self, chunk_size: int) -> None:
        from model.gdn2_stable import AdaptiveChunkwiseGDN2Backend

        self.chunk_size = chunk_size
        self.backend = AdaptiveChunkwiseGDN2Backend(chunk_size=chunk_size)

    def __call__(self, *args, **kwargs):
        import torch

        with torch.autocast(device_type="cuda", enabled=False):
            return self.backend(*args, **kwargs)


def run_case(torch, config, value: float, *, mode: str, force_fp32: bool) -> dict[str, object]:
    from model.gdn2_stable import StableGatedDeltaNet2

    # Recreate the exact same layer initialization for every decay and both
    # modes. The previous sweep seeded only source/upstream tensors, allowing
    # each row to test a different randomly initialized layer and making the
    # reported non-monotonic pattern non-reproducible across processes.
    torch.manual_seed(LAYER_INIT_SEED)
    torch.cuda.manual_seed_all(LAYER_INIT_SEED)
    reference = StableGatedDeltaNet2(
        config,
        backend=_FP32AdaptiveReferenceBackend(chunk_size=config.gdn_chunk_size),
    ).cuda()
    candidate = StableGatedDeltaNet2(
        config,
        backend=make_candidate_backend(config, force_fp32=force_fp32),
    ).cuda()
    candidate.load_state_dict(reference.state_dict(), strict=True)
    force_constant_log_decay(torch, reference, value)
    candidate.load_state_dict(reference.state_dict(), strict=True)

    generator = torch.Generator(device="cuda").manual_seed(INPUT_SEED)
    source = torch.randn(
        1,
        64,
        config.d_model,
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    upstream = torch.randn(
        source.shape,
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    ref_x = source.detach().clone().requires_grad_(True)
    fla_x = source.detach().clone().requires_grad_(True)
    names = [name for name, _ in reference.named_parameters()]
    ref_parameters = [parameter for _, parameter in reference.named_parameters()]
    fla_parameters = [parameter for _, parameter in candidate.named_parameters()]

    row: dict[str, object] = {
        "mode": mode,
        "force_fp32": force_fp32,
        "log_decay": value,
        "passed": False,
        "reference_valid": False,
        "candidate_failure": False,
        "output_ok": False,
        "reference_output_finite": False,
        "candidate_output_finite": False,
        "bad_gradients": [],
    }
    try:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            ref_output = reference(ref_x)
            fla_output = candidate(fla_x)
            ref_loss = (ref_output.float() * upstream).sum()
            fla_loss = (fla_output.float() * upstream).sum()

        ref_grads = torch.autograd.grad(
            ref_loss,
            [ref_x, *ref_parameters],
            allow_unused=False,
        )
        fla_grads = torch.autograd.grad(
            fla_loss,
            [fla_x, *fla_parameters],
            allow_unused=False,
        )
        torch.cuda.synchronize()

        reference_output_finite = bool(torch.isfinite(ref_output).all())
        candidate_output_finite = bool(torch.isfinite(fla_output).all())
        output_ok = reference_output_finite and candidate_output_finite
        if output_ok:
            try:
                torch.testing.assert_close(
                    fla_output,
                    ref_output,
                    atol=OUTPUT_ATOL,
                    rtol=OUTPUT_RTOL,
                )
            except AssertionError:
                output_ok = False

        bad: list[str] = []
        reference_bad: list[str] = []
        candidate_bad: list[str] = []
        worst_max_abs = 0.0
        for name, ref_grad, fla_grad in zip(
            ["x", *names],
            ref_grads,
            fla_grads,
            strict=True,
        ):
            ref_bad = nonfinite_count(torch, ref_grad)
            fla_bad = nonfinite_count(torch, fla_grad)
            if ref_bad:
                reference_bad.append(f"{name}(nonfinite={ref_bad})")
            if fla_bad:
                candidate_bad.append(f"{name}(nonfinite={fla_bad})")
            if ref_bad or fla_bad:
                bad.append(
                    f"{name}(ref_nonfinite={ref_bad},fla_nonfinite={fla_bad})"
                )
                continue

            diff = (ref_grad.detach().float() - fla_grad.detach().float()).abs()
            max_abs = float(diff.max().item())
            worst_max_abs = max(worst_max_abs, max_abs)
            try:
                torch.testing.assert_close(
                    fla_grad,
                    ref_grad,
                    atol=GRAD_ATOL,
                    rtol=GRAD_RTOL,
                )
            except AssertionError:
                candidate_bad.append(f"{name}(mismatch,max_abs={max_abs:.3e})")
                bad.append(f"{name}(mismatch,max_abs={max_abs:.3e})")

        reference_valid = reference_output_finite and not reference_bad
        candidate_failure = reference_valid and (
            not candidate_output_finite or not output_ok or bool(candidate_bad)
        )
        row.update(
            {
                "passed": reference_valid and output_ok and not candidate_bad,
                "reference_valid": reference_valid,
                "candidate_failure": candidate_failure,
                "output_ok": output_ok,
                "reference_output_finite": reference_output_finite,
                "candidate_output_finite": candidate_output_finite,
                "reference_bad_gradients": reference_bad,
                "candidate_bad_gradients": candidate_bad,
                "bad_gradients": bad,
                "worst_finite_grad_max_abs": worst_max_abs,
            }
        )
        status = "PASS" if row["passed"] else "FAIL"
        details = "" if not bad else " | " + "; ".join(bad[:6])
        print(
            f"[{mode}] g={value:>5.2f} span64={abs(value)*64:>6.1f} "
            f"{status} output={'ok' if output_ok else 'BAD'} "
            f"worst_max_abs={worst_max_abs:.3e}{details}",
            flush=True,
        )
        return row
    except Exception as error:
        row["exception"] = f"{type(error).__name__}: {error}"
        print(
            f"[{mode}] g={value:>5.2f} EXCEPTION "
            f"{type(error).__name__}: {error}",
            flush=True,
        )
        return row
    finally:
        del reference, candidate, source, upstream, ref_x, fla_x
        gc.collect()
        torch.cuda.empty_cache()


def summarize_mode(rows: list[dict[str, object]]) -> dict[str, object]:
    passing = [float(row["log_decay"]) for row in rows if bool(row["passed"])]
    invalid_reference = [
        float(row["log_decay"]) for row in rows if not bool(row.get("reference_valid"))
    ]
    candidate_failing = [
        float(row["log_decay"]) for row in rows if bool(row.get("candidate_failure"))
    ]
    failing = [float(row["log_decay"]) for row in rows if not bool(row["passed"])]
    first_failure = next((row for row in rows if bool(row.get("candidate_failure"))), None)
    compact_failure = None
    if first_failure is not None:
        compact_failure = {
            "log_decay": first_failure["log_decay"],
            "output_ok": first_failure.get("output_ok"),
            "candidate_bad_gradients": list(first_failure.get("candidate_bad_gradients", []))[:8],
            "exception": first_failure.get("exception"),
        }
    return {
        "passing": passing,
        "failing": failing,
        "candidate_failing": candidate_failing,
        "invalid_reference": invalid_reference,
        "all_passed": not failing,
        "all_references_valid": not invalid_reference,
        "reproduced_candidate_failure": bool(candidate_failing),
        "first_candidate_failure": compact_failure,
    }


def verdict(baseline: dict[str, object], fp32: dict[str, object]) -> str:
    if not bool(baseline["all_references_valid"]) or not bool(fp32["all_references_valid"]):
        return (
            "INVALID: the FP32 adaptive reference was non-finite in at least one row. "
            "Do not draw a kernel conclusion from this run."
        )
    if not bool(fp32["all_passed"]):
        return (
            "FAIL: full-FP32 FLA still fails at least one tested decay point against "
            "a finite FP32 adaptive reference. Do not resume training; localize the "
            "first failing backward intermediate."
        )
    if not bool(baseline["reproduced_candidate_failure"]):
        return (
            "INCONCLUSIVE-BUT-PASSING: full-FP32 FLA passed every tested point, "
            "but mixed-precision FLA did not reproduce a candidate-specific failure "
            "against the finite FP32 reference. Do not attribute a fix yet."
        )
    return (
        "PASS: mixed-precision FLA reproduced at least one candidate-specific failure "
        "against a finite FP32 adaptive reference while full-FP32 FLA passed every "
        "tested decay point. FP32 execution fixes the synthetic failure gate; next "
        "gate is real step-4000 forward/backward parity before any optimizer update "
        "or production resume."
    )


def main() -> int:
    ensure_exact_fla()

    import torch
    from model.config import ModelConfig

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required")

    print("=" * 88)
    print("Small-LLM FLA GDN-2 FP32 qualification")
    print("=" * 88)
    print(
        f"[env] python={sys.version.split()[0]} torch={torch.__version__} "
        f"cuda={torch.version.cuda} gpu={torch.cuda.get_device_name(0)} "
        f"fla-core={installed_fla_version()}"
    )
    print("[safety] synthetic full-layer diagnostic only; checkpoint and trainer are untouched")
    print(
        "[reference] adaptive PyTorch GDN-2 recurrence forced FP32 internally; "
        "surrounding layer keeps trainer autocast; saved/configured chunk=32"
    )
    print("[baseline] trainer autocast + current low-precision FLA compute tensors")
    print("[candidate] trainer autocast outside layer + complete FLA GDN-2 kernel forced FP32")
    print(f"[determinism] layer_init_seed={LAYER_INIT_SEED} input_seed={INPUT_SEED}")
    print(f"[decays] {list(DECAYS)}")

    config = ModelConfig.smoke(max_seq_len=64, gdn_chunk_size=32)

    print("\n" + "-" * 88)
    print("PHASE 1/2 — reproduce released v0.5.2 mixed-precision behavior")
    print("-" * 88)
    baseline_rows = [
        run_case(
            torch,
            config,
            value,
            mode="baseline_fp16",
            force_fp32=False,
        )
        for value in DECAYS
    ]

    print("\n" + "-" * 88)
    print("PHASE 2/2 — repeat with complete FLA GDN-2 execution forced FP32")
    print("-" * 88)
    fp32_rows = [
        run_case(
            torch,
            config,
            value,
            mode="candidate_fp32",
            force_fp32=True,
        )
        for value in DECAYS
    ]

    baseline_summary = summarize_mode(baseline_rows)
    fp32_summary = summarize_mode(fp32_rows)
    final_verdict = verdict(baseline_summary, fp32_summary)

    report = {
        "experiment": "gdn2_fla_fp32_qualification_v2",
        "fla_core_version": installed_fla_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "saved_gdn_chunk_size": 32,
        "fla_runtime_chunk_size": 64,
        "outer_trainer_contract": "FP32 parameters + CUDA FP16 autocast",
        "reference_recurrence_contract": "adaptive PyTorch recurrence with CUDA autocast disabled internally",
        "layer_init_seed": LAYER_INIT_SEED,
        "input_seed": INPUT_SEED,
        "baseline": baseline_summary,
        "candidate_fp32": fp32_summary,
        "verdict": final_verdict,
        "production_authorized": False,
        "checkpoint_touched": False,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("\n" + "=" * 88)
    print("COPY_PASTE_REPORT_BEGIN")
    print(json.dumps(report, indent=2, sort_keys=True))
    print("COPY_PASTE_REPORT_END")
    print("=" * 88)
    print(final_verdict)
    print(f"JSON report: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
