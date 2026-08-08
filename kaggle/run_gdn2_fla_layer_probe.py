#!/usr/bin/env python3
"""One-click Kaggle qualification for the integrated Small-LLM FLA GDN-2 backend.

Default:
    python kaggle/run_gdn2_fla_layer_probe.py

Optional checkpoint compatibility/behavior check:
    python kaggle/run_gdn2_fla_layer_probe.py --checkpoint /path/to/checkpoint.pt
"""

from __future__ import annotations

import argparse
import gc
import importlib
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FLA_VERSION = "0.5.1"
DEFAULT_REPORT = (
    Path("/kaggle/working/gdn2_fla_layer_probe.json")
    if Path("/kaggle/working").is_dir()
    else ROOT / "gdn2_fla_layer_probe.json"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qualify the integrated FLA GDN-2 layer on CUDA.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint-seq", type=int, default=64)
    parser.add_argument("--no-install", action="store_true")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _ensure_fla(*, allow_install: bool) -> None:
    try:
        from fla.ops.gdn2 import chunk_gdn2  # noqa: F401
        return
    except Exception as first_error:
        if not allow_install:
            raise RuntimeError("fla.ops.gdn2 is unavailable and --no-install was requested") from first_error
    print(f"[setup] installing fla-core=={FLA_VERSION} --no-deps (Torch/Triton unchanged)")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", f"fla-core=={FLA_VERSION}"],
        check=True,
    )
    for name in tuple(sys.modules):
        if name == "fla" or name.startswith("fla."):
            sys.modules.pop(name, None)
    importlib.invalidate_caches()
    from fla.ops.gdn2 import chunk_gdn2  # noqa: F401


def _metrics(torch: Any, reference: Any, candidate: Any) -> dict[str, float]:
    diff = (reference.detach().float() - candidate.detach().float()).abs()
    return {"max_abs": float(diff.max().item()), "mean_abs": float(diff.mean().item())}


def _assert_close(
    torch: Any,
    label: str,
    reference: Any,
    candidate: Any,
    *,
    atol: float,
    rtol: float,
) -> dict[str, float]:
    result = _metrics(torch, reference, candidate)
    torch.testing.assert_close(candidate, reference, atol=atol, rtol=rtol)
    print(
        f"    {label:<34} PASS  max_abs={result['max_abs']:.3e} "
        f"mean_abs={result['mean_abs']:.3e}"
    )
    return result


def _parameter_grads(torch: Any, layer: Any, x: Any, upstream: Any) -> tuple[Any, dict[str, Any]]:
    output = layer(x)
    loss = (output.float() * upstream.float()).sum()
    names = [name for name, _ in layer.named_parameters()]
    parameters = [parameter for _, parameter in layer.named_parameters()]
    grads = torch.autograd.grad(loss, [x, *parameters], allow_unused=False)
    return output, {"x": grads[0], **dict(zip(names, grads[1:], strict=True))}


def _force_log_decay_minus_six(torch: Any, layer: Any) -> None:
    """Make the layer's decay projection produce log_decay=-6 for every channel."""
    with torch.no_grad():
        layer.decay_proj[0].weight.zero_()
        layer.decay_proj[1].weight.zero_()
        layer.A_log.fill_(math.log(6.0))
        layer.dt_bias.fill_(math.log(math.expm1(1.0)))


def _layer_case(torch: Any, *, strong_decay: bool) -> dict[str, Any]:
    from model.config import ModelConfig
    from model.gdn2_stable import AdaptiveChunkwiseGDN2Backend, StableGatedDeltaNet2

    label = "strong_decay_-6" if strong_decay else "normal_decay"
    print(f"[layer] {label}  saved_config_chunk=32  FLA_runtime_chunk=64")
    # This deliberately matches the existing 500M checkpoint configuration:
    # adaptive/reference execution sees chunk 32, while CUDA candidate execution
    # uses FLA's fixed 64-token kernel without changing the model config.
    config = ModelConfig.smoke(max_seq_len=64, gdn_chunk_size=32)
    reference = StableGatedDeltaNet2(
        config,
        backend=AdaptiveChunkwiseGDN2Backend(chunk_size=32),
    ).cuda().half()
    candidate = StableGatedDeltaNet2(config).cuda().half()
    candidate.load_state_dict(reference.state_dict(), strict=True)

    if tuple(reference.state_dict()) != tuple(candidate.state_dict()):
        raise AssertionError("FLA integration changed GDN-2 checkpoint parameter keys")
    if strong_decay:
        _force_log_decay_minus_six(torch, reference)
        candidate.load_state_dict(reference.state_dict(), strict=True)

    generator = torch.Generator(device="cuda")
    generator.manual_seed(220 if strong_decay else 120)
    source = torch.randn(
        1,
        64,
        config.d_model,
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    )
    upstream = torch.randn(source.shape, device="cuda", dtype=torch.float16, generator=generator)
    ref_x = source.detach().clone().requires_grad_(True)
    fla_x = source.detach().clone().requires_grad_(True)

    result: dict[str, Any] = {
        "label": label,
        "saved_config_chunk_size": 32,
        "fla_runtime_chunk_size": 64,
        "passed": False,
    }
    try:
        ref_output, ref_grads = _parameter_grads(torch, reference, ref_x, upstream)
        fla_output, fla_grads = _parameter_grads(torch, candidate, fla_x, upstream)
        errors: dict[str, Any] = {
            "output": _assert_close(
                torch, "layer output", ref_output, fla_output, atol=3e-2, rtol=3e-2
            ),
            "gradients": {},
        }
        for name in ref_grads:
            errors["gradients"][name] = _assert_close(
                torch,
                f"grad {name}",
                ref_grads[name],
                fla_grads[name],
                atol=1e-1,
                rtol=1e-1,
            )
        result.update({"passed": True, "errors": errors})
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        result["traceback"] = traceback.format_exc()
        print(f"    FAIL: {result['error']}")
    finally:
        del reference, candidate, source, ref_x, fla_x, upstream
        gc.collect()
        torch.cuda.empty_cache()
    return result


def _replace_with_adaptive(model: Any) -> None:
    from model.gdn2_stable import AdaptiveChunkwiseGDN2Backend, StableGatedDeltaNet2

    for kind, block in zip(model.layer_kinds, model.blocks, strict=True):
        if kind in {"gdn", "gdn-2"}:
            if not isinstance(block.mixer, StableGatedDeltaNet2):
                raise TypeError("unexpected GDN mixer type")
            block.mixer.backend = AdaptiveChunkwiseGDN2Backend(model.config.gdn_chunk_size)


def _checkpoint_case(torch: Any, checkpoint: Path, sequence: int) -> dict[str, Any]:
    from model.config import ModelConfig
    from model.model import SmallLLM

    print(f"[checkpoint] {checkpoint}")
    result: dict[str, Any] = {"path": str(checkpoint), "passed": False}
    try:
        raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(raw, dict):
            raise ValueError("checkpoint root must be a mapping")
        model_state = raw.get("model")
        model_config = raw.get("model_config")
        if not isinstance(model_state, dict) or not isinstance(model_config, dict):
            raise ValueError("expected trainer checkpoint fields `model` and `model_config`")
        config = ModelConfig(**model_config)

        candidate = SmallLLM(config)
        reference = SmallLLM(config)
        candidate.load_state_dict(model_state, strict=True)
        reference.load_state_dict(model_state, strict=True)
        _replace_with_adaptive(reference)
        candidate = candidate.cuda().half().eval()
        reference = reference.cuda().half().eval()

        actual_sequence = min(sequence, config.max_seq_len)
        generator = torch.Generator(device="cuda")
        generator.manual_seed(9917)
        input_ids = torch.randint(
            0,
            config.semantic_vocab_size,
            (1, actual_sequence),
            device="cuda",
            generator=generator,
        )
        with torch.inference_mode():
            reference_logits = reference(input_ids)
            candidate_logits = candidate(input_ids)
        errors = _assert_close(
            torch,
            "checkpoint full-model logits",
            reference_logits,
            candidate_logits,
            atol=1e-1,
            rtol=5e-2,
        )
        result.update(
            {
                "passed": True,
                "global_step": raw.get("global_step"),
                "saved_config_chunk_size": config.gdn_chunk_size,
                "fla_runtime_chunk_size": 64,
                "sequence": actual_sequence,
                "logit_errors": errors,
            }
        )
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        result["traceback"] = traceback.format_exc()
        print(f"    FAIL: {result['error']}")
    finally:
        gc.collect()
        torch.cuda.empty_cache()
    return result


def main() -> int:
    args = _parse_args()
    if args.checkpoint_seq <= 0:
        raise SystemExit("--checkpoint-seq must be positive")

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required")
    print("=" * 78)
    print("Small-LLM integrated FLA GDN-2 layer qualification")
    print("=" * 78)
    print(f"[env] torch={torch.__version__} cuda={torch.version.cuda}")
    print(f"[env] gpu={torch.cuda.get_device_name(torch.cuda.current_device())}")
    _ensure_fla(allow_install=not args.no_install)
    try:
        import triton
        triton_version = getattr(triton, "__version__", "unknown")
    except Exception:
        triton_version = None
    print(
        f"[env] triton={triton_version} fla-core={_package_version('fla-core')} "
        f"flash-linear-attention={_package_version('flash-linear-attention')}"
    )

    report: dict[str, Any] = {
        "probe": "gdn2_fla_integrated_layer",
        "environment": {
            "torch": torch.__version__,
            "triton": triton_version,
            "fla_core": _package_version("fla-core"),
            "gpu": torch.cuda.get_device_name(torch.cuda.current_device()),
        },
        "layers": [],
        "checkpoint": None,
    }
    report["layers"].append(_layer_case(torch, strong_decay=False))
    report["layers"].append(_layer_case(torch, strong_decay=True))
    if args.checkpoint is not None:
        report["checkpoint"] = _checkpoint_case(torch, args.checkpoint, args.checkpoint_seq)

    layer_pass = all(row.get("passed", False) for row in report["layers"])
    checkpoint_pass = report["checkpoint"] is None or report["checkpoint"].get("passed", False)
    passed = layer_pass and checkpoint_pass
    report["summary"] = {
        "layer_forward_backward_parity": layer_pass,
        "checkpoint_parity": None if report["checkpoint"] is None else checkpoint_pass,
        "verdict": (
            "INTEGRATION QUALIFIED for checkpoint evaluation/resume geometry; fresh-training authorization remains separate."
            if passed
            else "NOT QUALIFIED; inspect failed layer/checkpoint parity before using FLA integration."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"layer_forward_backward_parity: {layer_pass}")
    print(f"checkpoint_parity: {report['summary']['checkpoint_parity']}")
    print(report["summary"]["verdict"])
    print(f"JSON report: {args.report}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
