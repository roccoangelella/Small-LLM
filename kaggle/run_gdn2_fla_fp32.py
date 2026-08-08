#!/usr/bin/env python3
"""Self-provisioning notebook entry point for the FP32 FLA GDN-2 qualification.

This wrapper keeps the scientific diagnostic in
``run_gdn2_fla_fp32_qualification.py`` unchanged. Its only job is to make a
Kaggle/Colab GPU runtime match the already-qualified FLA test stack closely
enough to run the diagnostic reproducibly.

Known-qualified T4 stack from the prior Small-LLM investigation:
    torch 2.10.0 + CUDA 12.8
    Triton 3.6.0
    fla-core 0.5.2 (installed by the wrapped diagnostic)

If the notebook starts with a CPU-only PyTorch wheel, this wrapper replaces it
with the official PyTorch CUDA 12.8 wheel before launching the diagnostic in a
fresh child process. It does not load or modify the production checkpoint and
does not start training.

Run:
    python kaggle/run_gdn2_fla_fp32.py
"""
from __future__ import annotations

import importlib.metadata
import importlib.util
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMPL = ROOT / "kaggle" / "run_gdn2_fla_fp32_qualification.py"

QUALIFIED_TORCH = "2.10.0"
QUALIFIED_TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
QUALIFIED_TRITON = "3.6.0"


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _pip_install(*args: str) -> None:
    print(f"[bootstrap] pip install {' '.join(args)}", flush=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", *args],
        check=True,
    )


def _gpu_visible_to_system() -> bool:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return False
    completed = subprocess.run(
        [nvidia_smi, "-L"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode == 0 and completed.stdout.strip():
        print(f"[bootstrap] {completed.stdout.strip()}", flush=True)
        return True
    return False


def _torch_version() -> str | None:
    try:
        return importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        return None


def _torch_cuda_build_available() -> bool:
    """Check the installed wheel without requiring it to see a live GPU."""

    try:
        import torch
    except Exception:
        return False
    return torch.version.cuda is not None and "+cpu" not in torch.__version__


def _install_qualified_cuda_torch() -> None:
    print(
        "[bootstrap] installing the known-qualified CUDA PyTorch stack: "
        f"torch=={QUALIFIED_TORCH} from cu128",
        flush=True,
    )
    _pip_install(
        "--upgrade",
        "--force-reinstall",
        f"torch=={QUALIFIED_TORCH}",
        "--index-url",
        QUALIFIED_TORCH_INDEX,
    )


def _torch_triton_requirement() -> str | None:
    """Return PyTorch's own Triton requirement when package metadata exposes it."""

    try:
        requirements = importlib.metadata.requires("torch") or []
    except importlib.metadata.PackageNotFoundError:
        return None

    for requirement in requirements:
        if re.match(r"^\s*triton\b", requirement, flags=re.IGNORECASE):
            return requirement.split(";", 1)[0].strip()
    return None


def _fallback_triton_requirement() -> str | None:
    """Fallback pins for PyTorch versions already exercised by this project."""

    torch_version = _torch_version()
    if torch_version is None:
        return None
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", torch_version)
    if match is None:
        return None
    major, minor, patch = (int(part) for part in match.groups())
    if major != 2:
        return None

    exact = {
        (7, 0): "triton==3.3.0",
        (7, 1): "triton==3.3.1",
        (8, 0): "triton==3.4.0",
        (9, 0): "triton==3.5.0",
        (10, 0): f"triton=={QUALIFIED_TRITON}",
    }
    return exact.get((minor, patch))


def ensure_runtime() -> None:
    if not _gpu_visible_to_system():
        raise SystemExit(
            "No NVIDIA GPU is visible to the runtime. Enable a GPU accelerator "
            "(Tesla T4 is the target for this qualification), restart the runtime, "
            "then rerun this same entry point."
        )

    before = _torch_version()
    print(f"[bootstrap] initial torch={before!r}", flush=True)

    # A CPU-only wheel cannot run FLA/Triton even when the notebook has an
    # attached GPU. Replace it with the exact CUDA stack used in the earlier T4
    # qualification instead of guessing compatibility from the notebook image.
    if before is None or not _torch_cuda_build_available():
        _install_qualified_cuda_torch()
        after = _torch_version()
        print(
            f"[bootstrap] torch package after CUDA install={after!r}; "
            "verification happens in a fresh child process",
            flush=True,
        )

    if not _module_available("einops"):
        _pip_install("einops")

    if not _module_available("triton"):
        requirement = _torch_triton_requirement() or _fallback_triton_requirement()
        if requirement is None:
            raise SystemExit(
                "Triton is still missing after CUDA PyTorch provisioning and no "
                "safe PyTorch-matched pin is available. Refusing to guess."
            )
        _pip_install(requirement)

    print(
        f"[bootstrap] package metadata: torch={_torch_version()} "
        f"triton={importlib.metadata.version('triton')} "
        f"einops={importlib.metadata.version('einops')}",
        flush=True,
    )


def _verify_cuda_in_fresh_process() -> None:
    code = (
        "import torch; "
        "print('[bootstrap-check] torch=' + torch.__version__); "
        "print('[bootstrap-check] torch_cuda=' + str(torch.version.cuda)); "
        "print('[bootstrap-check] cuda_available=' + str(torch.cuda.is_available())); "
        "print('[bootstrap-check] gpu=' + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else '<none>')); "
        "raise SystemExit(0 if torch.cuda.is_available() and torch.version.cuda is not None else 3)"
    )
    completed = subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=False)
    if completed.returncode != 0:
        raise SystemExit(
            "CUDA PyTorch verification failed in a fresh process. Restart the notebook "
            "runtime once, keep the GPU accelerator enabled, and rerun this entry point."
        )


def main() -> int:
    ensure_runtime()
    _verify_cuda_in_fresh_process()
    if not IMPL.is_file():
        raise SystemExit(f"qualification implementation not found: {IMPL}")
    print(f"[bootstrap] launching {IMPL.relative_to(ROOT)}", flush=True)
    completed = subprocess.run([sys.executable, str(IMPL)], cwd=ROOT)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
