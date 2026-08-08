#!/usr/bin/env python3
"""Self-provisioning Kaggle/Colab entry point for the FP32 FLA GDN-2 qualification.

This wrapper keeps the scientific diagnostic in
``run_gdn2_fla_fp32_qualification.py`` unchanged. Its only job is to ensure
that the lightweight ``fla-core --no-deps`` install can actually import in a
notebook runtime by provisioning the small runtime dependencies that may be
missing, especially the Triton version expected by the already-installed
PyTorch build.

Run:
    python kaggle/run_gdn2_fla_fp32.py

The wrapped diagnostic still does not load or modify the production checkpoint
and does not start training.
"""
from __future__ import annotations

import importlib.metadata
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMPL = ROOT / "kaggle" / "run_gdn2_fla_fp32_qualification.py"


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _pip_install(requirement: str) -> None:
    print(f"[bootstrap] installing {requirement}", flush=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", requirement],
        check=True,
    )


def _torch_triton_requirement() -> str | None:
    """Return PyTorch's own Triton requirement when package metadata exposes it."""

    try:
        requirements = importlib.metadata.requires("torch") or []
    except importlib.metadata.PackageNotFoundError:
        raise SystemExit("PyTorch must already be installed in the notebook runtime")

    for requirement in requirements:
        if re.match(r"^\s*triton\b", requirement, flags=re.IGNORECASE):
            # The wrapper runs on Linux CUDA notebook runtimes, so use the
            # requirement itself while dropping its environment marker. This
            # preserves PyTorch's exact Triton pin instead of guessing a latest
            # version that may be ABI/JIT-incompatible.
            return requirement.split(";", 1)[0].strip()
    return None


def _fallback_triton_requirement() -> str | None:
    """Fallback for notebook PyTorch builds whose metadata omits Triton."""

    try:
        torch_version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
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
        (10, 0): "triton==3.6.0",
    }
    return exact.get((minor, patch))


def ensure_runtime() -> None:
    torch_version = importlib.metadata.version("torch")
    print(f"[bootstrap] torch={torch_version}", flush=True)

    if not _module_available("einops"):
        _pip_install("einops")

    if not _module_available("triton"):
        requirement = _torch_triton_requirement() or _fallback_triton_requirement()
        if requirement is None:
            raise SystemExit(
                "Triton is missing and this PyTorch build does not expose a safe "
                "Triton pin in package metadata. Refusing to guess a version. "
                f"Installed torch={torch_version}."
            )
        _pip_install(requirement)

    # Import only after any installation so an unusable partial module cannot
    # be cached by this bootstrap process.
    import triton  # noqa: F401

    print(
        f"[bootstrap] triton={importlib.metadata.version('triton')} einops="
        f"{importlib.metadata.version('einops')}",
        flush=True,
    )


def main() -> int:
    ensure_runtime()
    if not IMPL.is_file():
        raise SystemExit(f"qualification implementation not found: {IMPL}")
    print(f"[bootstrap] launching {IMPL.relative_to(ROOT)}", flush=True)
    completed = subprocess.run([sys.executable, str(IMPL)], cwd=ROOT)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
