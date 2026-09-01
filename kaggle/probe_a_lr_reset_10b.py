#!/usr/bin/env python3
"""Safe entrypoint for disposable Probe A LR-reset branches on Kaggle.

The implementation lives in ``probe_a_lr_reset_10b_impl.py``. This thin shim
exists because the shared deep-decay HF runtime helper re-execs its own file;
Probe A must instead restart back into this probe entrypoint before delegating.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

KAGGLE = Path(__file__).resolve().parent
ROOT = KAGGLE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(KAGGLE) not in sys.path:
    sys.path.insert(0, str(KAGGLE))

import deep_decay_10b_from_15500 as deep_decay

HF_HUB_VERSION = getattr(deep_decay, "HF_HUB_VERSION", "1.5.0")


def _ensure_probe_hf_bucket_runtime(argv: Sequence[str]) -> None:
    """Re-exec this probe launcher, not the deep-decay launcher, with HF Hub 1.5."""

    if deep_decay._hf_bucket_api_available():
        return

    runtime = deep_decay._impl.WORK_ROOT / ".runtime" / f"huggingface-hub-{HF_HUB_VERSION}"
    marker = runtime / ".complete"
    if not marker.is_file():
        staging = runtime.with_name(runtime.name + ".tmp")
        shutil.rmtree(staging, ignore_errors=True)
        staging.parent.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--quiet",
                "--disable-pip-version-check",
                "--target",
                str(staging),
                f"huggingface_hub=={HF_HUB_VERSION}",
            ]
        )
        (staging / ".complete").write_text(HF_HUB_VERSION + "\n", encoding="utf-8")
        shutil.rmtree(runtime, ignore_errors=True)
        os.replace(staging, runtime)

    env = dict(os.environ)
    previous = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(runtime) + (os.pathsep + previous if previous else "")
    print(
        f"[kaggle-probe-a] Kaggle host lacks HF Storage Buckets API; "
        f"restarting this probe launcher with private huggingface_hub=={HF_HUB_VERSION}",
        flush=True,
    )
    os.execve(
        sys.executable,
        [sys.executable, str(Path(__file__).resolve()), *list(argv)],
        env,
    )


def _noop_hf_runtime_restart(argv: Sequence[str]) -> None:
    """Disable the imported deep-decay shim restart after this launcher handled it."""

    return None


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--dry-run" not in args:
        _ensure_probe_hf_bucket_runtime(args)

    # The implementation imports the deep-decay shim for checkpoint migration,
    # but it must never call that shim's self-reexec from Probe A. If the host
    # still lacks Storage Bucket support after the probe restart, downstream HF
    # calls should fail in-place instead of jumping to the deep-decay trainer.
    deep_decay._ensure_host_hf_bucket_runtime = _noop_hf_runtime_restart

    import probe_a_lr_reset_10b_impl as impl

    return int(impl.main(args))


if __name__ == "__main__":
    raise SystemExit(main())
