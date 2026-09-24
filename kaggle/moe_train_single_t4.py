#!/usr/bin/env python3
"""Single-T4 MoE trainer with the already-qualified six-config Triton autotune cap.

Import this shim before importing MoE or FLA; the cap only selects execution
kernels, never changes the saved architecture, optimizer, or data cursor.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv: list[str] | None = None) -> int:
    os.environ["TRITON_CACHE_AUTOTUNING"] = "1"
    os.environ["FLA_CACHE_RESULTS"] = "1"
    import torch

    if torch.cuda.device_count() != 2 or any(
        torch.cuda.get_device_name(i) != "Tesla T4" or torch.cuda.get_device_capability(i) != (7, 5)
        for i in range(2)
    ):
        raise RuntimeError("this execution shim is restricted to Kaggle's two Tesla T4 GPUs")
    source = ROOT / "kaggle" / "src" / "dual_t4_train.py"
    spec = importlib.util.spec_from_file_location("small_llm_t4_autotune", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("missing qualified T4 autotune policy")
    shim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shim)
    shim._require_runtime(torch)
    shim._install_bounded_autotune()
    from MOE_model.__main__ import main as train

    return train(argv)


if __name__ == "__main__":
    raise SystemExit(main())
