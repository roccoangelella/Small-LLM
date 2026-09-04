#!/usr/bin/env python3
"""Public Kaggle entrypoint for the canonical 100M/10B pretraining probes.

The scientific probe implementation lives in ``kaggle/src/probes_100m_10b.py``.
After the Kaggle source-tree consolidation, the imported historical deep-decay
implementation still derives repository paths relative to its old location.
Normalize those execution-only paths here before delegating so the shared
provider-neutral runtime resolves from ``beam/runtime.py`` and the dual-T4
wrapper resolves from ``kaggle/src``.

The scientific implementation can bootstrap a private Hugging Face Hub runtime
by re-executing ``Path(__file__)``. Point its execution file back at this public
wrapper so that bootstrap restart preserves the same path normalization instead
of accidentally jumping into the lower-level ``kaggle/src`` module.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Sequence

KAGGLE = Path(__file__).resolve().parent
ROOT = KAGGLE.parent
SRC = KAGGLE / "src"
BEAM = ROOT / "beam"

for candidate in (ROOT, SRC):
    value = str(candidate)
    if value not in sys.path:
        sys.path.insert(0, value)


def _load_impl():
    path = SRC / "probes_100m_10b.py"
    spec = importlib.util.spec_from_file_location("small_llm_100m_10b_probes_impl", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load canonical probe implementation from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _normalize_deep_decay_paths(impl) -> None:
    deep_impl = impl._impl
    deep_impl.ROOT = ROOT
    deep_impl.KAGGLE = SRC
    deep_impl.BEAM = BEAM

    # Keep private-HF bootstrap re-exec on the stable public entrypoint. The
    # implementation's _ensure_probe_hf_runtime() executes Path(__file__), so
    # leaving __file__ pointed at kaggle/src would discard this normalization.
    impl.__file__ = str(Path(__file__).resolve())

    # A prior Kaggle import may already have cached ``runtime`` from
    # kaggle/src/runtime.py. The deep-decay helper needs beam/runtime.py.
    cached = sys.modules.get("runtime")
    if cached is not None:
        cached_path = Path(getattr(cached, "__file__", "")).resolve()
        if cached_path != (BEAM / "runtime.py").resolve():
            sys.modules.pop("runtime", None)


def main(argv: Sequence[str] | None = None) -> int:
    impl = _load_impl()
    _normalize_deep_decay_paths(impl)
    return int(impl.main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
