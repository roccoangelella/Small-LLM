#!/usr/bin/env python3
"""Human entry point for R-SFT dataset production.

Examples:
  python post_training/R-SFT/produce.py pilot --dry-run \
    --s0-bundle /path/to/100m-2b-sft-s0-bundle \
    --token-spec /path/to/reasoning-tokens.json \
    --output-dir artifacts/rsft-r0-pilot-630

  python post_training/R-SFT/produce.py pilot \
    --s0-bundle /path/to/100m-2b-sft-s0-bundle \
    --token-spec /path/to/reasoning-tokens.json \
    --output-dir artifacts/rsft-r0-pilot-630
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _load_production():
    path = Path(__file__).with_name("production.py")
    spec = importlib.util.spec_from_file_location("small_llm_rsft_production_cli", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load R-SFT production module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


production = _load_production()


if __name__ == "__main__":
    raise SystemExit(production.main())
