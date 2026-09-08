#!/usr/bin/env python3
"""Beam RTX4090 pilot entry point for the dense and MoE 100M/2B arms.

The provider-neutral subprocess, resume and checkpoint protocol is implemented
in ``moe_pilot``. This file binds it to the existing Beam image and
durable Volumes and keeps CPU preparation ahead of GPU allocation.
"""

from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import moe_pilot as _pilot

_REPO_ROOT = Path(_pilot.__file__).resolve().parent


def _load_existing_launcher():
    """Load canonical Beam resources without duplicating its image recipe."""

    source = _REPO_ROOT / "beam" / "launch.py"
    spec = importlib.util.spec_from_file_location("small_llm_beam_base_launch", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load canonical Beam launcher: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_base = _load_existing_launcher()
DATA_ROOT = _base.DATA_ROOT
RUN_ROOT = _base.RUN_ROOT
CACHE_ROOT = _base.CACHE_ROOT
DATA_VOLUME = _base.DATA_VOLUME
RUN_VOLUME = _base.RUN_VOLUME
CACHE_VOLUME = _base.CACHE_VOLUME
CPU_IMAGE = _base.CPU_IMAGE
LEGACY_SERVERLESS_IMAGE = _base.LEGACY_SERVERLESS_IMAGE
SECRETS = _base.SECRETS
RUNTIME_ENV = _base.RUNTIME_ENV


def _require_source_commit(source_commit: str) -> None:
    """Bind the declared identity to the clean checkout copied into IMAGE."""

    actual = _base._local_source_commit()
    if source_commit != actual:
        raise RuntimeError(
            f"--source-commit {source_commit} does not match clean checkout HEAD {actual}"
        )


@_base.function(
    name="small-llm-moe-pilot-prepare",
    image=CPU_IMAGE,
    cpu=2,
    memory="8Gi",
    timeout=1800,
    retries=1,
    secrets=SECRETS,
    volumes=[DATA_VOLUME, RUN_VOLUME, CACHE_VOLUME],
    env=RUNTIME_ENV,
)
def prepare_pilot_cpu(
    payload: dict[str, object],
) -> dict[str, object]:
    """Verify the mounted schema-v2 dataset and freeze the arm contract."""

    return _pilot.prepare_provider_payload(
        payload, provider="beam", run_root=RUN_ROOT, dataset_root=DATA_ROOT,
        volume_commit=_base.NOOP_VOLUME.commit,
    )


@_base.function(
    name="small-llm-moe-pilot-rtx4090",
    gpu="RTX4090",
    image=LEGACY_SERVERLESS_IMAGE,
    **_base._GPU_FUNCTION_KWARGS,
)
def train_pilot_rtx4090(
    payload: dict[str, object],
) -> dict[str, object]:
    """Run one bounded RTX4090 segment, preserving every local checkpoint."""

    return _pilot.run_provider_payload(
        payload, provider="beam", run_root=RUN_ROOT, volume_commit=_base.NOOP_VOLUME.commit,
        cache_root=CACHE_ROOT, cache_commit=_base.NOOP_VOLUME.commit,
    )


def main(argv: list[str] | None = None) -> int:
    """CPU-prepare, then dispatch the exact same request to RTX4090."""

    args = _pilot.parse_pilot_args(argv)
    if args.source_commit is None:
        raise SystemExit("--source-commit is required for a provider pilot")
    _require_source_commit(args.source_commit)
    request = _pilot.request_from_args(
        args, provider="beam", run_root=RUN_ROOT, source_commit=args.source_commit
    )
    if args.dry_run:
        print(json.dumps(_pilot.dry_run_payload(request), indent=2, sort_keys=True), flush=True)
        return 0
    payload = _pilot.request_payload(request)
    prepared = prepare_pilot_cpu.remote(payload)
    if not isinstance(prepared, dict) or prepared.get("status") != "ready":
        raise RuntimeError(f"CPU pilot preparation did not authorize RTX4090 dispatch: {prepared!r}")
    result = train_pilot_rtx4090.remote(payload)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
