#!/usr/bin/env python3
"""Beam RTX4090 launcher for the accepted production MoE identity."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import moe_production as _production


def _load_existing_launcher():
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
CPU_IMAGE = _base.CPU_IMAGE
LEGACY_SERVERLESS_IMAGE = _base.LEGACY_SERVERLESS_IMAGE
SECRETS = _base.SECRETS
RUNTIME_ENV = _base.RUNTIME_ENV


def _require_source_commit(source_commit: str) -> None:
    actual = _base._local_source_commit()
    if source_commit != actual:
        raise RuntimeError(
            f"--source-commit {source_commit} does not match clean checkout HEAD {actual}"
        )


@_base.function(
    name="small-llm-moe-production-prepare",
    image=CPU_IMAGE,
    cpu=2,
    memory="8Gi",
    timeout=1800,
    retries=1,
    secrets=SECRETS,
    volumes=[_base.DATA_VOLUME, _base.RUN_VOLUME, _base.CACHE_VOLUME],
    env=RUNTIME_ENV,
)
def prepare_production_cpu(payload: dict[str, object]) -> dict[str, object]:
    """Decode a real training block with the accepted 8,000-token semantic bound."""

    request = _production.request_from_payload(payload)
    return _production.prepare_dataset(request)


@_base.function(
    name="small-llm-moe-production-rtx4090",
    gpu="RTX4090",
    image=LEGACY_SERVERLESS_IMAGE,
    **_base._GPU_FUNCTION_KWARGS,
)
def train_production_rtx4090(payload: dict[str, object]) -> dict[str, object]:
    """Run only ``MoEModelConfig.accepted()`` on the production RTX4090 path."""

    return _production.run_provider_payload(
        payload,
        run_root=RUN_ROOT,
        repo_root=_base._repo_root(),
        volume_commit=_base.NOOP_VOLUME.commit,
    )


def main(argv: list[str] | None = None) -> int:
    """CPU-gate the dataset, then dispatch the accepted 64E/Top-2 model to RTX4090."""

    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--precision", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--microbatch-size", type=int, default=8)
    parser.add_argument("--resume")
    parser.add_argument("--sequences-per-block", type=int)
    parser.add_argument("--checkpoint-every-steps", type=int, default=1000)
    parser.add_argument("--keep-last-checkpoints", type=int, default=3)
    parser.add_argument("--milestone-every-steps", type=int, default=0)
    # The Beam GPU function has no wall-clock timeout, so draining stays off by default.
    parser.add_argument("--max-wall-seconds", type=float, default=0.0)
    # Negative means "as many as the corpus run contract plans".
    parser.add_argument("--validation-blocks", type=int, default=-1)
    parser.add_argument("--compile", dest="compile_mode", choices=("off", "blocks"), default="off")
    parser.add_argument("--allow-partial-corpus", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    _require_source_commit(args.source_commit)
    request = _production.ProductionRequest(
        run_id=args.run_id,
        dataset_dir=args.dataset_dir,
        total_steps=args.steps,
        precision=args.precision,
        microbatch_size=args.microbatch_size,
        source_commit=args.source_commit,
        resume=args.resume,
        sequences_per_block=args.sequences_per_block,
        checkpoint_every_steps=args.checkpoint_every_steps,
        keep_last_checkpoints=args.keep_last_checkpoints,
        milestone_every_steps=args.milestone_every_steps,
        max_wall_seconds=args.max_wall_seconds,
        validation_blocks=None if args.validation_blocks < 0 else args.validation_blocks,
        compile_mode=args.compile_mode,
        allow_partial_corpus=args.allow_partial_corpus,
    )
    payload = _production.request_payload(request)
    display = {
        **payload,
        "provider": "beam",
        "gpu": "RTX4090",
        "command": _production.build_training_command(request, run_root=RUN_ROOT, schedule=_production.DISPLAY_SCHEDULE),
    }
    print(json.dumps(display, indent=2, sort_keys=True), flush=True)
    if args.dry_run:
        return 0

    prepared = prepare_production_cpu.remote(payload)
    if not isinstance(prepared, dict) or prepared.get("status") != "ready":
        raise RuntimeError(
            f"CPU production preparation did not authorize RTX4090 dispatch: {prepared!r}"
        )
    if prepared.get("identity") != _production.accepted_identity():
        raise RuntimeError("CPU production preparation returned a different model identity")
    result = train_production_rtx4090.remote(payload)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if result.get("status") == "incomplete":
        raise SystemExit(f"segment ended at step {result.get('steps_reached')} before the target: corpus exhausted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
