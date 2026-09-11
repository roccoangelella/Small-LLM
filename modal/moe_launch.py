#!/usr/bin/env python3
"""Modal H100 pilot entry point for the dense and MoE 100M/2B arms.

The subprocess, resume and checkpoint protocol lives in ``moe_pilot``.
This module only binds that protocol to the existing Modal image, Volumes and
secret. The CPU preparation function always runs before the H100 function.
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
    """Load canonical Modal resources without duplicating its image recipe."""

    source = _REPO_ROOT / "modal" / "launch.py"
    spec = importlib.util.spec_from_file_location("small_llm_modal_base_launch", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load canonical Modal launcher: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_base = _load_existing_launcher()
_modal = _base.modal
app = _modal.App("small-llm-moe-pilot", image=_base.IMAGE)

DATA_ROOT = _base.DATA_ROOT
RUN_ROOT = _base.RUN_ROOT
CACHE_ROOT = _base.CACHE_ROOT
DATA_VOLUME = _base.DATA_VOLUME
RUN_VOLUME = _base.RUN_VOLUME
CACHE_VOLUME = _base.CACHE_VOLUME
TRAINING_SECRET = _base.TRAINING_SECRET


def _require_source_commit(source_commit: str) -> None:
    """Bind the declared identity to the clean checkout copied into IMAGE."""

    actual = _base._local_source_commit()
    if source_commit != actual:
        raise RuntimeError(
            f"--source-commit {source_commit} does not match clean checkout HEAD {actual}"
        )


@app.function(
    cpu=2,
    memory=8192,
    timeout=30 * 60,
    retries=1,
    secrets=[TRAINING_SECRET],
    volumes={
        str(DATA_ROOT): DATA_VOLUME.with_mount_options(read_only=True),
        str(RUN_ROOT): RUN_VOLUME,
        str(CACHE_ROOT): CACHE_VOLUME,
    },
)
def prepare_pilot_cpu(
    payload: dict[str, object],
) -> dict[str, object]:
    """Verify the mounted schema-v2 dataset and freeze the arm contract."""

    return _pilot.prepare_provider_payload(
        payload, provider="modal", run_root=RUN_ROOT, dataset_root=DATA_ROOT,
        volume_commit=RUN_VOLUME.commit,
    )


@app.function(
    gpu="H100",
    timeout=24 * 60 * 60,
    retries=3,
    single_use_containers=True,
    secrets=[TRAINING_SECRET],
    volumes={
        str(DATA_ROOT): DATA_VOLUME.with_mount_options(read_only=True),
        str(RUN_ROOT): RUN_VOLUME,
        str(CACHE_ROOT): CACHE_VOLUME,
    },
)
def train_pilot_h100(
    payload: dict[str, object],
) -> dict[str, object]:
    """Run one bounded H100 segment, committing every child checkpoint."""

    return _pilot.run_provider_payload(
        payload, provider="modal", run_root=RUN_ROOT, volume_commit=RUN_VOLUME.commit,
        cache_root=CACHE_ROOT, cache_commit=CACHE_VOLUME.commit,
    )


def _argv(
    *,
    arm: str,
    run_id: str,
    dataset_dir: str,
    steps: int,
    precision: str,
    microbatch_size: int,
    source_commit: str,
    seed: int,
    gamma: float | None,
    probe_sequences: int,
    probe_lm_logits: bool,
    checkpoint_at_steps: str,
    profile_at_steps: str,
    dry_run: bool,
) -> list[str]:
    values = [
        "--arm", arm, "--run-id", run_id, "--dataset-dir", dataset_dir,
        "--steps", str(steps), "--precision", precision,
        "--microbatch-size", str(microbatch_size), "--source-commit", source_commit,
        "--seed", str(seed), "--probe-sequences", str(probe_sequences),
    ]
    if gamma is not None:
        values += ["--gamma", str(gamma)]
    if checkpoint_at_steps:
        values += ["--checkpoint-at-steps", checkpoint_at_steps]
    if profile_at_steps:
        values += ["--profile-at-steps", profile_at_steps]
    if probe_lm_logits:
        values.append("--probe-lm-logits")
    if dry_run:
        values.append("--dry-run")
    return values


@app.local_entrypoint()
def main(
    arm: str,
    run_id: str,
    dataset_dir: str,
    precision: str,
    microbatch_size: int,
    source_commit: str,
    steps: int = 100,
    gamma: float | None = None,
    seed: int = 17,
    probe_sequences: int = 2,
    probe_lm_logits: bool = False,
    checkpoint_at_steps: str = "",
    profile_at_steps: str = "",
    dry_run: bool = False,
) -> None:
    """CPU-prepare, then dispatch the exact same request to the H100."""

    args = _pilot.parse_pilot_args(_argv(
        arm=arm, run_id=run_id, dataset_dir=dataset_dir, steps=steps,
        precision=precision, microbatch_size=microbatch_size,
        source_commit=source_commit, seed=seed, gamma=gamma,
        probe_sequences=probe_sequences, probe_lm_logits=probe_lm_logits,
        checkpoint_at_steps=checkpoint_at_steps, profile_at_steps=profile_at_steps,
        dry_run=dry_run,
    ))
    _require_source_commit(source_commit)
    request = _pilot.request_from_args(
        args, provider="modal", run_root=RUN_ROOT, source_commit=source_commit
    )
    if args.dry_run:
        print(json.dumps(_pilot.dry_run_payload(request), indent=2, sort_keys=True), flush=True)
        return
    payload = _pilot.request_payload(request)
    prepared = prepare_pilot_cpu.remote(payload)
    if not isinstance(prepared, dict) or prepared.get("status") != "ready":
        raise RuntimeError(f"CPU pilot preparation did not authorize H100 dispatch: {prepared!r}")
    result = train_pilot_h100.remote(payload)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    raise SystemExit("use `modal run modal/moe_launch.py ...`")
