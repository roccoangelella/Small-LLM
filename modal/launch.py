#!/usr/bin/env python3
"""Canonical Modal launcher for new Small-LLM single-GPU pretraining runs."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import modal

LOCAL_REPO = Path(__file__).resolve().parents[1]
LOCAL_MODAL = Path(__file__).resolve().parent
REMOTE_REPO = Path("/root/small-llm")
REMOTE_MODAL = REMOTE_REPO / "modal"
DATA_ROOT, RUN_ROOT, CACHE_ROOT = Path("/data"), Path("/runs"), Path("/cache")
APP_NAME = "small-llm-training"

sys.path.insert(0, str(LOCAL_MODAL))
from profiles import (  # noqa: E402
    DEFAULT_GPU,
    DEFAULT_PRECISION,
    MICROBATCH_CANDIDATES,
    SEQUENCES_PER_BLOCK,
    SUPPORTED_GPUS,
    canonical_run_id,
    resolve_presets,
)

DATA_VOLUME = modal.Volume.from_name("small-llm-data", create_if_missing=True)
RUN_VOLUME = modal.Volume.from_name("small-llm-runs", create_if_missing=True)
CACHE_VOLUME = modal.Volume.from_name("small-llm-cache", create_if_missing=True)
TRAINING_SECRET = modal.Secret.from_name("small-llm-training")

IMAGE = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.13")
    .apt_install("git")
    .uv_pip_install(
        "torch==2.10.0",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .uv_pip_install(
        "numpy>=2,<3",
        "fla-core==0.5.2",
        "wandb==0.26.1",
        "google-api-python-client>=2.100,<3",
        "google-auth>=2.20,<3",
        "google-auth-oauthlib>=1.0.0,<2",
        "huggingface-hub>=1.5,<2",
    )
    .env(
        {
            "PYTHONPATH": str(REMOTE_REPO),
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "WANDB_INIT_TIMEOUT": "30",
            "PYTHONUNBUFFERED": "1",
        }
    )
    # copy=False mounts source at container startup, so Modal requires this to
    # remain the final image mutation; later build steps would be invalid.
    .add_local_dir(
        LOCAL_REPO,
        remote_path=str(REMOTE_REPO),
        copy=False,
        ignore=[".git/**", ".venv/**", ".pytest_cache/**", "**/__pycache__/**", "*.pyc"],
    )
)
app = modal.App(APP_NAME, image=IMAGE)


def _stage(name: str, **fields: object) -> None:
    print(json.dumps({"modal_stage": name, **fields}, sort_keys=True), flush=True)


def _local_source_commit() -> str:
    try:
        root = Path(
            subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=LOCAL_REPO, text=True).strip()
        ).resolve()
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=LOCAL_REPO, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=LOCAL_REPO, text=True).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("run the Modal launcher from a cloned Small-LLM repository") from error
    if root != LOCAL_REPO.resolve():
        raise RuntimeError(f"repository root mismatch: expected {LOCAL_REPO}, found {root}")
    if dirty:
        raise RuntimeError("the controlling Small-LLM checkout is dirty; commit changes before launch")
    return head


@app.function(
    gpu=DEFAULT_GPU,
    timeout=24 * 60 * 60,
    retries=modal.Retries(max_retries=10, initial_delay=0.0),
    single_use_containers=True,
    secrets=[TRAINING_SECRET],
    volumes={
        str(DATA_ROOT): DATA_VOLUME.with_mount_options(read_only=True),
        str(RUN_ROOT): RUN_VOLUME,
        str(CACHE_ROOT): CACHE_VOLUME,
    },
)
def train_remote(
    model: str,
    tokens: str,
    source_commit: str,
    dataset_dir: str = "",
    max_steps_this_session: int = 0,
    microbatch_size: int = 0,
    precision: str = DEFAULT_PRECISION,
) -> dict[str, object]:
    _stage(
        "remote_runtime_start",
        model=model,
        tokens=tokens,
        source_commit=source_commit,
    )
    sys.path.insert(0, str(REMOTE_MODAL))
    from runtime import run_training

    result = run_training(
        model=model,
        tokens=tokens,
        source_commit=source_commit,
        dataset_dir=dataset_dir,
        max_steps_this_session=max_steps_this_session,
        microbatch_size=microbatch_size,
        precision=precision,
        repo_root=REMOTE_REPO,
        data_root=DATA_ROOT,
        run_root=RUN_ROOT,
        cache_root=CACHE_ROOT,
        run_volume=RUN_VOLUME,
        cache_volume=CACHE_VOLUME,
    )
    _stage(
        "remote_runtime_complete",
        status=result.get("status"),
        run_id=result.get("run_id"),
        completed_steps=result.get("completed_steps"),
    )
    return result


@app.local_entrypoint()
def main(
    model: str,
    tokens: str,
    gpu: str = DEFAULT_GPU,
    dataset_dir: str = "",
    max_steps_this_session: int = 0,
    microbatch_size: int = 0,
    precision: str = DEFAULT_PRECISION,
    dry_run: bool = False,
) -> None:
    model_preset, token_preset = resolve_presets(model, tokens)
    if gpu not in SUPPORTED_GPUS:
        raise ValueError(f"unsupported Modal GPU {gpu!r}")
    if not 0 <= microbatch_size <= SEQUENCES_PER_BLOCK:
        raise ValueError(f"microbatch-size must be 0 (auto) or 1..{SEQUENCES_PER_BLOCK}")
    if max_steps_this_session < 0:
        raise ValueError("max-steps-this-session cannot be negative")
    if precision != "fp16":
        raise ValueError("the first Modal production migration is frozen to fp16")
    source_commit = _local_source_commit()
    payload = {
        "action": "train",
        "runtime": "modal/launch.py",
        "model": model_preset.label,
        "tokens": token_preset.label,
        "model_size": model_preset.trainer_size,
        "dataset_profile": token_preset.dataset_profile,
        "run_id": canonical_run_id(model_preset, token_preset),
        "gpu": gpu,
        "microbatch_size": (
            "auto:" + ",".join(str(value) for value in MICROBATCH_CANDIDATES)
            if microbatch_size == 0
            else microbatch_size
        ),
        "precision": precision,
        "source_commit": source_commit,
        "dataset_dir": dataset_dir or "auto-discover unique matching dataset",
        "max_steps_this_session": max_steps_this_session or "remaining plan",
        "resume": "automatic_verified_modal_volume_then_hf_bucket",
    }
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    if dry_run:
        return
    _stage(
        "dispatch_remote_training",
        run_id=payload["run_id"],
        gpu=gpu,
    )
    result = train_remote.with_options(gpu=gpu).spawn(
        model_preset.label,
        token_preset.label,
        source_commit,
        dataset_dir,
        max_steps_this_session,
        microbatch_size,
        precision,
    ).get()
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    raise SystemExit("Use: modal run --detach modal/launch.py --model 100M --tokens 2B")
