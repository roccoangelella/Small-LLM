#!/usr/bin/env python3
"""Canonical Modal launcher for new Small-LLM single-GPU pretraining runs."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import modal

REMOTE_REPO = Path("/root/small-llm")
REMOTE_MODAL = REMOTE_REPO / "modal"
_SOURCE_FILE = Path(__file__).resolve()
if _SOURCE_FILE.parent == Path("/root") and (REMOTE_MODAL / "profiles.py").is_file():
    LOCAL_REPO = REMOTE_REPO
    LOCAL_MODAL = REMOTE_MODAL
else:
    LOCAL_REPO = _SOURCE_FILE.parents[1]
    LOCAL_MODAL = _SOURCE_FILE.parent
DATA_ROOT, RUN_ROOT, CACHE_ROOT = Path("/data"), Path("/runs"), Path("/cache")
APP_NAME = "small-llm-training"

sys.path.insert(0, str(LOCAL_MODAL))
from cpu_supervision import await_stage_with_producer  # noqa: E402
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
    .add_local_python_source("profiles")
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
    timeout=2 * 60,
    volumes={str(RUN_ROOT): RUN_VOLUME},
)
def remote_import_preflight() -> dict[str, object]:
    """Fail on CPU before requesting an H100 if packaging or run-volume paths are broken."""

    profiles_path = Path("/root/profiles.py")
    runtime_path = REMOTE_MODAL / "runtime.py"
    checkpoint_transport_path = REMOTE_MODAL / "model_repo_checkpoint.py"
    rolling_path = REMOTE_MODAL / "rolling_dataset.py"
    producer_path = REMOTE_MODAL / "rolling_producer.py"
    if not profiles_path.is_file():
        raise RuntimeError(f"Modal image is missing explicitly packaged profiles helper: {profiles_path}")
    if not runtime_path.is_file():
        raise RuntimeError(f"Modal image is missing repository runtime: {runtime_path}")
    if not checkpoint_transport_path.is_file():
        raise RuntimeError(
            f"Modal image is missing model-repository checkpoint adapter: {checkpoint_transport_path}"
        )
    if not rolling_path.is_file():
        raise RuntimeError(f"Modal image is missing rolling-dataset runtime: {rolling_path}")
    if not producer_path.is_file():
        raise RuntimeError(f"Modal image is missing rolling-dataset producer adapter: {producer_path}")
    sys.path.insert(0, str(REMOTE_MODAL))
    import model_repo_checkpoint as checkpoint_transport  # noqa: PLC0415

    checkpoint_transport.install_model_repo_checkpoint_transport()
    import profiles as remote_profiles  # noqa: PLC0415
    import runtime as remote_runtime  # noqa: F401, PLC0415
    import rolling_dataset as remote_rolling  # noqa: F401, PLC0415
    import rolling_producer as remote_producer  # noqa: F401, PLC0415
    from dataset.src.remote import ensure_safe_directory  # noqa: PLC0415

    run_root = RUN_ROOT.resolve(strict=True)
    if not run_root.is_dir() or run_root.is_symlink():
        raise RuntimeError(f"Modal run Volume did not resolve to a real directory: {run_root}")
    probe = run_root / ".small-llm-safe-path-preflight"
    ensure_safe_directory(probe)
    probe.rmdir()

    return {
        "status": "ok",
        "profiles": str(Path(remote_profiles.__file__).resolve()),
        "runtime": str(runtime_path),
        "checkpoint_transport": str(checkpoint_transport_path),
        "rolling_runtime": str(rolling_path),
        "rolling_producer": str(producer_path),
        "run_root": str(run_root),
    }


@app.function(
    cpu=4.0,
    memory=8192,
    timeout=24 * 60 * 60,
    retries=modal.Retries(max_retries=3, initial_delay=1.0),
    single_use_containers=True,
    secrets=[TRAINING_SECRET],
    volumes={str(CACHE_ROOT): CACHE_VOLUME},
)
def produce_rolling_dataset_remote(model: str, tokens: str) -> dict[str, object]:
    """Cheap CPU producer: ClimbMix ranges -> verified immutable HF READY shards."""

    sys.path.insert(0, str(REMOTE_MODAL))
    from rolling_producer import produce_incremental_dataset  # noqa: PLC0415

    return produce_incremental_dataset(
        model=model,
        tokens=tokens,
        repo_root=REMOTE_REPO,
        producer_root=CACHE_ROOT / "producer",
        commit_cache_volume=lambda: getattr(CACHE_VOLUME, "commit")(),
    )


@app.function(
    timeout=6 * 60 * 60,
    retries=modal.Retries(max_retries=3, initial_delay=1.0),
    single_use_containers=True,
    secrets=[TRAINING_SECRET],
    volumes={
        str(RUN_ROOT): RUN_VOLUME.with_mount_options(read_only=True),
        str(CACHE_ROOT): CACHE_VOLUME,
    },
)
def stage_rolling_dataset_remote(model: str, tokens: str) -> dict[str, object]:
    """CPU-only gate: checkpoint-align, wait for lead, download, hash, commit."""

    sys.path.insert(0, str(REMOTE_MODAL))
    from model_repo_checkpoint import install_model_repo_checkpoint_transport  # noqa: PLC0415

    install_model_repo_checkpoint_transport()
    from rolling_dataset import stage_for_h100  # noqa: PLC0415

    result = stage_for_h100(
        model=model,
        tokens=tokens,
        cache_root=CACHE_ROOT,
        run_root=RUN_ROOT,
    )
    getattr(CACHE_VOLUME, "commit")()
    return result


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
    """Existing Modal-volume dataset path with unified HF model-repo checkpoints."""

    resolved_run_root = RUN_ROOT.resolve(strict=True)
    _stage(
        "remote_runtime_start",
        model=model,
        tokens=tokens,
        source_commit=source_commit,
        run_root=str(resolved_run_root),
    )
    sys.path.insert(0, str(REMOTE_MODAL))
    from model_repo_checkpoint import (  # noqa: PLC0415
        install_model_repo_checkpoint_transport,
        run_training,
    )

    install_model_repo_checkpoint_transport()
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
        run_root=resolved_run_root,
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


@app.function(
    gpu=DEFAULT_GPU,
    timeout=24 * 60 * 60,
    single_use_containers=True,
    secrets=[TRAINING_SECRET],
    volumes={
        str(RUN_ROOT): RUN_VOLUME,
        str(CACHE_ROOT): CACHE_VOLUME,
    },
)
def train_rolling_remote(
    model: str,
    tokens: str,
    source_commit: str,
    dataset_dir: str,
    max_steps_this_session: int = 0,
    microbatch_size: int = 0,
    precision: str = DEFAULT_PRECISION,
) -> dict[str, object]:
    """H100 path entered only after the CPU rolling-dataset gate succeeded."""

    resolved_run_root = RUN_ROOT.resolve(strict=True)
    _stage(
        "rolling_remote_runtime_start",
        model=model,
        tokens=tokens,
        source_commit=source_commit,
        dataset_dir=dataset_dir,
        run_root=str(resolved_run_root),
    )
    sys.path.insert(0, str(REMOTE_MODAL))
    from model_repo_checkpoint import install_model_repo_checkpoint_transport  # noqa: PLC0415

    install_model_repo_checkpoint_transport()
    from rolling_dataset import run_staged_training  # noqa: PLC0415

    result = run_staged_training(
        model=model,
        tokens=tokens,
        source_commit=source_commit,
        dataset_dir=dataset_dir,
        max_steps_this_session=max_steps_this_session,
        microbatch_size=microbatch_size,
        precision=precision,
        repo_root=REMOTE_REPO,
        run_root=resolved_run_root,
        cache_root=CACHE_ROOT,
        run_volume=RUN_VOLUME,
        cache_volume=CACHE_VOLUME,
    )
    _stage(
        "rolling_remote_runtime_complete",
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
    if token_preset.dataset_transport == "hf_rolling_shards" and dataset_dir:
        raise ValueError(
            "rolling HF datasets use the CPU-managed Modal cache path; do not supply --dataset-dir"
        )
    from dataset.qualification import get_profile  # noqa: PLC0415

    dataset_profile = get_profile(token_preset.dataset_profile)
    source_commit = _local_source_commit()
    payload = {
        "action": "train",
        "runtime": "modal/launch.py",
        "model": model_preset.label,
        "tokens": token_preset.label,
        "model_size": model_preset.trainer_size,
        "dataset_profile": token_preset.dataset_profile,
        "dataset_transport": token_preset.dataset_transport,
        "incremental_dataset_producer": bool(dataset_profile.incremental_frontier),
        "run_id": canonical_run_id(model_preset, token_preset),
        "gpu": gpu,
        "microbatch_size": (
            "auto:" + ",".join(str(value) for value in MICROBATCH_CANDIDATES)
            if microbatch_size == 0
            else microbatch_size
        ),
        "precision": precision,
        "source_commit": source_commit,
        "dataset_dir": (
            "CPU-stage checkpoint-aligned HF shard frontier"
            if token_preset.dataset_transport == "hf_rolling_shards"
            else dataset_dir or "auto-discover unique matching dataset"
        ),
        "max_steps_this_session": max_steps_this_session or "remaining plan",
        "resume": "automatic_verified_modal_volume_then_hf_model_repo",
    }
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    if dry_run:
        return

    _stage("remote_import_preflight_start")
    preflight = remote_import_preflight.remote()
    _stage(
        "remote_import_preflight_complete",
        status=preflight.get("status"),
        profiles=preflight.get("profiles"),
        runtime=preflight.get("runtime"),
        rolling_runtime=preflight.get("rolling_runtime"),
        rolling_producer=preflight.get("rolling_producer"),
        run_root=preflight.get("run_root"),
    )

    producer_call = None
    producer_result: dict[str, object] | None = None
    if token_preset.dataset_transport == "hf_rolling_shards" and dataset_profile.incremental_frontier:
        _stage(
            "cpu_dataset_producer_start",
            dataset_profile=token_preset.dataset_profile,
            dataset_run_id=dataset_profile.run_id,
        )
        # Do not wait here. The independent CPU staging call below polls the
        # READY frontier while this producer continues extending it.
        producer_call = produce_rolling_dataset_remote.spawn(
            model_preset.label,
            token_preset.label,
        )

    resolved_dataset_dir = dataset_dir
    if token_preset.dataset_transport == "hf_rolling_shards":
        _stage(
            "cpu_dataset_stage_start",
            run_id=payload["run_id"],
            dataset_profile=token_preset.dataset_profile,
        )
        stage_call = stage_rolling_dataset_remote.spawn(model_preset.label, token_preset.label)
        staged, producer_result = await_stage_with_producer(stage_call, producer_call)
        _stage(
            "cpu_dataset_stage_complete",
            status=staged.get("status"),
            next_block_id=staged.get("next_block_id"),
            dataset_dir=staged.get("dataset_dir"),
            incremental_frontier=staged.get("incremental_frontier"),
            h100_dispatch_allowed=staged.get("h100_dispatch_allowed"),
        )
        if staged.get("status") == "training_complete":
            if producer_call is not None and producer_result is None:
                try:
                    producer_result = producer_call.get(timeout=0)
                except TimeoutError:
                    try:
                        producer_call.cancel(terminate_containers=True)
                    except Exception:
                        pass
                    producer_result = {"status": "cancelled_training_already_complete"}
            result = {
                "status": "already_complete_cpu_gate",
                "run_id": payload["run_id"],
                "completed_steps": staged.get("completed_steps"),
                "dataset_transport": token_preset.dataset_transport,
                "dataset_producer": producer_result,
                "h100_allocated": False,
            }
            print(json.dumps(result, indent=2, sort_keys=True), flush=True)
            return
        if staged.get("status") != "ready" or staged.get("h100_dispatch_allowed") is not True:
            raise RuntimeError("CPU rolling-dataset stage did not authorize H100 dispatch")
        remote_dataset_dir = staged.get("dataset_dir")
        if not isinstance(remote_dataset_dir, str) or not remote_dataset_dir:
            raise RuntimeError("CPU rolling-dataset stage returned no dataset directory")
        resolved_dataset_dir = remote_dataset_dir

    _stage(
        "dispatch_remote_training",
        run_id=payload["run_id"],
        gpu=gpu,
        dataset_transport=token_preset.dataset_transport,
    )
    if token_preset.dataset_transport == "hf_rolling_shards":
        result = train_rolling_remote.with_options(gpu=gpu).spawn(
            model_preset.label,
            token_preset.label,
            source_commit,
            resolved_dataset_dir,
            max_steps_this_session,
            microbatch_size,
            precision,
        ).get()
    else:
        result = train_remote.with_options(gpu=gpu).spawn(
            model_preset.label,
            token_preset.label,
            source_commit,
            resolved_dataset_dir,
            max_steps_this_session,
            microbatch_size,
            precision,
        ).get()
    result = dict(result)
    if producer_call is not None:
        if producer_result is not None:
            result["dataset_producer"] = producer_result
        elif max_steps_this_session == 0 or result.get("status") == "complete":
            producer_result = producer_call.get()
            result["dataset_producer"] = producer_result
            _stage(
                "cpu_dataset_producer_complete",
                status=producer_result.get("status"),
                producer_complete=producer_result.get("producer_complete"),
            )
        else:
            try:
                producer_result = producer_call.get(timeout=0)
            except TimeoutError:
                result["dataset_producer"] = {
                    "status": "running_after_training_segment",
                    "function_call_id": getattr(producer_call, "object_id", None),
                }
            else:
                result["dataset_producer"] = producer_result
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    raise SystemExit(
        "Use: modal run --detach modal/launch.py --model 100M --tokens 2B "
        "or --model 100M --tokens 10B"
    )
