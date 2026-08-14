#!/usr/bin/env python3
"""Beam launcher for Small-LLM single-GPU pretraining.

Run this file from the repository root so Beam syncs the complete checkout:

    python beam/launch.py --model 100M --tokens 10B --gpu RTX5090

The Beam adapter preserves the same scientific contract as the Modal adapter.
For the incremental 10B path it starts dataset production on CPU, stages and
verifies the checkpoint-aligned lead window on CPU, and only then dispatches a
GPU function.
"""
from __future__ import annotations

import argparse
import errno
import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

from beam import Image, Volume, function

LOCAL_REPO = Path(__file__).resolve().parents[1]
LOCAL_BEAM = Path(__file__).resolve().parent
DATA_ROOT = Path("/data")
RUN_ROOT = Path("/runs")
CACHE_ROOT = Path("/cache")

sys.path.insert(0, str(LOCAL_BEAM))
from profiles import (  # noqa: E402
    DEFAULT_GPU,
    DEFAULT_PRECISION,
    MICROBATCH_CANDIDATES,
    SEQUENCES_PER_BLOCK,
    SUPPORTED_GPUS,
    canonical_run_id,
    resolve_presets,
)

SECRETS = ["WANDB_API_KEY", "HF_TOKEN", "SMALL_LLM_HF_REPO_ID"]
RUNTIME_ENV = {
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    # Keep Triton's generated kernels on container-local scratch. Beam's
    # distributed Volumes are excellent for durable checkpoints and dataset
    # bytes, but kernel compilation needs ordinary local filesystem semantics.
    "TRITON_CACHE_DIR": "/tmp/small-llm-triton-cache",
    # Beam Volumes persist writes without a client commit call. POSIX fsync on
    # their distributed filesystem can block after an otherwise-complete
    # atomic checkpoint rename, so retain atomic manifests but skip local-disk
    # power-loss barriers on this provider only.
    "SMALL_LLM_CHECKPOINT_FSYNC": "0",
    # One-time, exact infrastructure migration from the launch source whose
    # step-250 checkpoint exposed the Beam fsync hang.
    "SMALL_LLM_INFRA_MIGRATION_PARENT_COMMIT": (
        "42b0376511ba1fc7ceecfbbafbeae2027530fc2d"
    ),
    "WANDB_INIT_TIMEOUT": "30",
    "PYTHONUNBUFFERED": "1",
}

DATA_VOLUME = Volume(name="small-llm-data", mount_path=str(DATA_ROOT))
RUN_VOLUME = Volume(name="small-llm-runs", mount_path=str(RUN_ROOT))
CACHE_VOLUME = Volume(name="small-llm-cache", mount_path=str(CACHE_ROOT))

_COMMON_PACKAGES = [
    "numpy>=2,<3",
    "packaging>=24",
    "fla-core==0.5.2",
    "wandb==0.26.1",
    "huggingface-hub>=1.5,<2",
]


def _training_image(*, base_image: str, torch_index: str) -> Image:
    return (
        Image(base_image=base_image, python_version="python3.12")
        .add_commands(
            [
                "apt-get update -y && apt-get install -y git",
                (
                    "python -m pip install torch==2.10.0 "
                    f"--index-url https://download.pytorch.org/whl/{torch_index}"
                ),
            ]
        )
        .add_python_packages(_COMMON_PACKAGES)
    )


# Blackwell first appears in CUDA 12.8. RTX5090 therefore gets a cu128 image,
# while the older serverless RTX4090/A10G lane stays on the CUDA-12.4 host
# family with PyTorch's cu126 wheel. CPU helpers use the older image and never
# allocate a GPU.
BLACKWELL_IMAGE = _training_image(
    base_image="docker.io/nvidia/cuda:12.8.1-devel-ubuntu24.04",
    torch_index="cu128",
)
LEGACY_SERVERLESS_IMAGE = _training_image(
    base_image="docker.io/nvidia/cuda:12.4.1-devel-ubuntu22.04",
    torch_index="cu126",
)
CPU_IMAGE = LEGACY_SERVERLESS_IMAGE


def _stage(name: str, **fields: object) -> None:
    print(json.dumps({"beam_stage": name, **fields}, sort_keys=True), flush=True)


def _require_remote_mapping(result: object, *, label: str) -> dict[str, object]:
    """Turn Beam's failure-as-None controller behavior into a useful error."""

    if result is None:
        raise RuntimeError(
            f"Beam remote {label} returned no result; inspect the failed task above or run "
            "`beam task list --filter status=error`"
        )
    if not isinstance(result, dict):
        raise RuntimeError(
            f"Beam remote {label} returned {type(result).__name__}, expected an object"
        )
    return result


def _retry_transient_volume_io(
    operation: Callable[[], object],
    *,
    timeout_seconds: float = 60.0,
) -> object:
    """Retry only EAGAIN from Beam's distributed-volume filesystem."""

    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            return operation()
        except OSError as error:
            if error.errno != errno.EAGAIN or time.monotonic() >= deadline:
                raise
            time.sleep(1.0)


def _repo_root() -> Path:
    """Resolve the synced repository root in both local and Beam containers."""

    candidate = Path(__file__).resolve().parents[1]
    if (candidate / "trainer").is_dir() and (candidate / "dataset").is_dir():
        return candidate
    cwd = Path.cwd().resolve()
    if (cwd / "trainer").is_dir() and (cwd / "dataset").is_dir():
        return cwd
    raise RuntimeError("Beam did not sync a complete Small-LLM repository checkout")


def _install_beam_imports() -> Path:
    repo = _repo_root()
    adapter = repo / "beam"
    for path in (adapter, repo):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    return repo


def _local_source_commit() -> str:
    try:
        root = Path(
            subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"], cwd=LOCAL_REPO, text=True
            ).strip()
        ).resolve()
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=LOCAL_REPO, text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=LOCAL_REPO, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("run the Beam launcher from a cloned Small-LLM repository") from error
    if root != LOCAL_REPO.resolve():
        raise RuntimeError(f"repository root mismatch: expected {LOCAL_REPO}, found {root}")
    if Path.cwd().resolve() != root:
        raise RuntimeError("run `python beam/launch.py ...` from the repository root so Beam syncs all source files")
    if dirty:
        raise RuntimeError("the controlling Small-LLM checkout is dirty; commit changes before launch")
    return head


class _BeamDurabilityHook:
    """Compatibility shim for the provider-neutral runtime's Volume commit hook.

    Beam distributed-volume writes are persistent without an explicit commit API.
    Cross-container visibility is checked separately before GPU dispatch.
    """

    def commit(self) -> None:
        return None


NOOP_VOLUME = _BeamDurabilityHook()


@function(
    name="small-llm-import-preflight",
    image=CPU_IMAGE,
    cpu=2,
    memory="4Gi",
    timeout=300,
    retries=1,
    volumes=[RUN_VOLUME],
    env=RUNTIME_ENV,
)
def remote_import_preflight() -> dict[str, object]:
    """Fail on CPU before requesting a GPU if source sync/imports are broken."""

    repo = _install_beam_imports()
    required = [
        repo / "beam" / "profiles.py",
        repo / "beam" / "runtime.py",
        repo / "beam" / "model_repo_checkpoint.py",
        repo / "beam" / "rolling_dataset.py",
        repo / "beam" / "rolling_producer.py",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("Beam source sync is incomplete: " + json.dumps(missing))

    import model_repo_checkpoint as checkpoint_transport  # noqa: PLC0415

    checkpoint_transport.install_model_repo_checkpoint_transport()
    import profiles as remote_profiles  # noqa: PLC0415
    import rolling_dataset as remote_rolling  # noqa: F401, PLC0415
    import rolling_producer as remote_producer  # noqa: F401, PLC0415
    import runtime as remote_runtime  # noqa: F401, PLC0415
    from dataset.src.remote import ensure_safe_directory  # noqa: PLC0415

    _retry_transient_volume_io(lambda: ensure_safe_directory(RUN_ROOT))
    probe = RUN_ROOT / ".small-llm-safe-path-preflight"
    _retry_transient_volume_io(lambda: ensure_safe_directory(probe))
    _retry_transient_volume_io(probe.rmdir)
    return {
        "status": "ok",
        "repo": str(repo),
        "profiles": str(Path(remote_profiles.__file__).resolve()),
        "run_root": str(RUN_ROOT.resolve()),
    }


@function(
    name="small-llm-dataset-producer",
    image=CPU_IMAGE,
    cpu=4,
    memory="16Gi",
    timeout=-1,
    retries=3,
    headless=True,
    secrets=SECRETS,
    volumes=[CACHE_VOLUME],
    env=RUNTIME_ENV,
)
def produce_rolling_dataset_remote(model: str, tokens: str) -> dict[str, object]:
    """Cheap CPU producer: ClimbMix ranges -> verified immutable HF READY shards."""

    repo = _install_beam_imports()
    from rolling_producer import produce_incremental_dataset  # noqa: PLC0415

    return produce_incremental_dataset(
        model=model,
        tokens=tokens,
        repo_root=repo,
        producer_root=CACHE_ROOT / "producer",
        commit_cache_volume=NOOP_VOLUME.commit,
    )


@function(
    name="small-llm-dataset-stage",
    image=CPU_IMAGE,
    cpu=2,
    memory="8Gi",
    timeout=-1,
    retries=3,
    secrets=SECRETS,
    volumes=[RUN_VOLUME, CACHE_VOLUME],
    env=RUNTIME_ENV,
)
def stage_rolling_dataset_remote(model: str, tokens: str) -> dict[str, object]:
    """CPU gate: resolve checkpoint, wait for READY lead, download and verify it."""

    _install_beam_imports()
    from model_repo_checkpoint import install_model_repo_checkpoint_transport  # noqa: PLC0415

    install_model_repo_checkpoint_transport()
    from rolling_dataset import stage_for_h100  # noqa: PLC0415

    return _retry_transient_volume_io(
        lambda: stage_for_h100(
            model=model,
            tokens=tokens,
            cache_root=CACHE_ROOT,
            run_root=RUN_ROOT,
        )
    )


@function(
    name="small-llm-dataset-visibility",
    image=CPU_IMAGE,
    cpu=2,
    memory="8Gi",
    timeout=180,
    retries=1,
    secrets=SECRETS,
    volumes=[CACHE_VOLUME],
    env=RUNTIME_ENV,
)
def verify_staged_dataset_visible_remote(
    model: str,
    tokens: str,
    dataset_dir: str,
    required_block: int,
) -> dict[str, object]:
    """Verify a stage from a fresh container before any GPU is allocated.

    Beam documents up to ~60 seconds of distributed-volume propagation delay.
    Polling from a new CPU container proves the GPU container can see a complete,
    hash-verified bootstrap window rather than assuming immediate coherence.
    """

    _install_beam_imports()
    from dataset.qualification import get_profile  # noqa: PLC0415
    from rolling_dataset import hf_dataset_bucket_id  # noqa: PLC0415

    _, token_preset = resolve_presets(model, tokens)
    profile = get_profile(token_preset.dataset_profile)
    if profile.run_id is None:
        raise RuntimeError("rolling dataset profile has no run ID")
    dataset = Path(dataset_dir)
    bucket_id = hf_dataset_bucket_id()
    deadline = time.monotonic() + 120.0
    last_error: str | None = None
    while time.monotonic() < deadline:
        try:
            marker = json.loads((dataset / "rolling_cache_stage.json").read_text(encoding="utf-8"))
            if marker.get("transport") == "hf-bucket-incremental-frontier-v1":
                from dataset.incremental_stage import verify_incremental_stage  # noqa: PLC0415

                result = verify_incremental_stage(
                    destination=dataset,
                    bucket_id=bucket_id,
                    run_id=profile.run_id,
                    required_train_block=required_block,
                )
            else:
                from dataset.rolling_cache import verify_staged_dataset  # noqa: PLC0415

                result = verify_staged_dataset(
                    destination=dataset,
                    bucket_id=bucket_id,
                    run_id=profile.run_id,
                    required_train_block=required_block,
                )
            return {"status": "visible", "verification": result, "dataset_dir": str(dataset)}
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
            last_error = f"{type(error).__name__}: {error}"
            time.sleep(2.0)
    raise RuntimeError(f"Beam cache Volume did not become fully visible on CPU: {last_error}")


def _train_impl(
    model: str,
    tokens: str,
    source_commit: str,
    dataset_dir: str,
    max_steps_this_session: int,
    microbatch_size: int,
    precision: str,
) -> dict[str, object]:
    repo = _install_beam_imports()
    from model_repo_checkpoint import (  # noqa: PLC0415
        install_model_repo_checkpoint_transport,
        run_training,
    )

    install_model_repo_checkpoint_transport()
    _, token_preset = resolve_presets(model, tokens)
    if token_preset.dataset_transport == "hf_rolling_shards":
        from rolling_dataset import run_staged_training  # noqa: PLC0415

        return run_staged_training(
            model=model,
            tokens=tokens,
            source_commit=source_commit,
            dataset_dir=dataset_dir,
            max_steps_this_session=max_steps_this_session,
            microbatch_size=microbatch_size,
            precision=precision,
            repo_root=repo,
            run_root=RUN_ROOT,
            cache_root=CACHE_ROOT,
            run_volume=NOOP_VOLUME,
            cache_volume=NOOP_VOLUME,
        )
    return run_training(
        model=model,
        tokens=tokens,
        source_commit=source_commit,
        dataset_dir=dataset_dir,
        max_steps_this_session=max_steps_this_session,
        microbatch_size=microbatch_size,
        precision=precision,
        repo_root=repo,
        data_root=DATA_ROOT,
        run_root=RUN_ROOT,
        cache_root=CACHE_ROOT,
        run_volume=NOOP_VOLUME,
        cache_volume=NOOP_VOLUME,
    )


_GPU_FUNCTION_KWARGS = {
    "cpu": 4,
    "memory": "32Gi",
    "timeout": -1,
    "retries": 3,
    "secrets": SECRETS,
    "volumes": [DATA_VOLUME, RUN_VOLUME, CACHE_VOLUME],
    "env": RUNTIME_ENV,
}


@function(
    name="small-llm-train-rtx5090",
    gpu="RTX5090",
    image=BLACKWELL_IMAGE,
    **_GPU_FUNCTION_KWARGS,
)
def train_rtx5090_remote(
    model: str,
    tokens: str,
    source_commit: str,
    dataset_dir: str = "",
    max_steps_this_session: int = 0,
    microbatch_size: int = 0,
    precision: str = DEFAULT_PRECISION,
) -> dict[str, object]:
    return _train_impl(
        model, tokens, source_commit, dataset_dir, max_steps_this_session, microbatch_size, precision
    )


@function(
    name="small-llm-train-rtx4090",
    gpu="RTX4090",
    image=LEGACY_SERVERLESS_IMAGE,
    **_GPU_FUNCTION_KWARGS,
)
def train_rtx4090_remote(
    model: str,
    tokens: str,
    source_commit: str,
    dataset_dir: str = "",
    max_steps_this_session: int = 0,
    microbatch_size: int = 0,
    precision: str = DEFAULT_PRECISION,
) -> dict[str, object]:
    return _train_impl(
        model, tokens, source_commit, dataset_dir, max_steps_this_session, microbatch_size, precision
    )


@function(
    name="small-llm-train-a10g",
    gpu="A10G",
    image=LEGACY_SERVERLESS_IMAGE,
    **_GPU_FUNCTION_KWARGS,
)
def train_a10g_remote(
    model: str,
    tokens: str,
    source_commit: str,
    dataset_dir: str = "",
    max_steps_this_session: int = 0,
    microbatch_size: int = 0,
    precision: str = DEFAULT_PRECISION,
) -> dict[str, object]:
    return _train_impl(
        model, tokens, source_commit, dataset_dir, max_steps_this_session, microbatch_size, precision
    )


GPU_FUNCTIONS = {
    "RTX5090": train_rtx5090_remote,
    "RTX4090": train_rtx4090_remote,
    "A10G": train_a10g_remote,
}


def _start_remote_thread(
    label: str,
    call: Callable[..., dict[str, object]],
    output: "queue.Queue[tuple[str, str, object]]",
    *args: object,
) -> threading.Thread:
    def worker() -> None:
        try:
            result = _require_remote_mapping(call(*args), label=label)
        except BaseException as error:  # noqa: BLE001 - relay remote failure to controller
            output.put((label, "error", error))
        else:
            output.put((label, "done", result))

    thread = threading.Thread(target=worker, name=f"beam-{label}", daemon=True)
    thread.start()
    return thread


def _stage_with_incremental_producer(model: str, tokens: str) -> tuple[dict[str, object], object]:
    """Supervise producer+stager concurrently until the CPU stage authorizes GPU dispatch."""

    events: "queue.Queue[tuple[str, str, object]]" = queue.Queue()
    _start_remote_thread(
        "producer",
        produce_rolling_dataset_remote.remote,
        events,
        model,
        tokens,
    )
    _start_remote_thread(
        "stage",
        stage_rolling_dataset_remote.remote,
        events,
        model,
        tokens,
    )
    producer_result: object = {"status": "running_headless"}
    while True:
        label, status, payload = events.get()
        if status == "error":
            raise RuntimeError(f"Beam CPU {label} failed before GPU dispatch") from payload
        if label == "producer":
            producer_result = payload
            continue
        if label == "stage":
            if not isinstance(payload, dict):
                raise RuntimeError("Beam CPU stage returned a non-object result")
            return payload, producer_result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--gpu", default=DEFAULT_GPU, choices=sorted(SUPPORTED_GPUS))
    parser.add_argument("--dataset-dir", default="")
    parser.add_argument("--max-steps-this-session", type=int, default=0)
    parser.add_argument("--microbatch-size", type=int, default=0)
    parser.add_argument("--precision", default=DEFAULT_PRECISION)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    model_preset, token_preset = resolve_presets(args.model, args.tokens)
    if not 0 <= args.microbatch_size <= SEQUENCES_PER_BLOCK:
        raise ValueError(f"microbatch-size must be 0 (auto) or 1..{SEQUENCES_PER_BLOCK}")
    if args.max_steps_this_session < 0:
        raise ValueError("max-steps-this-session cannot be negative")
    if args.precision != "fp16":
        raise ValueError("the Beam adapter is frozen to fp16")
    if token_preset.dataset_transport == "hf_rolling_shards" and args.dataset_dir:
        raise ValueError("rolling HF datasets use the CPU-managed Beam cache; do not supply --dataset-dir")

    from dataset.qualification import get_profile

    dataset_profile = get_profile(token_preset.dataset_profile)
    source_commit = _local_source_commit()
    payload = {
        "action": "train",
        "runtime": "beam/launch.py",
        "model": model_preset.label,
        "tokens": token_preset.label,
        "model_size": model_preset.trainer_size,
        "dataset_profile": token_preset.dataset_profile,
        "dataset_transport": token_preset.dataset_transport,
        "incremental_dataset_producer": bool(dataset_profile.incremental_frontier),
        "run_id": canonical_run_id(model_preset, token_preset),
        "gpu": args.gpu,
        "microbatch_size": (
            "auto:" + ",".join(str(value) for value in MICROBATCH_CANDIDATES)
            if args.microbatch_size == 0
            else args.microbatch_size
        ),
        "precision": args.precision,
        "source_commit": source_commit,
        "dataset_dir": (
            "CPU-stage checkpoint-aligned HF shard frontier"
            if token_preset.dataset_transport == "hf_rolling_shards"
            else args.dataset_dir or "auto-discover unique matching Beam dataset"
        ),
        "max_steps_this_session": args.max_steps_this_session or "remaining plan",
        "resume": "automatic_verified_beam_volume_then_hf_model_repo",
        "gpu_dispatch_gate": "CPU import + dataset stage + fresh-container visibility verification",
    }
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    if args.dry_run:
        return 0

    _stage("remote_import_preflight_start")
    preflight = _require_remote_mapping(
        remote_import_preflight.remote(),
        label="import preflight",
    )
    _stage("remote_import_preflight_complete", **preflight)

    resolved_dataset_dir = args.dataset_dir
    producer_result: object = None
    if token_preset.dataset_transport == "hf_rolling_shards":
        _stage("cpu_dataset_stage_start", dataset_profile=token_preset.dataset_profile)
        if dataset_profile.incremental_frontier:
            staged, producer_result = _stage_with_incremental_producer(
                model_preset.label, token_preset.label
            )
        else:
            staged = _require_remote_mapping(
                stage_rolling_dataset_remote.remote(model_preset.label, token_preset.label),
                label="dataset stage",
            )
        _stage(
            "cpu_dataset_stage_complete",
            status=staged.get("status"),
            next_block_id=staged.get("next_block_id"),
            dataset_dir=staged.get("dataset_dir"),
            gpu_dispatch_allowed=staged.get("h100_dispatch_allowed"),
        )
        if staged.get("status") == "training_complete":
            print(
                json.dumps(
                    {
                        "status": "already_complete_cpu_gate",
                        "run_id": payload["run_id"],
                        "completed_steps": staged.get("completed_steps"),
                        "gpu_allocated": False,
                        "dataset_producer": producer_result,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                flush=True,
            )
            return 0
        if staged.get("status") != "ready" or staged.get("h100_dispatch_allowed") is not True:
            raise RuntimeError("Beam CPU rolling-dataset stage did not authorize GPU dispatch")
        remote_dataset_dir = staged.get("dataset_dir")
        next_block = staged.get("next_block_id")
        if not isinstance(remote_dataset_dir, str) or not remote_dataset_dir:
            raise RuntimeError("Beam CPU stage returned no dataset directory")
        if isinstance(next_block, bool) or not isinstance(next_block, int):
            raise RuntimeError("Beam CPU stage returned no valid next block")
        _stage("cpu_volume_visibility_start", dataset_dir=remote_dataset_dir, required_block=next_block)
        visible = _require_remote_mapping(
            verify_staged_dataset_visible_remote.remote(
                model_preset.label,
                token_preset.label,
                remote_dataset_dir,
                next_block,
            ),
            label="dataset visibility",
        )
        _stage("cpu_volume_visibility_complete", status=visible.get("status"))
        resolved_dataset_dir = remote_dataset_dir

    _stage(
        "dispatch_remote_training",
        run_id=payload["run_id"],
        gpu=args.gpu,
        dataset_transport=token_preset.dataset_transport,
    )
    gpu_function = GPU_FUNCTIONS[args.gpu]
    result = _require_remote_mapping(
        gpu_function.remote(
            model_preset.label,
            token_preset.label,
            source_commit,
            resolved_dataset_dir,
            args.max_steps_this_session,
            args.microbatch_size,
            args.precision,
        ),
        label="training",
    )
    if producer_result is not None:
        result["dataset_producer"] = producer_result
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
