#!/usr/bin/env python3
"""Observable/watchdog wrapper for the disposable dual-T4 qualification.

The scientific qualification remains implemented in ``qualify_dual_t4.py``.
This wrapper only adds stage/block progress, runtime identity checks, GPU-memory
headroom checks, and bounded worker lifetimes so a stuck CUDA/Triton/NCCL
subprocess cannot silently consume a Kaggle session.
"""
from __future__ import annotations

import importlib.metadata
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

KAGGLE_DIR = Path(__file__).resolve().parent
REPO = KAGGLE_DIR.parent
if str(KAGGLE_DIR) not in sys.path:
    sys.path.insert(0, str(KAGGLE_DIR))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import qualify_dual_t4 as qualification

DEFAULT_WORKER_TIMEOUT_SECONDS = 600
MINIMUM_FREE_GPU_BYTES = 12 * 1024**3
EXPECTED_TORCH_VERSION = "2.10.0"
EXPECTED_CUDA_VERSION = "12.8"
EXPECTED_TRITON_VERSION = "3.6.0"
EXPECTED_FLA_VERSION = "0.5.2"
_STARTED = time.monotonic()


def _progress(message: str) -> None:
    elapsed = time.monotonic() - _STARTED
    print(f"[dual-t4 +{elapsed:7.1f}s] {message}", flush=True)


def _worker_name(command: Sequence[str]) -> str:
    try:
        index = list(command).index("--worker")
        return str(command[index + 1])
    except (ValueError, IndexError):
        return "unknown"


def _timeout_seconds() -> int:
    raw = os.environ.get(
        "SMALL_LLM_DUAL_T4_WORKER_TIMEOUT_SECONDS",
        str(DEFAULT_WORKER_TIMEOUT_SECONDS),
    )
    try:
        value = int(raw)
    except ValueError as error:
        raise qualification.QualificationFailure(
            "SMALL_LLM_DUAL_T4_WORKER_TIMEOUT_SECONDS must be an integer"
        ) from error
    if value <= 0:
        raise qualification.QualificationFailure(
            "SMALL_LLM_DUAL_T4_WORKER_TIMEOUT_SECONDS must be positive"
        )
    return value


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def _runtime_snapshot() -> dict[str, str]:
    import torch

    return {
        "torch": torch.__version__,
        "torch_base": torch.__version__.split("+", 1)[0],
        "cuda": str(torch.version.cuda),
        "fla_core": _distribution_version("fla-core"),
        "triton": _distribution_version("triton"),
    }


def _require_expected_runtime() -> dict[str, str]:
    snapshot = _runtime_snapshot()
    expected = {
        "torch_base": EXPECTED_TORCH_VERSION,
        "cuda": EXPECTED_CUDA_VERSION,
        "fla_core": EXPECTED_FLA_VERSION,
        "triton": EXPECTED_TRITON_VERSION,
    }
    mismatches = {
        key: {"expected": value, "actual": snapshot.get(key)}
        for key, value in expected.items()
        if snapshot.get(key) != value
    }
    if mismatches:
        raise qualification.QualificationFailure(
            "dual-T4 qualification runtime drifted from the qualified T4 FLA stack: "
            f"{mismatches}; full_runtime={snapshot}"
        )
    return snapshot


def _nvidia_smi() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"nvidia-smi unavailable: {error}"
    return result.stdout.strip()


def _gpu_headroom(snapshot: dict[str, Any]) -> dict[str, Any]:
    import torch

    memory: list[dict[str, int]] = []
    insufficient: list[dict[str, int]] = []
    for index in range(qualification.WORLD_SIZE):
        with torch.cuda.device(index):
            free_bytes, total_bytes = torch.cuda.mem_get_info()
        row = {
            "index": index,
            "free_bytes": int(free_bytes),
            "total_bytes": int(total_bytes),
        }
        memory.append(row)
        if free_bytes < MINIMUM_FREE_GPU_BYTES:
            insufficient.append(row)
    snapshot["memory_before_workers"] = memory
    gib = 1024**3
    _progress(
        "GPU free memory before workers: "
        + ", ".join(
            f"cuda:{row['index']}={row['free_bytes'] / gib:.2f}/{row['total_bytes'] / gib:.2f} GiB"
            for row in memory
        )
    )
    if insufficient:
        raise qualification.QualificationFailure(
            "dual-T4 qualification requires at least "
            f"{MINIMUM_FREE_GPU_BYTES / gib:.1f} GiB free on each T4 before launch. "
            "A previous interrupted CUDA process is likely still holding VRAM. "
            "Restart the Kaggle session, or inspect/terminate the stale process before retrying.\n"
            + _nvidia_smi()
        )
    return snapshot


def _terminate_process_group(process: subprocess.Popen[Any], label: str) -> None:
    if process.poll() is not None:
        return
    _progress(f"terminating {label} process group pid={process.pid}")
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait()


def _install_observability() -> None:
    original_load_blocks = qualification._load_blocks
    original_build_model = qualification._build_model
    original_run_one_update = qualification._run_one_update
    original_worker = qualification._worker
    original_require_two_t4s = qualification.require_two_t4s

    def require_two_t4s() -> dict[str, Any]:
        return _gpu_headroom(original_require_two_t4s())

    def load_blocks(dataset: Path, count: int) -> list[Any]:
        _progress(f"loading {count} real optimizer blocks from {dataset}")
        blocks = original_load_blocks(dataset, count)
        _progress(
            f"loaded {len(blocks)} blocks; ids={blocks[0].block_id}..{blocks[-1].block_id}"
            if blocks
            else "loaded 0 blocks"
        )
        return blocks

    def build_model(device: int):
        runtime = _require_expected_runtime()
        _progress(
            f"cuda:{device} runtime verified; torch={runtime['torch']} cuda={runtime['cuda']} "
            f"fla-core={runtime['fla_core']} triton={runtime['triton']}"
        )
        _progress(f"cuda:{device} building 20M model + hybrid Muon/AdamW state")
        result = original_build_model(device)
        import torch

        _progress(
            f"cuda:{device} model ready; gpu={torch.cuda.get_device_name(device)!r}"
        )
        return result

    def run_one_update(*args: Any, **kwargs: Any) -> dict[str, Any]:
        block = args[4] if len(args) > 4 else kwargs.get("block")
        rank = int(kwargs.get("rank", 0))
        distributed = bool(kwargs.get("distributed", False))
        mode = "ddp" if distributed else "single"
        block_id = getattr(block, "block_id", "?")
        _progress(f"{mode} rank={rank} block={block_id} forward/backward start")
        started = time.perf_counter()
        result = original_run_one_update(*args, **kwargs)
        elapsed = time.perf_counter() - started
        _progress(
            f"{mode} rank={rank} block={block_id} forward/backward done "
            f"in {elapsed:.2f}s loss={float(result['loss']):.6f} "
            f"grad_norm={float(result['gradient_norm']):.6f} "
            f"overflow_retries={int(result['overflow_retries'])}"
        )
        return result

    def worker(args: Any) -> int:
        _progress(f"worker={args.worker} process start pid={os.getpid()}")
        result = original_worker(args)
        _progress(f"worker={args.worker} state saved; process complete")
        return result

    def worker_command(args: Any, worker_kind: str, output: Path) -> list[str]:
        base = [
            str(Path(__file__).resolve()),
            "--dataset-dir",
            str(qualification.resolve_dataset(args.dataset_dir)),
            "--warmup-blocks",
            str(args.warmup_blocks),
            "--measure-blocks",
            str(args.measure_blocks),
            "--worker",
            worker_kind,
            "--worker-output",
            str(output),
        ]
        if worker_kind == "single":
            return [sys.executable, *base]
        return [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={qualification.WORLD_SIZE}",
            *base,
        ]

    def run_child(command: Sequence[str], *, visible_devices: str) -> None:
        worker_kind = _worker_name(command)
        timeout = _timeout_seconds()
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = visible_devices
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        _progress(
            f"starting {worker_kind} worker on CUDA_VISIBLE_DEVICES={visible_devices}; "
            f"watchdog={timeout}s"
        )
        process = subprocess.Popen(
            list(command),
            cwd=REPO,
            env=env,
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as error:
            _progress(f"{worker_kind} worker exceeded {timeout}s")
            _terminate_process_group(process, worker_kind)
            raise qualification.QualificationFailure(
                f"{worker_kind} worker exceeded {timeout}s. The qualification is hung; "
                "use the last progress line to identify whether the stall is model/JIT, "
                "a specific training block, state save, or DDP/NCCL."
            ) from error
        except BaseException:
            # Manual notebook interruption must not leave a detached CUDA worker
            # holding most of a T4, which would poison the next qualification.
            _terminate_process_group(process, worker_kind)
            raise
        if returncode:
            raise subprocess.CalledProcessError(returncode, command)
        _progress(f"{worker_kind} worker completed successfully")

    qualification.require_two_t4s = require_two_t4s
    qualification._load_blocks = load_blocks
    qualification._build_model = build_model
    qualification._run_one_update = run_one_update
    qualification._worker = worker
    qualification._worker_command = worker_command
    qualification._run_child = run_child


def main(argv: Sequence[str] | None = None) -> int:
    _install_observability()
    _progress(
        "qualification watchdog active; each single/DDP worker has a "
        f"{_timeout_seconds()}s limit"
    )
    return int(qualification.main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
