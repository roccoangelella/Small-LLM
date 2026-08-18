#!/usr/bin/env python3
"""Build, package, publish, and preseed the Kaggle Tesla-T4 Triton cache.

The production cache is deliberately an execution optimization, never a
scientific dependency.  A compatible attached Kaggle Dataset is copied into a
canonical writable path before torchrun.  Missing or stale caches fall back to
normal Triton JIT/autotuning.

Build once on a Kaggle 2xT4 session:

    python kaggle/triton_cache.py build

Optionally create/update a private Kaggle Dataset:

    python kaggle/triton_cache.py build --publish OWNER/DATASET-SLUG

Publish an already-packaged cache without rebuilding or repackaging it:

    python kaggle/triton_cache.py publish OWNER/DATASET-SLUG

Future notebooks only need that private dataset attached.  The canonical
``kaggle/launch.py deep-decay ...`` path discovers and validates it
automatically.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import fcntl
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO = Path(__file__).resolve().parents[1]
KAGGLE = REPO / "kaggle"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(KAGGLE) not in sys.path:
    sys.path.insert(0, str(KAGGLE))

import dual_t4_runtime

SCHEMA_VERSION = 1
CACHE_ID = "t4-sm75-py313-torch210-cu128-triton360-fla052-100m-gdn2-mb2-ctx2048-v1"
MANIFEST_NAME = "small_llm_triton_cache_manifest.json"
ARCHIVE_NAME = "triton-cache.tar"
BUILD_STAMP_NAME = ".small_llm_triton_cache_build_stamp.json"
DEFAULT_CACHE_ROOT = Path("/kaggle/working/small-llm/runtime-cache") / CACHE_ID
DEFAULT_PACKAGE_DIR = Path("/kaggle/working/small-llm-t4-triton-cache-dataset")
INPUT_ROOT = Path("/kaggle/input")
DATASET_DIR_ENV = "SMALL_LLM_KAGGLE_TRITON_CACHE_DATASET_DIR"
CACHE_DIR_ENV = "SMALL_LLM_KAGGLE_TRITON_CACHE_DIR"
STRICT_ENV = "SMALL_LLM_KAGGLE_TRITON_CACHE_STRICT"

MODEL_NAME = "100M"
ARCHITECTURE = "gdn2_hybrid"
PRECISION = "fp16"
MICROBATCH_SIZE = 2
CONTEXT_LENGTH = 2048
GDN_CHUNK_SIZE = 32
WORLD_SIZE = 2
PUBLISH_VERIFY_TIMEOUT_SECONDS = 1200.0
PUBLISH_VERIFY_POLL_SECONDS = 3.0
EXPECTED_GPU_NAME = "Tesla T4"
EXPECTED_COMPUTE_CAPABILITY = (7, 5)

KERNEL_CONTRACT_FILES = (
    "model/config.py",
    "model/components.py",
    "model/gdn2.py",
    "model/gdn2_fla.py",
    "model/model.py",
    "trainer/precision.py",
    "kaggle/dual_t4_train.py",
    "kaggle/dual_t4_train_block64.py",
)


class TritonCacheError(RuntimeError):
    pass


def canonical_cache_dir() -> Path:
    raw = os.environ.get(CACHE_DIR_ENV, "").strip()
    path = Path(raw).expanduser() if raw else DEFAULT_CACHE_ROOT
    return path.resolve()


def expected_contract() -> dict[str, object]:
    return {
        "gpu_name": EXPECTED_GPU_NAME,
        "compute_capability": list(EXPECTED_COMPUTE_CAPABILITY),
        "python": "3.13",
        "torch": dual_t4_runtime.TORCH_VERSION,
        "cuda": "12.8",
        "triton": dual_t4_runtime.TRITON_VERSION,
        "fla_core": dual_t4_runtime.FLA_VERSION,
        "model": MODEL_NAME,
        "architecture": ARCHITECTURE,
        "precision": PRECISION,
        "microbatch_size": MICROBATCH_SIZE,
        "context_length": CONTEXT_LENGTH,
        "gdn_chunk_size": GDN_CHUNK_SIZE,
        "world_size": WORLD_SIZE,
    }


def kernel_contract_sha256() -> str:
    digest = hashlib.sha256()
    for relative in KERNEL_CONTRACT_FILES:
        path = REPO / relative
        if not path.is_file():
            raise TritonCacheError(f"kernel contract file is missing: {relative}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as error:
        raise TritonCacheError(f"cannot read cache JSON {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise TritonCacheError(f"cache JSON is not an object: {path}")
    return value


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _tree_sha256(files: Sequence[Mapping[str, object]]) -> str:
    digest = hashlib.sha256()
    for row in files:
        digest.update(str(row["path"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(row["sha256"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(row["size"]).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _cache_file_rows(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == BUILD_STAMP_NAME:
            continue
        relative = path.relative_to(root).as_posix()
        rows.append(
            {
                "path": relative,
                "size": int(path.stat().st_size),
                "sha256": _sha256_path(path),
            }
        )
    if not rows:
        raise TritonCacheError(f"Triton cache is empty: {root}")
    return rows


def _runtime_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def _live_contract() -> dict[str, object]:
    import torch

    if torch.cuda.device_count() < WORLD_SIZE:
        raise TritonCacheError(
            f"cache build requires {WORLD_SIZE} visible GPUs; found {torch.cuda.device_count()}"
        )
    names = [torch.cuda.get_device_name(index) for index in range(WORLD_SIZE)]
    if any(name != EXPECTED_GPU_NAME for name in names):
        raise TritonCacheError(
            f"cache build requires two {EXPECTED_GPU_NAME} GPUs; found {names}"
        )
    capabilities = [
        tuple(int(item) for item in torch.cuda.get_device_capability(index))
        for index in range(WORLD_SIZE)
    ]
    if any(capability != EXPECTED_COMPUTE_CAPABILITY for capability in capabilities):
        raise TritonCacheError(
            "cache build requires compute capability "
            f"{EXPECTED_COMPUTE_CAPABILITY}; found {capabilities}"
        )
    return {
        "gpu_name": EXPECTED_GPU_NAME,
        "compute_capability": list(EXPECTED_COMPUTE_CAPABILITY),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "torch": torch.__version__.split("+", 1)[0],
        "cuda": str(torch.version.cuda),
        "triton": _runtime_version("triton"),
        "fla_core": _runtime_version("fla-core"),
        "model": MODEL_NAME,
        "architecture": ARCHITECTURE,
        "precision": PRECISION,
        "microbatch_size": MICROBATCH_SIZE,
        "context_length": CONTEXT_LENGTH,
        "gdn_chunk_size": GDN_CHUNK_SIZE,
        "world_size": WORLD_SIZE,
    }


def _require_live_contract() -> dict[str, object]:
    actual = _live_contract()
    expected = expected_contract()
    drift = {
        key: {"expected": expected[key], "actual": actual.get(key)}
        for key in expected
        if actual.get(key) != expected[key]
    }
    if drift:
        raise TritonCacheError(
            "cache build runtime drifted from the qualified T4 contract: "
            + json.dumps(drift, sort_keys=True)
        )
    return actual


def _run_exact_prewarm(*, phase: str) -> dict[str, object]:
    import torch
    from model.config import ModelConfig
    from model.initialization import initialize_model
    from model.model import SmallLLM
    from trainer.precision import autocast_context

    runtime = _require_live_contract()
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.manual_seed(17)
    config = ModelConfig.substantive(
        architecture=ARCHITECTURE,
        gdn_chunk_size=GDN_CHUNK_SIZE,
    )
    model = SmallLLM(config)
    initialize_model(model, "normal")
    model = model.to(device)
    inputs = torch.zeros(
        (MICROBATCH_SIZE, CONTEXT_LENGTH),
        dtype=torch.long,
        device=device,
    )
    print(
        f"[triton-cache] {phase}: exact {MODEL_NAME} {PRECISION} prewarm "
        f"on {EXPECTED_GPU_NAME} at {MICROBATCH_SIZE}x{CONTEXT_LENGTH}; "
        "no optimizer or training state is created",
        flush=True,
    )
    started = time.perf_counter()
    model.zero_grad(set_to_none=True)
    with autocast_context(PRECISION, device):
        logits = model(inputs)
        objective = logits[..., 0].float().mean()
    objective.backward()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    model.zero_grad(set_to_none=True)
    del objective, logits, inputs, model
    torch.cuda.empty_cache()
    print(
        f"[triton-cache] {phase}: prewarm complete in {elapsed:.2f}s",
        flush=True,
    )
    return runtime


def _worker(phase: str) -> int:
    root = canonical_cache_dir()
    root.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(root)
    os.environ["TRITON_CACHE_AUTOTUNING"] = "1"
    os.environ["TRITON_PRINT_AUTOTUNING"] = "1"
    os.environ["FLA_CACHE_RESULTS"] = "1"
    runtime = _run_exact_prewarm(phase=phase)
    stamp = {
        "schema_version": SCHEMA_VERSION,
        "cache_id": CACHE_ID,
        "phase": phase,
        "contract": runtime,
        "kernel_contract_sha256": kernel_contract_sha256(),
    }
    _write_json(root / BUILD_STAMP_NAME, stamp)
    return 0


def _uv_worker_command(phase: str, uv: str) -> list[str]:
    return [
        uv,
        "run",
        "--python",
        "3.13",
        "--no-project",
        *dual_t4_runtime.qualified_runtime_uv_args(),
        "python",
        str(Path(__file__).resolve()),
        "_worker",
        "--phase",
        phase,
    ]


def _require_build_stamp(root: Path) -> Mapping[str, Any]:
    stamp = _read_json(root / BUILD_STAMP_NAME)
    expected = expected_contract()
    if stamp.get("schema_version") != SCHEMA_VERSION:
        raise TritonCacheError("cache build stamp schema drifted")
    if stamp.get("cache_id") != CACHE_ID:
        raise TritonCacheError("cache build stamp ID drifted")
    if stamp.get("contract") != expected:
        raise TritonCacheError("cache build stamp runtime/geometry drifted")
    if stamp.get("kernel_contract_sha256") != kernel_contract_sha256():
        raise TritonCacheError("cache build stamp source contract drifted")
    return stamp


def package_cache(
    *,
    cache_root: Path | None = None,
    output_dir: Path | None = None,
) -> Path:
    root = (cache_root or canonical_cache_dir()).resolve()
    output = (output_dir or DEFAULT_PACKAGE_DIR).resolve()
    _require_build_stamp(root)
    rows = _cache_file_rows(root)

    shutil.rmtree(output, ignore_errors=True)
    output.mkdir(parents=True, exist_ok=False)
    archive = output / ARCHIVE_NAME
    with tarfile.open(archive, "w") as handle:
        for row in rows:
            relative = Path(str(row["path"]))
            handle.add(root / relative, arcname=relative.as_posix(), recursive=False)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "cache_id": CACHE_ID,
        "canonical_cache_dir": str(canonical_cache_dir()),
        "contract": expected_contract(),
        "kernel_contract_sha256": kernel_contract_sha256(),
        "archive": {
            "name": ARCHIVE_NAME,
            "size": int(archive.stat().st_size),
            "sha256": _sha256_path(archive),
        },
        "files": rows,
        "tree_sha256": _tree_sha256(rows),
    }
    _write_json(output / MANIFEST_NAME, manifest)
    print(
        f"[triton-cache] package ready: {output} "
        f"({len(rows)} cache files, {archive.stat().st_size / (1024**2):.1f} MiB)",
        flush=True,
    )
    return output


def _validate_manifest(package_dir: Path) -> tuple[Mapping[str, Any], Path]:
    manifest = _read_json(package_dir / MANIFEST_NAME)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise TritonCacheError("cache manifest schema is incompatible")
    if manifest.get("cache_id") != CACHE_ID:
        raise TritonCacheError("cache manifest ID is incompatible")
    if manifest.get("canonical_cache_dir") != str(canonical_cache_dir()):
        raise TritonCacheError(
            "cache manifest canonical path differs from this launcher; "
            "Triton group metadata is path-sensitive"
        )
    if manifest.get("contract") != expected_contract():
        raise TritonCacheError("cache manifest runtime/geometry contract is incompatible")
    if manifest.get("kernel_contract_sha256") != kernel_contract_sha256():
        raise TritonCacheError("cache manifest source/kernel contract is stale")

    archive_info = manifest.get("archive")
    if not isinstance(archive_info, Mapping):
        raise TritonCacheError("cache manifest lacks archive metadata")
    archive_name = archive_info.get("name")
    if archive_name != ARCHIVE_NAME:
        raise TritonCacheError("cache manifest archive name drifted")
    archive = package_dir / ARCHIVE_NAME
    if not archive.is_file():
        raise TritonCacheError(f"cache archive is missing: {archive}")
    if int(archive_info.get("size", -1)) != archive.stat().st_size:
        raise TritonCacheError("cache archive size mismatch")
    if archive_info.get("sha256") != _sha256_path(archive):
        raise TritonCacheError("cache archive checksum mismatch")

    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise TritonCacheError("cache manifest has no files")
    if manifest.get("tree_sha256") != _tree_sha256(files):
        raise TritonCacheError("cache manifest tree hash mismatch")
    return manifest, archive


def _safe_member_path(name: str) -> Path:
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise TritonCacheError(f"unsafe cache archive member: {name!r}")
    return path


def _extract_verified(
    *,
    package_dir: Path,
    destination: Path,
) -> None:
    manifest, archive = _validate_manifest(package_dir)
    files_value = manifest["files"]
    assert isinstance(files_value, list)
    expected_rows: dict[str, Mapping[str, Any]] = {}
    for item in files_value:
        if not isinstance(item, Mapping):
            raise TritonCacheError("cache manifest file row is not an object")
        relative = _safe_member_path(str(item.get("path", ""))).as_posix()
        if relative in expected_rows:
            raise TritonCacheError(f"duplicate cache manifest path: {relative}")
        expected_rows[relative] = item

    staging = destination.with_name("." + destination.name + ".seed")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=False)
    try:
        seen: set[str] = set()
        with tarfile.open(archive, "r") as handle:
            for member in handle.getmembers():
                if member.isdir():
                    continue
                if not member.isfile():
                    raise TritonCacheError(
                        f"cache archive contains unsupported member type: {member.name}"
                    )
                relative = _safe_member_path(member.name).as_posix()
                if relative not in expected_rows:
                    raise TritonCacheError(
                        f"cache archive contains unmanifested file: {relative}"
                    )
                target = staging / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                source = handle.extractfile(member)
                if source is None:
                    raise TritonCacheError(f"cannot extract cache member: {relative}")
                with source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
                seen.add(relative)

        if seen != set(expected_rows):
            missing = sorted(set(expected_rows) - seen)
            raise TritonCacheError(f"cache archive is missing manifest files: {missing}")

        for relative, row in expected_rows.items():
            path = staging / relative
            if path.stat().st_size != int(row.get("size", -1)):
                raise TritonCacheError(f"cache file size mismatch after extract: {relative}")
            if _sha256_path(path) != row.get("sha256"):
                raise TritonCacheError(f"cache file checksum mismatch after extract: {relative}")

        _write_json(staging / MANIFEST_NAME, manifest)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(destination, ignore_errors=True)
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _candidate_packages(explicit: Path | None = None) -> list[Path]:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if path.is_file():
            path = path.parent
        return [path]

    configured = os.environ.get(DATASET_DIR_ENV, "").strip()
    if configured:
        path = Path(configured).expanduser().resolve()
        if path.is_file():
            path = path.parent
        return [path]

    if not INPUT_ROOT.is_dir():
        return []
    return sorted(
        {
            manifest.parent.resolve()
            for manifest in INPUT_ROOT.rglob(MANIFEST_NAME)
            if manifest.is_file()
        }
    )


@contextlib.contextmanager
def _seed_lock(root: Path):
    root.parent.mkdir(parents=True, exist_ok=True)
    lock_path = root.parent / f".{CACHE_ID}.seed.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _local_seed_valid(root: Path) -> bool:
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        return False
    try:
        manifest = _read_json(manifest_path)
        return bool(
            manifest.get("schema_version") == SCHEMA_VERSION
            and manifest.get("cache_id") == CACHE_ID
            and manifest.get("contract") == expected_contract()
            and manifest.get("kernel_contract_sha256") == kernel_contract_sha256()
        )
    except TritonCacheError:
        return False


def prepare_environment(
    *,
    explicit_package: Path | None = None,
    strict: bool | None = None,
) -> dict[str, object]:
    root = canonical_cache_dir()
    os.environ["TRITON_CACHE_DIR"] = str(root)
    os.environ["TRITON_CACHE_AUTOTUNING"] = "1"
    os.environ["FLA_CACHE_RESULTS"] = "1"

    with _seed_lock(root):
        if _local_seed_valid(root):
            print(f"[triton-cache] reusing validated local seed at {root}", flush=True)
            return {"status": "local_seed", "cache_dir": str(root)}

        failures: list[str] = []
        for candidate in _candidate_packages(explicit_package):
            try:
                _extract_verified(package_dir=candidate, destination=root)
            except TritonCacheError as error:
                failures.append(f"{candidate}: {error}")
                continue
            print(
                f"[triton-cache] preseeded T4 cache from {candidate} -> {root}",
                flush=True,
            )
            return {
                "status": "seeded",
                "cache_dir": str(root),
                "package_dir": str(candidate),
            }

        root.mkdir(parents=True, exist_ok=True)
        is_strict = (
            strict
            if strict is not None
            else os.environ.get(STRICT_ENV, "").strip().lower() in {"1", "true", "yes"}
        )
        if is_strict and failures:
            raise TritonCacheError(
                "no compatible Triton cache seed was accepted: " + "; ".join(failures)
            )
        if failures:
            print(
                "[triton-cache] attached cache seed(s) rejected; using normal JIT fallback: "
                + "; ".join(failures),
                flush=True,
            )
        else:
            print(
                f"[triton-cache] no attached {CACHE_ID} dataset found; "
                f"using normal JIT fallback at {root}",
                flush=True,
            )
        return {
            "status": "jit_fallback",
            "cache_dir": str(root),
            "rejections": failures,
        }


def _write_dataset_metadata(package_dir: Path, handle: str) -> None:
    if "/" not in handle or handle.startswith("/") or handle.endswith("/"):
        raise TritonCacheError(
            "--publish must use Kaggle OWNER/DATASET-SLUG form"
        )
    metadata = {
        "title": "Small-LLM T4 Triton Cache",
        "id": handle,
        "licenses": [{"name": "other"}],
        "description": (
            "Private generated execution cache for Small-LLM Kaggle Tesla T4 runs. "
            "Contains Triton-generated artifacts for the pinned runtime/geometry "
            "recorded in small_llm_triton_cache_manifest.json; not model weights or data."
        ),
    }
    _write_json(package_dir / "dataset-metadata.json", metadata)


def _completed_output(result: subprocess.CompletedProcess[str]) -> str:
    return "\n".join(
        part.strip()
        for part in (result.stdout or "", result.stderr or "")
        if part and part.strip()
    )


def _run_kaggle_capture(
    command: Sequence[str],
    *,
    cwd: Path,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )


def _remote_dataset_files(
    kaggle: str,
    *,
    package_dir: Path,
    handle: str,
) -> set[str]:
    result = _run_kaggle_capture(
        [
            kaggle,
            "datasets",
            "files",
            handle,
            "--page-size",
            "200",
            "-v",
        ],
        cwd=package_dir,
    )
    if result.returncode != 0:
        detail = _completed_output(result) or f"exit status {result.returncode}"
        raise TritonCacheError(
            f"Kaggle Dataset file verification failed for {handle}: {detail}"
        )

    lines = [
        line
        for line in (result.stdout or "").splitlines()
        if not line.startswith("Next Page Token = ")
    ]
    try:
        rows = list(csv.DictReader(io.StringIO("\n".join(lines))))
    except (csv.Error, TypeError) as error:
        raise TritonCacheError(
            f"cannot parse Kaggle Dataset file listing for {handle}: {error}"
        ) from error
    names = {str(row.get("name", "")).strip() for row in rows}
    names.discard("")
    if not names:
        raise TritonCacheError(
            f"Kaggle Dataset file verification returned no files for {handle}"
        )
    return names


def _verify_published_dataset(
    kaggle: str,
    *,
    package_dir: Path,
    handle: str,
) -> None:
    expected = {ARCHIVE_NAME, MANIFEST_NAME}
    deadline = time.monotonic() + PUBLISH_VERIFY_TIMEOUT_SECONDS
    last_detail = "remote Dataset is not yet resolvable"
    while True:
        status = _run_kaggle_capture(
            [kaggle, "datasets", "status", handle],
            cwd=package_dir,
        )
        if status.returncode == 0:
            try:
                actual = _remote_dataset_files(
                    kaggle,
                    package_dir=package_dir,
                    handle=handle,
                )
            except TritonCacheError as error:
                last_detail = str(error)
            else:
                if actual == expected:
                    return
                last_detail = (
                    "Kaggle Dataset file verification mismatch for "
                    f"{handle}: expected {sorted(expected)}, found {sorted(actual)}"
                )
        else:
            last_detail = _completed_output(status) or f"exit status {status.returncode}"

        if time.monotonic() >= deadline:
            raise TritonCacheError(
                f"Kaggle Dataset publication was not confirmed for {handle}: "
                f"{last_detail}"
            )
        time.sleep(PUBLISH_VERIFY_POLL_SECONDS)


def publish_package(package_dir: Path, handle: str) -> None:
    kaggle = shutil.which("kaggle")
    if not kaggle:
        raise TritonCacheError("Kaggle CLI is required for --publish")
    _validate_manifest(package_dir)
    _write_dataset_metadata(package_dir, handle)
    status = _run_kaggle_capture(
        [kaggle, "datasets", "status", handle],
        cwd=package_dir,
    )
    if status.returncode == 0:
        command = [
            kaggle,
            "datasets",
            "version",
            "-p",
            str(package_dir),
            "-m",
            f"refresh {CACHE_ID}",
            "-r",
            "skip",
        ]
        action = "version"
    else:
        command = [
            kaggle,
            "datasets",
            "create",
            "-p",
            str(package_dir),
            "-r",
            "skip",
        ]
        action = "create"
    print(f"[triton-cache] Kaggle Dataset {action}: {handle}", flush=True)
    upload = _run_kaggle_capture(command, cwd=package_dir)
    output = _completed_output(upload)
    if output:
        print(output, flush=True)
    if upload.returncode != 0:
        raise TritonCacheError(
            f"Kaggle Dataset {action} command failed for {handle}: "
            + (output or f"exit status {upload.returncode}")
        )
    if "Dataset creation error:" in output or "Dataset version creation error:" in output:
        raise TritonCacheError(
            f"Kaggle Dataset {action} was rejected for {handle}: {output}"
        )

    _verify_published_dataset(
        kaggle,
        package_dir=package_dir,
        handle=handle,
    )
    print(
        f"[triton-cache] published private dataset {handle}; verified remote files "
        f"{ARCHIVE_NAME} + {MANIFEST_NAME}; attach it to future Kaggle notebooks "
        "so launch.py can preseed automatically",
        flush=True,
    )


def build_cache(*, output_dir: Path, publish: str | None) -> Path:
    uv = shutil.which("uv")
    if not uv:
        raise TritonCacheError("uv is required to build the qualified T4 cache")
    root = canonical_cache_dir()
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=False)

    env = dict(os.environ)
    env["TRITON_CACHE_DIR"] = str(root)
    env["TRITON_CACHE_AUTOTUNING"] = "1"
    env["TRITON_PRINT_AUTOTUNING"] = "1"
    env["FLA_CACHE_RESULTS"] = "1"

    print(
        "[triton-cache] compile pass: full Triton autotuning under the pinned "
        "Kaggle T4 runtime",
        flush=True,
    )
    subprocess.check_call(_uv_worker_command("compile", uv), cwd=REPO, env=env)

    print(
        "[triton-cache] fresh-process validation pass: reusing the same canonical "
        "disk cache",
        flush=True,
    )
    subprocess.check_call(_uv_worker_command("validate", uv), cwd=REPO, env=env)

    package = package_cache(cache_root=root, output_dir=output_dir)
    if publish:
        publish_package(package, publish)
    return package


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)

    build = subparsers.add_parser(
        "build",
        help="compile + fresh-process validate + package the exact 100M/T4 cache",
    )
    build.add_argument("--output-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    build.add_argument(
        "--publish",
        metavar="OWNER/DATASET-SLUG",
        help="create or version a private Kaggle Dataset after packaging",
    )

    seed = subparsers.add_parser(
        "seed",
        help="validate and install an attached/package cache into the canonical path",
    )
    seed.add_argument("--package-dir", type=Path)
    seed.add_argument("--strict", action="store_true")

    package = subparsers.add_parser(
        "package",
        help="package an already-built canonical cache",
    )
    package.add_argument("--output-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    package.add_argument("--publish", metavar="OWNER/DATASET-SLUG")

    publish = subparsers.add_parser(
        "publish",
        help="publish an existing opaque cache archive + manifest without rebuilding",
    )
    publish.add_argument("handle", metavar="OWNER/DATASET-SLUG")
    publish.add_argument(
        "--package-dir",
        type=Path,
        default=DEFAULT_PACKAGE_DIR,
        help="existing package directory containing the opaque archive + manifest",
    )

    worker = subparsers.add_parser("_worker", help=argparse.SUPPRESS)
    worker.add_argument("--phase", choices=("compile", "validate"), required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.action == "_worker":
        return _worker(args.phase)
    if args.action == "seed":
        result = prepare_environment(
            explicit_package=args.package_dir,
            strict=args.strict,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.action == "package":
        output = package_cache(output_dir=args.output_dir)
        if args.publish:
            publish_package(output, args.publish)
        return 0
    if args.action == "publish":
        publish_package(args.package_dir.expanduser().resolve(), args.handle)
        return 0
    if args.action == "build":
        build_cache(output_dir=args.output_dir, publish=args.publish)
        return 0
    raise AssertionError(f"unhandled action: {args.action}")


if __name__ == "__main__":
    raise SystemExit(main())
