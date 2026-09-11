"""Portable Triton/FLA cache seed for provider lanes (Modal, Beam).

Execution optimisation only, never a scientific dependency: a missing, stale or
incompatible seed means ordinary Triton JIT + autotuning, not a failed run.
Generalises the Kaggle-only lifecycle of ``kaggle/src/triton_cache.py`` (ADR 0102)
to a seed archive kept on the provider cache volume and extracted to a fixed local
path before the training child starts.  The Kaggle tool is untouched.

Layout on the cache volume::

    <seed_root>/<cache_id>/small_llm_triton_seed_manifest.json
    <seed_root>/<cache_id>/triton-cache-<sha256>.tar

The cache id encodes GPU compute capability, Python/torch/CUDA/Triton/FLA versions,
model geometry, precision and microbatch, plus a SHA-256 of the kernel-facing
source files.  Triton metadata stores absolute paths, so the local cache directory
is a fixed canonical path recorded in the manifest and required to match.
"""
from __future__ import annotations

import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

SCHEMA_VERSION = 2
MANIFEST_NAME = "small_llm_triton_seed_manifest.json"
ARCHIVE_NAME = "triton-cache.tar"
LOCAL_ROOT = Path("/tmp/small-llm-triton")
STRICT_ENV = "SMALL_LLM_TRITON_SEED_STRICT"
DISABLE_ENV = "SMALL_LLM_TRITON_SEED_DISABLE"
KERNEL_CONTRACT_FILES = (
    "model/config.py",
    "model/components.py",
    "model/gdn2.py",
    "model/gdn2_stable.py",
    "model/gdn2_fla.py",
    "model/model.py",
    "trainer/precision.py",
    "MOE_model/config.py",
    "MOE_model/model.py",
    "MOE_model/router.py",
)
REPO = Path(__file__).resolve().parents[1]


class TritonSeedError(RuntimeError):
    pass


# --- contract ---------------------------------------------------------------

def kernel_contract_sha256(repo: Path = REPO) -> str:
    digest = hashlib.sha256()
    for relative in KERNEL_CONTRACT_FILES:
        path = repo / relative
        if not path.is_file():
            raise TritonSeedError(f"kernel contract file is missing: {relative}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _runtime_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def _gpu_query() -> tuple[str, tuple[int, int]] | None:
    """Return (name, compute capability) of GPU 0 via nvidia-smi, without creating a CUDA context."""

    if shutil.which("nvidia-smi") is None:
        return None
    try:
        output = subprocess.run(
            ["nvidia-smi", "--id=0", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
            check=True, capture_output=True, text=True, timeout=30,
        ).stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        return None
    if not output:
        return None
    name, _, capability = output[0].rpartition(",")
    major, _, minor = capability.strip().partition(".")
    try:
        return name.strip(), (int(major), int(minor))
    except ValueError:
        return None


def _cuda_wheel_version() -> str:
    """The CUDA flavour of the installed torch wheel (e.g. '12.8' from '2.10.0+cu128'), not the driver."""

    version = _runtime_version("torch")
    local = version.partition("+")[2]
    if local.startswith("cu") and local[2:].isdigit() and len(local) >= 5:
        digits = local[2:]
        return f"{digits[:-1]}.{digits[-1]}"
    return local or "unknown"


def live_contract(*, model: str, architecture: str, precision: str, microbatch_size: int,
                  context_length: int, gdn_chunk_size: int, repo: Path = REPO) -> dict[str, object] | None:
    """Return the seed contract for GPU 0, or None when no GPU can be queried.

    Uses nvidia-smi and package metadata only, so the pilot parent never imports
    torch nor holds a CUDA context next to the training child.
    """

    queried = _gpu_query()
    if queried is None:
        return None
    name, capability = queried
    return {
        "gpu_name": name,
        "compute_capability": list(capability),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "torch": _runtime_version("torch"),
        "host_abi": [platform.machine(), sys.implementation.cache_tag, *platform.libc_ver()],
        "cuda": _cuda_wheel_version(),
        "triton": _runtime_version("triton"),
        "fla_core": _runtime_version("fla-core"),
        "model": model,
        "architecture": architecture,
        "precision": precision,
        "microbatch_size": int(microbatch_size),
        "context_length": int(context_length),
        "gdn_chunk_size": int(gdn_chunk_size),
        "kernel_contract_sha256": kernel_contract_sha256(repo),
    }


def cache_id(contract: Mapping[str, object]) -> str:
    capability = contract["compute_capability"]
    assert isinstance(capability, (list, tuple)) and len(capability) == 2
    short = hashlib.sha256(json.dumps(dict(contract), sort_keys=True).encode()).hexdigest()[:16]
    return (
        f"sm{capability[0]}{capability[1]}-py{contract['python']}-torch{contract['torch']}"
        f"-cu{contract['cuda']}-triton{contract['triton']}-fla{contract['fla_core']}"
        f"-{contract['model']}-{contract['architecture']}-{contract['precision']}"
        f"-mb{contract['microbatch_size']}-ctx{contract['context_length']}"
        f"-chunk{contract['gdn_chunk_size']}-k{short}-v{SCHEMA_VERSION}"
    ).replace("/", "_").replace(" ", "")


def local_cache_dir(contract: Mapping[str, object], local_root: Path = LOCAL_ROOT) -> Path:
    return local_root / cache_id(contract)


# --- hashing helpers --------------------------------------------------------

def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as error:
        raise TritonSeedError(f"cannot read seed JSON {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise TritonSeedError(f"seed JSON is not an object: {path}")
    return value


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix="." + path.name, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _tree_sha256(files: Sequence[Mapping[str, object]]) -> str:
    digest = hashlib.sha256()
    for row in files:
        for field in ("path", "sha256", "size"):
            digest.update(str(row[field]).encode("utf-8"))
            digest.update(b"\0")
    return digest.hexdigest()


def _cache_file_rows(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == MANIFEST_NAME:
            continue
        if any(part.startswith("tmp.pid_") or part == "lock" for part in path.relative_to(root).parts):
            continue  # Triton in-flight scratch, never part of a seed
        relative = path.relative_to(root).as_posix()
        rows.append({"path": relative, "size": int(path.stat().st_size), "sha256": _sha256_path(path)})
    return rows


def _safe_member_path(name: str) -> Path:
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise TritonSeedError(f"unsafe seed archive member: {name!r}")
    return path


# --- package / validate / extract ------------------------------------------

def package(*, cache_dir: Path, seed_dir: Path, contract: Mapping[str, object]) -> Mapping[str, object]:
    """Archive a populated local cache directory into ``seed_dir`` atomically."""

    cache_dir = cache_dir.resolve()
    rows = _cache_file_rows(cache_dir)
    if not rows:
        raise TritonSeedError(f"local Triton cache is empty, nothing to seed: {cache_dir}")
    seed_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".seed-", dir=seed_dir.parent))
    try:
        archive = staging / ARCHIVE_NAME
        with tarfile.open(archive, "w") as handle:
            for row in rows:
                relative = Path(str(row["path"]))
                handle.add(cache_dir / relative, arcname=relative.as_posix(), recursive=False)
        archive_digest = _sha256_path(archive)
        archive_name = f"triton-cache-{archive_digest}.tar"
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "cache_id": cache_id(contract),
            "canonical_cache_dir": str(cache_dir),
            "contract": dict(contract),
            "archive": {"name": archive_name, "size": int(archive.stat().st_size), "sha256": archive_digest},
            "files": rows,
            "tree_sha256": _tree_sha256(rows),
        }
        # Immutable archive first, then one atomic manifest pointer. Concurrent readers
        # see either complete generation; writers never remove another writer's bytes.
        seed_dir.mkdir(parents=True, exist_ok=True)
        os.replace(archive, seed_dir / archive_name)
        _write_json(seed_dir / MANIFEST_NAME, manifest)
        shutil.rmtree(staging)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def validate(seed_dir: Path, *, contract: Mapping[str, object], destination: Path) -> tuple[Mapping[str, Any], Path]:
    manifest = _read_json(seed_dir / MANIFEST_NAME)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise TritonSeedError("seed manifest schema is incompatible")
    if manifest.get("cache_id") != cache_id(contract):
        raise TritonSeedError("seed cache id differs from the live contract")
    if manifest.get("contract") != dict(contract):
        raise TritonSeedError("seed runtime/geometry/kernel contract differs from the live one")
    if manifest.get("canonical_cache_dir") != str(destination):
        raise TritonSeedError("seed canonical path differs; Triton metadata is path-sensitive")
    archive_info = manifest.get("archive")
    if not isinstance(archive_info, Mapping) or archive_info.get("name") != f"triton-cache-{archive_info.get('sha256')}.tar":
        raise TritonSeedError("seed manifest lacks archive metadata")
    archive_name = str(archive_info["name"])
    _safe_member_path(archive_name)
    archive = seed_dir / archive_name
    if not archive.is_file():
        raise TritonSeedError(f"seed archive is missing: {archive}")
    if int(archive_info.get("size", -1)) != archive.stat().st_size:
        raise TritonSeedError("seed archive size mismatch")
    if archive_info.get("sha256") != _sha256_path(archive):
        raise TritonSeedError("seed archive checksum mismatch")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise TritonSeedError("seed manifest has no files")
    if manifest.get("tree_sha256") != _tree_sha256(files):
        raise TritonSeedError("seed manifest tree hash mismatch")
    return manifest, archive


def extract_verified(seed_dir: Path, *, contract: Mapping[str, object], destination: Path) -> None:
    manifest, archive = validate(seed_dir, contract=contract, destination=destination)
    expected: dict[str, Mapping[str, Any]] = {}
    for item in manifest["files"]:
        relative = _safe_member_path(str(item.get("path", ""))).as_posix()
        if relative in expected:
            raise TritonSeedError(f"duplicate seed manifest path: {relative}")
        expected[relative] = item
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
                    raise TritonSeedError(f"seed archive contains unsupported member type: {member.name}")
                relative = _safe_member_path(member.name).as_posix()
                if relative in seen:
                    raise TritonSeedError(f"duplicate seed archive member: {relative}")
                if relative not in expected:
                    raise TritonSeedError(f"seed archive contains unmanifested file: {relative}")
                target = staging / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                source = handle.extractfile(member)
                if source is None:
                    raise TritonSeedError(f"cannot extract seed member: {relative}")
                with source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
                seen.add(relative)
        if seen != set(expected):
            raise TritonSeedError(f"seed archive is missing manifest files: {sorted(set(expected) - seen)}")
        for relative, row in expected.items():
            path = staging / relative
            if path.stat().st_size != int(row.get("size", -1)) or _sha256_path(path) != row.get("sha256"):
                raise TritonSeedError(f"seed file mismatch after extract: {relative}")
        _write_json(staging / MANIFEST_NAME, manifest)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(destination, ignore_errors=True)
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


# --- lifecycle --------------------------------------------------------------

@contextmanager
def _lock(root: Path) -> Iterator[None]:
    root.parent.mkdir(parents=True, exist_ok=True)
    with (root.parent / f".{root.name}.lock").open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _local_seed_valid(root: Path, contract: Mapping[str, object]) -> bool:
    try:
        manifest = _read_json(root / MANIFEST_NAME)
    except TritonSeedError:
        return False
    try:
        if (manifest.get("schema_version") != SCHEMA_VERSION
                or manifest.get("cache_id") != cache_id(contract)
                or manifest.get("contract") != dict(contract)
                or manifest.get("canonical_cache_dir") != str(root)):
            return False
        rows = manifest["files"]
        if not isinstance(rows, list) or not rows or manifest.get("tree_sha256") != _tree_sha256(rows):
            return False
        for row in rows:
            path = root / _safe_member_path(row["path"])
            if path.is_symlink() or path.stat().st_size != row["size"] or _sha256_path(path) != row["sha256"]:
                return False
        return True
    except (OSError, ValueError, TypeError, KeyError, TritonSeedError):
        return False


def prepare(*, contract: Mapping[str, object] | None, seed_root: Path | None,
            local_root: Path = LOCAL_ROOT, strict: bool | None = None) -> dict[str, object]:
    """Return the environment for the training child and the seed status.

    Bad or absent seeds fall back unless ``strict``; filesystem access errors
    are handled by the pilot boundary. The fallback is
    ordinary Triton JIT at the same canonical local path, so a later
    :func:`harvest` can publish what this run compiled.
    """

    is_strict = strict if strict is not None else os.environ.get(STRICT_ENV, "").strip().lower() in {"1", "true", "yes"}
    if contract is None:
        if is_strict:
            raise TritonSeedError("strict seed qualification requires a CUDA runtime contract")
        return {"status": "disabled", "reason": "no CUDA device or torch unavailable", "env": {}}
    if os.environ.get(DISABLE_ENV, "").strip().lower() in {"1", "true", "yes"}:
        if is_strict:
            raise TritonSeedError("strict seed qualification cannot be disabled")
        return {"status": "disabled", "reason": f"{DISABLE_ENV} set", "env": {}}
    root = local_cache_dir(contract, local_root)
    env = {"TRITON_CACHE_DIR": str(root), "TRITON_CACHE_AUTOTUNING": "1", "FLA_CACHE_RESULTS": "1"}
    seed_dir = None if seed_root is None else seed_root / cache_id(contract)
    with _lock(root):
        if _local_seed_valid(root, contract):
            return {"status": "local_seed", "cache_dir": str(root), "env": env, "seed_dir": str(seed_dir)}
        if (root / MANIFEST_NAME).exists():
            shutil.rmtree(root)  # Never retain known-invalid local binaries for the JIT fallback.
        rejection = None
        if seed_dir is not None and (seed_dir / MANIFEST_NAME).is_file():
            try:
                extract_verified(seed_dir, contract=contract, destination=root)
                return {"status": "seeded", "cache_dir": str(root), "env": env, "seed_dir": str(seed_dir)}
            except (TritonSeedError, OSError, ValueError, TypeError, KeyError, tarfile.TarError) as error:
                rejection = str(error)
                if is_strict:
                    raise TritonSeedError(str(error)) from error
        elif is_strict:
            raise TritonSeedError(f"no Triton seed at {seed_dir} and strict mode is on")
        root.mkdir(parents=True, exist_ok=True)
    return {"status": "jit_fallback", "cache_dir": str(root), "env": env,
            "seed_dir": None if seed_dir is None else str(seed_dir), "rejection": rejection}


def harvest(*, contract: Mapping[str, object], seed_root: Path, local_root: Path = LOCAL_ROOT,
            overwrite: bool = False) -> dict[str, object]:
    """Publish the local cache as the seed for this contract if none exists yet."""

    root = local_cache_dir(contract, local_root)
    seed_dir = seed_root / cache_id(contract)
    if (seed_dir / MANIFEST_NAME).is_file() and not overwrite:
        try:
            validate(seed_dir, contract=contract, destination=root)
            return {"status": "exists", "seed_dir": str(seed_dir)}
        except (TritonSeedError, OSError, ValueError, TypeError, KeyError, tarfile.TarError):
            pass  # Rebuild a rejected seed from the successful JIT run.
    if not root.is_dir():
        return {"status": "no_local_cache", "cache_dir": str(root)}
    with _lock(root):
        manifest = package(cache_dir=root, seed_dir=seed_dir, contract=contract)
        _write_json(root / MANIFEST_NAME, manifest)
    return {"status": "published", "seed_dir": str(seed_dir),
            "files": len(manifest["files"]), "archive_bytes": manifest["archive"]["size"]}


__all__ = ["ARCHIVE_NAME", "KERNEL_CONTRACT_FILES", "LOCAL_ROOT", "MANIFEST_NAME", "TritonSeedError",
           "cache_id", "extract_verified", "harvest", "kernel_contract_sha256", "live_contract",
           "local_cache_dir", "package", "prepare", "validate"]
