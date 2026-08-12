"""Filesystem safety, locking, disk preflight, and interrupted-finalize recovery."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Mapping

from dataset import config
from dataset.src.storage import read_json, write_json_atomic

RUN_LOCK_SUFFIX = ".production.lock"
PROGRESS_BACKUP_FILENAME = "progress.production.safe.json"


class RunLock:
    def __init__(self, output_dir: Path) -> None:
        self.path = output_dir.parent / f".{output_dir.name}{RUN_LOCK_SUFFIX}"
        self._handle = None

    def __enter__(self) -> "RunLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a+", encoding="utf-8")
        try:
            import fcntl
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except ImportError as error:  # pragma: no cover - production target is Linux
            self._handle.close()
            self._handle = None
            raise RuntimeError("production dataset locking requires a POSIX host") from error
        except BlockingIOError as error:
            self._handle.close()
            self._handle = None
            raise RuntimeError(f"another dataset process holds {self.path}") from error
        self._handle.seek(0)
        self._handle.truncate()
        self._handle.write(json.dumps({"pid": os.getpid()}) + "\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._handle is None:
            return
        try:
            import fcntl
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle.close()
            self._handle = None


def unlink_durable(path: Path) -> None:
    if not path.exists():
        return
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"refusing to remove unsafe production artifact: {path}")
    path.unlink()
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def recover_progress_backup(output_dir: Path) -> None:
    backup_path = output_dir / PROGRESS_BACKUP_FILENAME
    if not backup_path.exists():
        return
    backup = read_json(backup_path)
    if not isinstance(backup, Mapping) or not isinstance(backup.get("production"), Mapping):
        raise RuntimeError("production progress backup is invalid")
    progress_path = output_dir / config.PROGRESS_FILENAME
    try:
        current = read_json(progress_path) if progress_path.exists() else None
    except (OSError, ValueError, TypeError):
        current = None
    committed = (
        isinstance(current, Mapping)
        and isinstance(current.get("production"), Mapping)
        and current.get("complete") is True
    )
    if not committed:
        write_json_atomic(progress_path, dict(backup))
    unlink_durable(backup_path)


def discard_uncheckpointed_artifacts(output_dir: Path, state: Mapping[str, object]) -> None:
    raw_shards = state.get("finalized_shards", [])
    if not isinstance(raw_shards, list):
        raise ValueError("production state has an invalid finalized_shards list")
    expected: set[str] = set()
    for item in raw_shards:
        if not isinstance(item, Mapping) or not isinstance(item.get("filename"), str):
            raise ValueError("production state has invalid finalized shard metadata")
        expected.add(str(item["filename"]))

    for split in ("train", "validation"):
        directory = output_dir / split
        if not directory.exists():
            continue
        for path in directory.glob(f".{split}-*.bin.tmp"):
            unlink_durable(path)
        for path in directory.glob(f"{split}-*.bin"):
            if path.relative_to(output_dir).as_posix() not in expected:
                unlink_durable(path)
    if state.get("complete") is not True:
        unlink_durable(output_dir / config.MANIFEST_FILENAME)


def required_free_bytes(maximum_source_tokens: int) -> int:
    return int(
        maximum_source_tokens
        * 2
        * (1.0 + config.DISK_EOD_OVERHEAD_FRACTION)
        * config.DISK_SAFETY_MULTIPLIER
    )


def required_remote_shard_free_bytes(
    target_shard_bytes: int,
    *,
    resident_shards: int = 3,
) -> int:
    """Bound local disk for upload-verified-and-evicted shard production.

    ``resident_shards=3`` deliberately budgets an active writer, a just-finalized
    upload/readback candidate, and one additional tail/checkpoint margin.  It is
    independent of the total corpus horizon, so a 10B dataset never requires a
    20+ GiB local preflight merely because its canonical copy is remote.
    """

    if isinstance(target_shard_bytes, bool) or target_shard_bytes <= 0:
        raise ValueError("target_shard_bytes must be positive")
    if isinstance(resident_shards, bool) or resident_shards <= 0:
        raise ValueError("resident_shards must be positive")
    return int(target_shard_bytes * resident_shards * config.DISK_SAFETY_MULTIPLIER)


def preflight_disk(output_dir: Path, maximum_source_tokens: int, *, allow_unsafe: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if allow_unsafe:
        return
    free = shutil.disk_usage(output_dir).free
    required = required_free_bytes(maximum_source_tokens)
    if free < required:
        raise RuntimeError(
            "insufficient free disk for production cache: "
            f"free={free}, required={required}; use --allow-unsafe-low-disk only for bounded tests"
        )


def preflight_remote_shard_disk(
    output_dir: Path,
    target_shard_bytes: int,
    *,
    allow_unsafe: bool,
    resident_shards: int = 3,
) -> None:
    """Preflight only the bounded rolling footprint for remotely durable shards."""

    output_dir.mkdir(parents=True, exist_ok=True)
    if allow_unsafe:
        return
    free = shutil.disk_usage(output_dir).free
    required = required_remote_shard_free_bytes(
        target_shard_bytes,
        resident_shards=resident_shards,
    )
    if free < required:
        raise RuntimeError(
            "insufficient free disk for rolling remote-shard production: "
            f"free={free}, required={required}; use --allow-unsafe-low-disk only for bounded tests"
        )
