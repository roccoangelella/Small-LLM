"""Atomic JSON artefact I/O for the production pipeline.

All durable state (``progress.json``, ``work_plan.json``, ``manifest.json``) is
written via a temporary file plus ``fsync`` plus atomic ``replace`` so a crash
can never leave a half-written checkpoint that could be mistaken for valid state.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def ensure_parent(path: Path) -> None:
    """Create the file's parent directory if necessary."""

    path.parent.mkdir(parents=True, exist_ok=True)


def write_json_atomic(path: Path, payload: Any, *, sort_keys: bool = True) -> None:
    """Atomically write JSON so a checkpoint is never partially valid on disk."""

    ensure_parent(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=sort_keys)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    # Persist the directory entry as well as the file contents.  Without this
    # fsync, a power loss can theoretically forget the rename even though the
    # temporary file itself was durable.
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def read_json(path: Path) -> Any:
    """Read a UTF-8 JSON artefact with an actionable missing-file error."""

    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Required artefact does not exist: {path}") from error


def sha256_file(path: Path, *, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    """Return the SHA-256 of a file without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(payload: Any, *, exclude_keys: tuple[str, ...] = ()) -> bytes:
    """Return a stable UTF-8 bytes encoding of a payload for hashing.

    ``exclude_keys`` lets callers hash a payload without the hash field it
    contains, so a work plan or manifest can self-verify with a fixed rule.
    """

    if exclude_keys:
        filtered = {k: v for k, v in payload.items() if k not in exclude_keys}
    else:
        filtered = payload
    return json.dumps(
        filtered, ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8") + b"\n"
