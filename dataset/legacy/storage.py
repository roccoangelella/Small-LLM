"""Atomic JSON and JSONL artifact I/O."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator


def ensure_parent(path: Path) -> None:
    """Create a file's parent directory if necessary."""

    path.parent.mkdir(parents=True, exist_ok=True)


def write_json_atomic(path: Path, payload: Any) -> None:
    """Atomically write JSON so checkpoints never become partially valid."""

    ensure_parent(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def read_json(path: Path) -> Any:
    """Read a UTF-8 JSON artifact with a useful missing-file error."""

    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Required artifact does not exist: {path}") from error


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write a complete UTF-8 JSONL artifact."""

    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield non-empty JSONL rows."""

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL in {path}:{line_number}") from error
