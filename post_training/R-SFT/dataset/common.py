"""Shared contracts for R-SFT reasoning-dataset production."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

CONTEXT_LENGTH = 2_048
CANONICAL_FIELDS = ("skill", "difficulty", "problem", "reasoning", "answer")
RESERVED_MARKERS = ("<think>", "</think>", "<answer>")
FIT_SCHEMA = "small-llm-rsft-fit-v1"
OVER_CONTEXT_SCHEMA = "small-llm-rsft-overcontext-v1"
SOURCE_MANIFEST_SCHEMA = "small-llm-rsft-source-manifest-v1"


def sha256_path(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalized_prompt_hash(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("prompt must be non-empty text")
    normalized = " ".join(text.split()).casefold().encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


def read_jsonl(path: Path | str) -> Iterator[dict[str, Any]]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise RuntimeError(f"blank JSONL row at {source}:{line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"invalid JSON at {source}:{line_number}") from error
            if not isinstance(row, dict):
                raise RuntimeError(f"JSONL row must be an object at {source}:{line_number}")
            yield row


def atomic_json(path: Path | str, payload: Mapping[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def write_jsonl(path: Path | str, rows: Sequence[Mapping[str, Any]]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    temporary.replace(destination)
    return destination


def default_token_counter() -> Callable[[str], int]:
    try:
        import tiktoken
    except ImportError as error:  # pragma: no cover - environment dependency
        raise RuntimeError(
            "tiktoken is required; install the project post-training extra"
        ) from error
    encoding = tiktoken.get_encoding("gpt2")

    def count(text: str) -> int:
        return len(encoding.encode(text, allowed_special=set(), disallowed_special=()))

    return count


def atomic_rsft_serialized_tokens(
    *,
    problem: str,
    reasoning: str,
    answer: str,
    token_counter: Callable[[str], int] | None = None,
) -> int:
    for name, value in (
        ("problem", problem),
        ("reasoning", reasoning),
        ("answer", answer),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be non-empty text")
    count = token_counter or default_token_counter()
    return (
        count("User:\n")
        + count(problem)
        + count("\n\nAssistant:\n")
        + 3
        + count(reasoning)
        + count(answer)
        + 1
    )


def assistant_target_tokens(
    *,
    reasoning: str,
    answer: str,
    token_counter: Callable[[str], int] | None = None,
) -> int:
    count = token_counter or default_token_counter()
    return 3 + count(reasoning) + count(answer) + 1


def canonical_rsft_record(row: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for field in CANONICAL_FIELDS:
        value = row.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"R-SFT row field {field!r} must be non-empty text")
        result[field] = value.strip()
    return result


def validate_reasoning_text(row: Mapping[str, Any]) -> None:
    for field in ("problem", "reasoning", "answer"):
        value = row.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"candidate field {field!r} must be non-empty text")
        for marker in RESERVED_MARKERS:
            if marker in value:
                raise ValueError(f"candidate field {field!r} contains reserved marker {marker}")


def stable_conversation_id(row: Mapping[str, Any]) -> str:
    record = canonical_rsft_record(row)
    payload = "\x1f".join(record[field] for field in CANONICAL_FIELDS).encode("utf-8")
    return f"rsft-{hashlib.sha256(payload).hexdigest()[:20]}"


def stable_key(seed: int, label: str, identity: str) -> bytes:
    return hashlib.sha256(f"{seed}:{label}:{identity}".encode("utf-8")).digest()
