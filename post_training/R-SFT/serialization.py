"""Serialize accepted reasoning examples into the existing SFT conversation contract.

The three control-marker strings remain explicit inputs so this module does not
silently freeze the pending textual-versus-atomic token ablation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from typing import Any


def _load_schema() -> ModuleType:
    module_name = "small_llm_rsft_schema"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    path = Path(__file__).with_name("schema.py")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load R-SFT schema module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


schema = _load_schema()
DEFAULT_REASONING_SOURCE = "r0-reasoning"
_VALID_SPLITS = frozenset({"train", "validation", "test"})


@dataclass(frozen=True, slots=True)
class ReasoningMarkers:
    reasoning_start: str
    reasoning_end: str
    answer_start: str

    def __post_init__(self) -> None:
        values = (self.reasoning_start, self.reasoning_end, self.answer_start)
        if any(not isinstance(value, str) or not value for value in values):
            raise ValueError("all reasoning serialization markers must be non-empty strings")
        if len(set(values)) != len(values):
            raise ValueError("reasoning serialization markers must be distinct")


def stable_conversation_id(example: Any) -> str:
    """Derive identity from the actual training content without storing provenance."""

    payload = "\x1f".join(
        (
            example.skill,
            example.difficulty,
            example.problem,
            example.reasoning,
            example.answer,
        )
    ).encode("utf-8")
    return f"rsft-{hashlib.sha256(payload).hexdigest()[:20]}"


def render_assistant_target(example: Any, markers: ReasoningMarkers) -> str:
    """Render reasoning then final answer using caller-selected control markers."""

    return (
        f"{markers.reasoning_start}{example.reasoning}"
        f"{markers.reasoning_end}{markers.answer_start}{example.answer}"
    )


def to_conversation_mapping(
    example: Any,
    *,
    markers: ReasoningMarkers,
    source: str = DEFAULT_REASONING_SOURCE,
    split: str = "train",
) -> dict[str, object]:
    """Convert an accepted record to the mapping consumed by S0 ConversationRecord."""

    if not isinstance(source, str) or not source.strip():
        raise ValueError("source must be a non-empty string")
    if split not in _VALID_SPLITS:
        raise ValueError("split must be train, validation, or test")
    return {
        "conversation_id": stable_conversation_id(example),
        "source": source.strip(),
        "split": split,
        "messages": [
            {"role": "user", "content": example.problem},
            {
                "role": "assistant",
                "content": render_assistant_target(example, markers),
            },
        ],
        "metadata": {
            "skill": example.skill,
            "difficulty": example.difficulty,
            "rsft_format": "reasoning-answer-v1",
        },
    }


__all__ = [
    "DEFAULT_REASONING_SOURCE",
    "ReasoningMarkers",
    "render_assistant_target",
    "stable_conversation_id",
    "to_conversation_mapping",
]
