"""Minimal strict schemas for reasoning-SFT teacher output and stored records."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

TEACHER_FIELDS = frozenset({"problem", "reasoning", "answer"})


@dataclass(frozen=True, slots=True)
class TeacherExample:
    """Exactly one Gemini-produced training example before project metadata is attached."""

    problem: str
    reasoning: str
    answer: str

    def __post_init__(self) -> None:
        for name in ("problem", "reasoning", "answer"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True, slots=True)
class ReasoningExample:
    """One accepted R-SFT record with only the metadata needed downstream."""

    skill: str
    difficulty: str
    problem: str
    reasoning: str
    answer: str

    def __post_init__(self) -> None:
        for name in ("skill", "difficulty", "problem", "reasoning", "answer"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")

    @classmethod
    def from_teacher(
        cls,
        example: TeacherExample,
        *,
        skill: str,
        difficulty: str,
    ) -> "ReasoningExample":
        return cls(
            skill=skill,
            difficulty=difficulty,
            problem=example.problem,
            reasoning=example.reasoning,
            answer=example.answer,
        )

    def as_dict(self) -> dict[str, str]:
        return {
            "skill": self.skill,
            "difficulty": self.difficulty,
            "problem": self.problem,
            "reasoning": self.reasoning,
            "answer": self.answer,
        }


def _parse_teacher_item(raw: Any, *, index: int) -> TeacherExample:
    if not isinstance(raw, Mapping):
        raise ValueError(f"teacher item {index} must be a JSON object")
    keys = set(raw)
    if keys != TEACHER_FIELDS:
        missing = sorted(TEACHER_FIELDS - keys)
        extra = sorted(keys - TEACHER_FIELDS)
        raise ValueError(
            f"teacher item {index} must contain exactly problem/reasoning/answer; "
            f"missing={missing} extra={extra}"
        )
    values: dict[str, str] = {}
    for name in ("problem", "reasoning", "answer"):
        value = raw[name]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"teacher item {index} field {name!r} must be non-empty text")
        values[name] = value.strip()
    return TeacherExample(**values)


def parse_teacher_batch(text: str, *, expected_count: int) -> tuple[TeacherExample, ...]:
    """Parse one Gemini response and fail closed on any schema drift."""

    if not isinstance(text, str) or not text.strip():
        raise ValueError("teacher response must be non-empty text")
    if isinstance(expected_count, bool) or not isinstance(expected_count, int) or expected_count <= 0:
        raise ValueError("expected_count must be a positive integer")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError("teacher response must be valid JSON with no surrounding text") from error
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes, bytearray)):
        raise ValueError("teacher response must be a top-level JSON array")
    if len(payload) != expected_count:
        raise ValueError(
            f"teacher response returned {len(payload)} records; expected {expected_count}"
        )
    return tuple(_parse_teacher_item(raw, index=index) for index, raw in enumerate(payload))


def write_jsonl(records: Sequence[ReasoningExample], path: str | Path) -> Path:
    """Write accepted records as compact UTF-8 JSONL."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.as_dict(), ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    return destination


__all__ = [
    "ReasoningExample",
    "TEACHER_FIELDS",
    "TeacherExample",
    "parse_teacher_batch",
    "write_jsonl",
]
