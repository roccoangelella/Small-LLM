"""Deterministic S0 scope and structural filters.

These filters intentionally reject only high-confidence out-of-scope material.
They do not attempt semantic capability classification.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Pattern

from .schema import ConversationRecord

DEFAULT_ALLOWED_SOURCES = frozenset(
    {
        "smol-magpie-ultra-short",
        "smol-contraints",
        "smollm-rewrite-30k",
        "smol-summarize-20k",
    }
)

_CODE_PATTERNS = (
    re.compile(r"```(?:python|javascript|typescript|java|c\+\+|rust|go|bash|sql)?", re.I),
    re.compile(r"\b(?:write|implement|debug|refactor)\b.{0,40}\b(?:code|function|class|script|program)\b", re.I | re.S),
    re.compile(r"\b(?:python|javascript|typescript|java|c\+\+|rust|golang)\b.{0,30}\b(?:code|function|class|program)\b", re.I | re.S),
)
_TOOL_PATTERNS = (
    re.compile(r"<tool_call>|</tool_call>|<function_call>|</function_call>", re.I),
    re.compile(r'"(?:tool_calls|function_call)"\s*:', re.I),
)
_LONG_REASONING_PATTERNS = (
    re.compile(r"\b(?:show|write|provide|reveal)\b.{0,30}\b(?:chain of thought|complete reasoning trace|internal reasoning)\b", re.I | re.S),
    re.compile(r"\blet'?s think step by step\b", re.I),
)
_ADVANCED_MATH_PATTERNS = (
    re.compile(r"\b(?:olympiad|imo|putnam|abstract algebra|measure theory|functional analysis)\b", re.I),
    re.compile(r"\\begin\{(?:align|equation|proof)\}|\\int_|\\sum_", re.I),
)
_ROLEPLAY_PATTERNS = (
    re.compile(r"\b(?:pretend|act|role-?play)\s+(?:that\s+)?you are\b", re.I),
    re.compile(r"\byou are now (?:a|an|the)\b", re.I),
)


@dataclass(frozen=True, slots=True)
class FilterDecision:
    accepted: bool
    reason: str = "accepted"


@dataclass(frozen=True, slots=True)
class S0RecordFilter:
    allowed_sources: frozenset[str] = DEFAULT_ALLOWED_SOURCES
    maximum_total_characters: int = 40_000
    reject_roleplay: bool = True

    def __post_init__(self) -> None:
        if self.maximum_total_characters <= 0:
            raise ValueError("maximum_total_characters must be positive")
        if not self.allowed_sources:
            raise ValueError("allowed_sources cannot be empty")

    @staticmethod
    def _matches(patterns: tuple[Pattern[str], ...], text: str) -> bool:
        return any(pattern.search(text) is not None for pattern in patterns)

    def evaluate(self, record: ConversationRecord) -> FilterDecision:
        if record.source not in self.allowed_sources:
            return FilterDecision(False, "source_not_allowed")
        text = "\n".join(message.content for message in record.messages)
        if len(text) > self.maximum_total_characters:
            return FilterDecision(False, "too_many_characters")
        if self._matches(_CODE_PATTERNS, text):
            return FilterDecision(False, "code")
        if self._matches(_TOOL_PATTERNS, text):
            return FilterDecision(False, "tool_call")
        if self._matches(_LONG_REASONING_PATTERNS, text):
            return FilterDecision(False, "long_reasoning_request")
        if self._matches(_ADVANCED_MATH_PATTERNS, text):
            return FilterDecision(False, "advanced_math")
        if self.reject_roleplay and self._matches(_ROLEPLAY_PATTERNS, text):
            return FilterDecision(False, "roleplay")
        return FilterDecision(True)


__all__ = [
    "DEFAULT_ALLOWED_SOURCES",
    "FilterDecision",
    "S0RecordFilter",
]
