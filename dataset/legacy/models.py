"""Small data structures shared by the dataset pipeline modules."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .storage import read_json


@dataclass(frozen=True)
class SourceDocument:
    """A tokenized source row plus its stable source position."""

    source_index: int
    cluster_id: int
    tokens: list[int]
    token_count: int


@dataclass(frozen=True)
class TextMetrics:
    """Deterministic signals used for selection and later verification."""

    character_count: int
    line_count: int
    word_count: int
    code_line_count: int
    code_line_fraction: float
    fenced_code_fraction: float
    code_symbol_fraction: float
    api_marker_count: int
    repository_path_count: int
    ascii_letter_ratio: float
    english_marker_hits: int
    english_marker_ratio: float
    likely_english: bool
    code_dominated: bool
    rejection_reason: str | None


@dataclass
class SelectionState:
    """Crash-safe checkpoint for an append-only selection run."""

    version: int = 1
    last_source_index: int = -1
    total_documents: int = 0
    total_tokens: int = 0
    total_text_bytes: int = 0
    per_cluster: dict[str, dict[str, int]] = field(default_factory=dict)
    next_shard_id: int = 0
    active_shard: str | None = None
    active_shard_bytes: int = 0

    @classmethod
    def load(cls, path: Path) -> "SelectionState":
        """Load a durable state file or start a new selection."""

        if not path.exists():
            return cls()
        payload = read_json(path)
        if payload.get("version") != 1:
            raise ValueError(f"Unsupported selection-state version in {path}")
        return cls(**payload)

    def cluster(self, cluster_id: int) -> dict[str, int]:
        """Return mutable counters for one configured cluster."""

        key = str(cluster_id)
        if key not in self.per_cluster:
            self.per_cluster[key] = {"documents": 0, "tokens": 0, "text_bytes": 0}
        return self.per_cluster[key]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the state without exposing implementation details."""

        return asdict(self)
