"""Immutable logical and tokenized SFT record contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

import torch

from trainer.types import TokenBatch

Role = Literal["system", "user", "assistant"]
Split = Literal["train", "validation", "test"]
IGNORE_INDEX = -100


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: Role
    content: str

    def __post_init__(self) -> None:
        if self.role not in {"system", "user", "assistant"}:
            raise ValueError(f"unsupported chat role: {self.role!r}")
        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError("chat message content must be non-empty text")


@dataclass(frozen=True, slots=True)
class ConversationRecord:
    conversation_id: str
    source: str
    messages: tuple[ChatMessage, ...]
    split: Split = "train"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.conversation_id:
            raise ValueError("conversation_id must be non-empty")
        if not self.source:
            raise ValueError("source must be non-empty")
        if self.split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation, or test")
        if not self.messages:
            raise ValueError("conversation must contain messages")

        roles = [message.role for message in self.messages]
        start = 1 if roles[0] == "system" else 0
        if "system" in roles[start:]:
            raise ValueError("system message is allowed only at the beginning")
        dialogue = roles[start:]
        if not dialogue or dialogue[0] != "user":
            raise ValueError("conversation must begin with a user message")
        for index, role in enumerate(dialogue):
            expected = "user" if index % 2 == 0 else "assistant"
            if role != expected:
                raise ValueError("user and assistant messages must alternate")
        if dialogue[-1] != "assistant":
            raise ValueError("training conversation must end with an assistant response")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any], *, default_source: str | None = None) -> "ConversationRecord":
        raw_messages = payload.get("messages")
        if not isinstance(raw_messages, Sequence) or isinstance(raw_messages, (str, bytes)):
            raise ValueError("record messages must be a sequence")
        messages: list[ChatMessage] = []
        for raw in raw_messages:
            if not isinstance(raw, Mapping):
                raise ValueError("each message must be an object")
            messages.append(ChatMessage(role=str(raw.get("role")), content=str(raw.get("content", ""))))  # type: ignore[arg-type]
        source = payload.get("source", default_source)
        if not isinstance(source, str) or not source:
            raise ValueError("record source is missing")
        identity = payload.get("conversation_id", payload.get("id"))
        if identity is None:
            raise ValueError("record conversation_id or id is missing")
        split = str(payload.get("split", "train"))
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("record metadata must be an object")
        return cls(
            conversation_id=str(identity),
            source=source,
            messages=tuple(messages),
            split=split,  # type: ignore[arg-type]
            metadata=dict(metadata),
        )


@dataclass(frozen=True, slots=True)
class TokenizedSFTRecord:
    record_id: str
    source: str
    split: Split
    token_ids: tuple[int, ...]
    target_mask: tuple[bool, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.record_id or not self.source:
            raise ValueError("tokenized record identity and source must be non-empty")
        if self.split not in {"train", "validation", "test"}:
            raise ValueError("invalid tokenized record split")
        if len(self.token_ids) < 2:
            raise ValueError("tokenized record needs at least two tokens")
        if len(self.target_mask) != len(self.token_ids) - 1:
            raise ValueError("target_mask must align with next-token labels")
        if not any(self.target_mask):
            raise ValueError("tokenized record has no loss-bearing targets")
        for token in self.token_ids:
            if isinstance(token, bool) or not isinstance(token, int) or not 0 <= token <= 65_535:
                raise ValueError("token IDs must be uint16-compatible integers")

    @property
    def target_token_count(self) -> int:
        return sum(self.target_mask)

    @property
    def serialized_token_count(self) -> int:
        return len(self.token_ids)


@dataclass(frozen=True, slots=True)
class SFTBlock:
    block_id: int
    split: Split
    records: tuple[TokenizedSFTRecord, ...]

    def __post_init__(self) -> None:
        if isinstance(self.block_id, bool) or not isinstance(self.block_id, int) or self.block_id < 0:
            raise ValueError("block_id must be a non-negative integer")
        if self.split not in {"train", "validation", "test"}:
            raise ValueError("invalid block split")
        if not self.records:
            raise ValueError("SFT block must contain records")
        if any(record.split != self.split for record in self.records):
            raise ValueError("all block records must have the block split")

    @property
    def target_token_count(self) -> int:
        return sum(record.target_token_count for record in self.records)

    @property
    def serialized_token_count(self) -> int:
        return sum(record.serialized_token_count for record in self.records)

    def to_token_batch(self, *, pad_token_id: int, ignore_index: int = IGNORE_INDEX) -> TokenBatch:
        """Right-pad records and preserve only explicitly active target labels."""

        if not 0 <= pad_token_id <= 65_535:
            raise ValueError("pad_token_id must be uint16-compatible")
        maximum_targets = max(len(record.token_ids) - 1 for record in self.records)
        inputs = torch.full(
            (len(self.records), maximum_targets),
            pad_token_id,
            dtype=torch.long,
        )
        labels = torch.full(
            (len(self.records), maximum_targets),
            ignore_index,
            dtype=torch.long,
        )
        for row, record in enumerate(self.records):
            token_ids = torch.tensor(record.token_ids, dtype=torch.long)
            length = token_ids.numel() - 1
            inputs[row, :length] = token_ids[:-1]
            active = torch.tensor(record.target_mask, dtype=torch.bool)
            labels[row, :length][active] = token_ids[1:][active]

        split = "validation" if self.split == "validation" else self.split
        return TokenBatch(
            block_id=self.block_id,
            split=split,
            input_ids=inputs,
            labels=labels,
            sequence_count=len(self.records),
            target_token_count=self.target_token_count,
            cumulative_source_tokens=None,
        )


__all__ = [
    "ChatMessage",
    "ConversationRecord",
    "IGNORE_INDEX",
    "Role",
    "SFTBlock",
    "Split",
    "TokenizedSFTRecord",
]
