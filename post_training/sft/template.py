"""Byte-exact S0 chat serialization and assistant-only target masking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from .schema import ChatMessage, ConversationRecord, TokenizedSFTRecord


class TokenEncoder(Protocol):
    def encode(self, text: str) -> list[int]: ...


@dataclass(frozen=True, slots=True)
class GPT2ChatTemplate:
    """Use existing GPT-2 tokens while providing explicit role/turn boundaries."""

    eos_token_id: int = 50_256
    maximum_context_tokens: int = 2_048
    maximum_assistant_tokens: int = 512

    def __post_init__(self) -> None:
        if not 0 <= self.eos_token_id <= 65_535:
            raise ValueError("eos_token_id must be uint16-compatible")
        if self.maximum_context_tokens <= 0:
            raise ValueError("maximum_context_tokens must be positive")
        if self.maximum_assistant_tokens <= 0:
            raise ValueError("maximum_assistant_tokens must be positive")

    @staticmethod
    def _role_prefix(role: str) -> str:
        return f"{role.capitalize()}:\n"

    def encode_conversation(
        self,
        record: ConversationRecord,
        encoder: TokenEncoder,
    ) -> TokenizedSFTRecord:
        """Serialize one conversation and mark assistant content plus per-turn EOS."""

        token_ids: list[int] = [self.eos_token_id]
        target_mask: list[bool] = []
        assistant_lengths: list[int] = []

        def append_masked(text: str) -> None:
            encoded = encoder.encode(text)
            token_ids.extend(encoded)
            target_mask.extend(False for _ in encoded)

        def append_assistant(text: str) -> None:
            encoded = encoder.encode(text)
            if len(encoded) > self.maximum_assistant_tokens:
                raise ValueError("assistant response exceeds maximum_assistant_tokens")
            assistant_lengths.append(len(encoded))
            token_ids.extend(encoded)
            target_mask.extend(True for _ in encoded)
            token_ids.append(self.eos_token_id)
            target_mask.append(True)

        for index, message in enumerate(record.messages):
            if index > 0:
                append_masked("\n")
            append_masked(self._role_prefix(message.role))
            if message.role == "assistant":
                append_assistant(message.content)
            else:
                append_masked(message.content)
                append_masked("\n")

        if len(target_mask) != len(token_ids) - 1:
            raise RuntimeError("chat target mask construction drifted from tokenization")
        if len(token_ids) - 1 > self.maximum_context_tokens:
            raise ValueError("serialized conversation exceeds model context")

        return TokenizedSFTRecord(
            record_id=record.conversation_id,
            source=record.source,
            split=record.split,
            token_ids=tuple(token_ids),
            target_mask=tuple(target_mask),
            metadata={
                **dict(record.metadata),
                "assistant_token_lengths": assistant_lengths,
                "chat_template": "small-llm-s0-v1",
            },
        )

    def encode_generation_prompt(
        self,
        messages: Sequence[ChatMessage],
        encoder: TokenEncoder,
    ) -> tuple[int, ...]:
        """Serialize chat history and finish at the Assistant generation prefix."""

        if not messages:
            raise ValueError("generation prompt must contain at least one message")
        roles = [message.role for message in messages]
        start = 1 if roles[0] == "system" else 0
        if "system" in roles[start:]:
            raise ValueError("system message is allowed only at the beginning")
        dialogue = roles[start:]
        if not dialogue or dialogue[0] != "user" or dialogue[-1] != "user":
            raise ValueError("generation dialogue must start and end with user")
        for index, role in enumerate(dialogue):
            expected = "user" if index % 2 == 0 else "assistant"
            if role != expected:
                raise ValueError("generation user/assistant history must alternate")

        token_ids: list[int] = [self.eos_token_id]
        for index, message in enumerate(messages):
            if index > 0:
                token_ids.extend(encoder.encode("\n"))
            token_ids.extend(encoder.encode(self._role_prefix(message.role)))
            token_ids.extend(encoder.encode(message.content))
            if message.role == "assistant":
                token_ids.append(self.eos_token_id)
            else:
                token_ids.extend(encoder.encode("\n"))
        token_ids.extend(encoder.encode("\nAssistant:\n"))
        if len(token_ids) > self.maximum_context_tokens:
            raise ValueError("generation prompt exceeds model context")
        return tuple(token_ids)


class TiktokenGPT2Encoder:
    """Lazy tiktoken adapter so dataset-only tests do not need the extra."""

    def __init__(self) -> None:
        try:
            import tiktoken
        except ImportError as error:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "tiktoken is required; install the project post-training extra"
            ) from error
        self._encoding = tiktoken.get_encoding("gpt2")

    def encode(self, text: str) -> list[int]:
        return list(self._encoding.encode(text, allowed_special=set(), disallowed_special=()))


__all__ = ["GPT2ChatTemplate", "TiktokenGPT2Encoder", "TokenEncoder"]
