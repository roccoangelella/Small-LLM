"""GPT-2-compatible tokenizer extension for R-SFT reasoning control tokens.

The token IDs are fixed by the accepted padded-row promotion contract, while the
three marker strings remain artifact-provided until the serialization ablation
freezes their spelling.  This module deliberately wraps the existing GPT-2
encoding instead of mutating the base S0 tokenizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence
import re


BASE_ENCODING_NAME = "gpt2"
BASE_SEMANTIC_VOCAB_SIZE = 50_257
REASONING_START_TOKEN_ID = 50_257
REASONING_END_TOKEN_ID = 50_258
ANSWER_START_TOKEN_ID = 50_259
R_SFT_SEMANTIC_VOCAB_SIZE = 50_260
TOKENIZER_METADATA_KEY = "reasoning_tokenizer"
TOKENIZER_METADATA_VERSION = 1

_EXPECTED_IDS = {
    "reasoning_start": REASONING_START_TOKEN_ID,
    "reasoning_end": REASONING_END_TOKEN_ID,
    "answer_start": ANSWER_START_TOKEN_ID,
}


@dataclass(frozen=True, slots=True)
class ReasoningTokenSpec:
    """Artifact-carried spelling for the three fixed R-SFT token IDs."""

    reasoning_start: str
    reasoning_end: str
    answer_start: str

    def __post_init__(self) -> None:
        values = (self.reasoning_start, self.reasoning_end, self.answer_start)
        if any(not isinstance(value, str) or not value for value in values):
            raise ValueError("reasoning token strings must be non-empty strings")
        if len(set(values)) != 3:
            raise ValueError("reasoning token strings must be distinct")

    @property
    def special_tokens(self) -> dict[str, int]:
        return {
            self.reasoning_start: REASONING_START_TOKEN_ID,
            self.reasoning_end: REASONING_END_TOKEN_ID,
            self.answer_start: ANSWER_START_TOKEN_ID,
        }

    @property
    def id_to_token(self) -> dict[int, str]:
        return {token_id: token for token, token_id in self.special_tokens.items()}

    def to_metadata(self) -> dict[str, object]:
        """Return the checkpoint pipeline-state representation."""

        return {
            "version": TOKENIZER_METADATA_VERSION,
            "base_encoding": BASE_ENCODING_NAME,
            "semantic_vocab_size": R_SFT_SEMANTIC_VOCAB_SIZE,
            "special_tokens": {
                "reasoning_start": {
                    "text": self.reasoning_start,
                    "id": REASONING_START_TOKEN_ID,
                },
                "reasoning_end": {
                    "text": self.reasoning_end,
                    "id": REASONING_END_TOKEN_ID,
                },
                "answer_start": {
                    "text": self.answer_start,
                    "id": ANSWER_START_TOKEN_ID,
                },
            },
        }

    @classmethod
    def from_metadata(cls, payload: Mapping[str, object]) -> "ReasoningTokenSpec":
        """Validate and reconstruct the artifact-carried token spelling."""

        if not isinstance(payload, Mapping):
            raise ValueError("reasoning tokenizer metadata must be an object")
        if payload.get("version") != TOKENIZER_METADATA_VERSION:
            raise ValueError("unsupported reasoning tokenizer metadata version")
        if payload.get("base_encoding") != BASE_ENCODING_NAME:
            raise ValueError("R-SFT reasoning tokenizer must extend GPT-2")
        if payload.get("semantic_vocab_size") != R_SFT_SEMANTIC_VOCAB_SIZE:
            raise ValueError(
                f"R-SFT semantic vocabulary must be {R_SFT_SEMANTIC_VOCAB_SIZE}"
            )
        raw_tokens = payload.get("special_tokens")
        if not isinstance(raw_tokens, Mapping) or set(raw_tokens) != set(_EXPECTED_IDS):
            raise ValueError("reasoning tokenizer metadata must declare exactly three control tokens")

        strings: dict[str, str] = {}
        for role, expected_id in _EXPECTED_IDS.items():
            raw = raw_tokens.get(role)
            if not isinstance(raw, Mapping):
                raise ValueError(f"reasoning token {role!r} must be an object")
            if set(raw) != {"text", "id"}:
                raise ValueError(f"reasoning token {role!r} must contain exactly text/id")
            text = raw.get("text")
            token_id = raw.get("id")
            if not isinstance(text, str) or not text:
                raise ValueError(f"reasoning token {role!r} text must be non-empty")
            if isinstance(token_id, bool) or token_id != expected_id:
                raise ValueError(
                    f"reasoning token {role!r} must use fixed token ID {expected_id}"
                )
            strings[role] = text
        return cls(**strings)


def spec_from_pipeline_state(pipeline_state: Mapping[str, object]) -> ReasoningTokenSpec:
    """Load the R-SFT token contract from verified checkpoint pipeline metadata."""

    if not isinstance(pipeline_state, Mapping):
        raise ValueError("R-SFT checkpoint has no pipeline_state object")
    payload = pipeline_state.get(TOKENIZER_METADATA_KEY)
    if not isinstance(payload, Mapping):
        raise ValueError(
            f"R-SFT checkpoint is missing pipeline_state.{TOKENIZER_METADATA_KEY}"
        )
    return ReasoningTokenSpec.from_metadata(payload)


class ReasoningGPT2Encoder:
    """GPT-2 encoder/decoder with three artifact-defined atomic token strings."""

    def __init__(self, spec: ReasoningTokenSpec, *, base_encoding: object | None = None) -> None:
        self.spec = spec
        if base_encoding is None:
            try:
                import tiktoken
            except ImportError as error:  # pragma: no cover - optional dependency
                raise RuntimeError(
                    "tiktoken is required; install the project post-training extra"
                ) from error
            base_encoding = tiktoken.get_encoding(BASE_ENCODING_NAME)
        self._base = base_encoding
        self._special_tokens = spec.special_tokens
        self._id_to_token = spec.id_to_token
        alternatives = sorted(self._special_tokens, key=len, reverse=True)
        self._special_pattern = re.compile("|".join(re.escape(token) for token in alternatives))

    def _encode_base(self, text: str) -> list[int]:
        if not text:
            return []
        encode = getattr(self._base, "encode", None)
        if not callable(encode):
            raise TypeError("base GPT-2 encoding must expose encode(text)")
        try:
            encoded = encode(text, allowed_special=set(), disallowed_special=())
        except TypeError:
            encoded = encode(text)
        return [int(token_id) for token_id in encoded]

    def encode(self, text: str) -> list[int]:
        """Encode configured marker strings atomically and all other text as GPT-2."""

        if not isinstance(text, str):
            raise TypeError("text must be a string")
        token_ids: list[int] = []
        cursor = 0
        for match in self._special_pattern.finditer(text):
            token_ids.extend(self._encode_base(text[cursor : match.start()]))
            token_ids.append(self._special_tokens[match.group(0)])
            cursor = match.end()
        token_ids.extend(self._encode_base(text[cursor:]))
        return token_ids

    def _decode_base(self, token_ids: Sequence[int]) -> str:
        if not token_ids:
            return ""
        decode = getattr(self._base, "decode", None)
        if not callable(decode):
            raise TypeError("base GPT-2 encoding must expose decode(token_ids)")
        return str(decode(list(token_ids)))

    def decode(self, token_ids: Sequence[int]) -> str:
        """Decode both ordinary GPT-2 IDs and the three promoted reasoning IDs."""

        parts: list[str] = []
        ordinary: list[int] = []

        def flush() -> None:
            if ordinary:
                parts.append(self._decode_base(ordinary))
                ordinary.clear()

        for raw_token_id in token_ids:
            if isinstance(raw_token_id, bool) or not isinstance(raw_token_id, int):
                raise ValueError("token IDs must be integers")
            token = self._id_to_token.get(raw_token_id)
            if token is not None:
                flush()
                parts.append(token)
                continue
            if not 0 <= raw_token_id < BASE_SEMANTIC_VOCAB_SIZE:
                raise ValueError(f"token ID {raw_token_id} is outside the R-SFT vocabulary")
            ordinary.append(raw_token_id)
        flush()
        return "".join(parts)

    def decode_single_token_bytes(self, token_id: int) -> bytes:
        """Support the chat streamer's byte-exact incremental decoding contract."""

        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise ValueError("token ID must be an integer")
        token = self._id_to_token.get(token_id)
        if token is not None:
            return token.encode("utf-8")
        if not 0 <= token_id < BASE_SEMANTIC_VOCAB_SIZE:
            raise ValueError(f"token ID {token_id} is outside the R-SFT vocabulary")
        decode_bytes = getattr(self._base, "decode_single_token_bytes", None)
        if not callable(decode_bytes):
            raise TypeError("base GPT-2 encoding must expose decode_single_token_bytes(token_id)")
        return bytes(decode_bytes(token_id))


__all__ = [
    "ANSWER_START_TOKEN_ID",
    "BASE_ENCODING_NAME",
    "BASE_SEMANTIC_VOCAB_SIZE",
    "REASONING_END_TOKEN_ID",
    "REASONING_START_TOKEN_ID",
    "R_SFT_SEMANTIC_VOCAB_SIZE",
    "ReasoningGPT2Encoder",
    "ReasoningTokenSpec",
    "TOKENIZER_METADATA_KEY",
    "spec_from_pipeline_state",
]
