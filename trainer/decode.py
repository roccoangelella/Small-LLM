"""Schema-v2 context+1 decoding."""
from __future__ import annotations
import sys
import torch
from .types import PreparedBlockLike, TokenBatch

SCHEMA_VERSION = 2

class PreparedBlockDecoder:
    def __init__(self, *, context_length: int, semantic_vocab_size: int,
                 expected_split: str = "train") -> None:
        if context_length <= 0:
            raise ValueError("context_length must be positive")
        if not 0 < semantic_vocab_size <= 65_536:
            raise ValueError("semantic_vocab_size must fit uint16")
        if expected_split not in {"train", "validation"}:
            raise ValueError("expected_split must be train or validation")
        self.context_length = context_length
        self.stored_tokens_per_sequence = context_length + 1
        self.semantic_vocab_size = semantic_vocab_size
        self.expected_split = expected_split

    def validate(self, block: PreparedBlockLike) -> int:
        if block.schema_version != SCHEMA_VERSION or block.split != self.expected_split:
            raise ValueError("prepared block schema or split mismatch")
        if isinstance(block.block_id, bool) or not isinstance(block.block_id, int) or block.block_id < 0:
            raise ValueError("prepared block has an invalid block ID")
        if isinstance(block.sequence_count, bool) or not isinstance(block.sequence_count, int) or block.sequence_count <= 0:
            raise ValueError("prepared block has an invalid sequence count")
        tokens = block.sequence_count * self.stored_tokens_per_sequence
        if block.token_count != tokens or not isinstance(block.payload, bytes) or len(block.payload) != tokens * 2:
            raise ValueError("prepared block payload does not match sequence geometry")
        return tokens

    def decode(self, block: PreparedBlockLike) -> TokenBatch:
        self.validate(block)
        if sys.byteorder != "little":  # pragma: no cover
            raise RuntimeError("schema-v2 decoding requires a little-endian host")
        raw = torch.frombuffer(bytearray(block.payload), dtype=torch.uint16)
        tokens = raw.long().reshape(block.sequence_count, self.stored_tokens_per_sequence)
        if int(tokens.max()) >= self.semantic_vocab_size:
            raise ValueError("prepared block contains a token outside the semantic vocabulary")
        inputs, labels = tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()
        return TokenBatch(block.block_id, block.split, inputs, labels, block.sequence_count,
                          labels.numel(), int(block.cumulative_source_tokens))
