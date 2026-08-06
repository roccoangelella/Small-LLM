"""Trainer-facing data contracts."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping, Protocol
import torch
from torch import Tensor

IGNORE_INDEX = -100


@dataclass(frozen=True, slots=True)
class TokenBatch:
    block_id: int
    split: str
    input_ids: Tensor
    labels: Tensor
    sequence_count: int
    target_token_count: int
    cumulative_source_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.input_ids.dtype != torch.long or self.labels.dtype != torch.long:
            raise ValueError("token batches must use torch.long tensors")
        if self.input_ids.ndim != 2 or self.input_ids.shape != self.labels.shape:
            raise ValueError("input_ids and labels must have the same rank-2 shape")
        if self.sequence_count != self.input_ids.shape[0]:
            raise ValueError("sequence_count does not match tensor geometry")
        active_targets = int(self.labels.ne(IGNORE_INDEX).sum().item())
        if self.target_token_count != active_targets:
            raise ValueError(
                "target_token_count must equal the number of non-masked labels"
            )
        if self.target_token_count <= 0:
            raise ValueError("token batches must contain at least one active target")


class PreparedBlockLike(Protocol):
    block_id: int
    split: str
    sequence_count: int
    token_count: int
    payload: bytes
    cumulative_source_tokens: int
    schema_version: int


class BatchSource(Protocol):
    last_acknowledged_block_id: int
    def next_batch(self, timeout: float | None = None) -> TokenBatch: ...
    def acknowledge(self, block_id: int) -> None: ...
    def pipeline_state(self) -> dict[str, object]: ...
    def load_pipeline_state(self, state: Mapping[str, object]) -> None: ...


__all__ = [
    "BatchSource",
    "IGNORE_INDEX",
    "PreparedBlockLike",
    "TokenBatch",
]
