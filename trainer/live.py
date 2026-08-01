"""Bounded live prepared-block consumer."""
from __future__ import annotations
import queue
from typing import Mapping
from .decode import PreparedBlockDecoder
from .types import PreparedBlockLike, TokenBatch

class LiveBlockConsumer:
    """Acknowledge a queued block only after its optimizer update succeeds."""
    def __init__(self, maxsize: int, *, context_length: int, semantic_vocab_size: int,
                 split: str = "train", last_consumed_block_id: int = -1) -> None:
        if maxsize <= 0 or last_consumed_block_id < -1:
            raise ValueError("invalid queue size or consumed cursor")
        self.queue: queue.Queue[PreparedBlockLike] = queue.Queue(maxsize=maxsize)
        self.decoder = PreparedBlockDecoder(context_length=context_length,
            semantic_vocab_size=semantic_vocab_size, expected_split=split)
        self.last_acknowledged_block_id = last_consumed_block_id
        self._last_submitted = last_consumed_block_id
        self._outstanding: TokenBatch | None = None

    def submit(self, block: PreparedBlockLike) -> None:
        if block.block_id != self._last_submitted + 1:
            raise ValueError("prepared blocks must be submitted contiguously")
        self.decoder.validate(block)
        self.queue.put(block)
        self._last_submitted = block.block_id

    def next_batch(self, timeout: float | None = None) -> TokenBatch:
        if self._outstanding is not None:
            raise RuntimeError("the previous block has not been acknowledged")
        block = self.queue.get() if timeout is None else self.queue.get(timeout=timeout)
        batch = self.decoder.decode(block)
        if batch.block_id != self.last_acknowledged_block_id + 1:
            raise RuntimeError("dequeued block does not follow the consumed cursor")
        self._outstanding = batch
        return batch

    def acknowledge(self, block_id: int) -> None:
        if self._outstanding is None or block_id != self._outstanding.block_id:
            raise ValueError("only the current trained block may be acknowledged")
        self.last_acknowledged_block_id = block_id
        self._outstanding = None
        self.queue.task_done()

    def pipeline_state(self) -> dict[str, object]:
        if self._outstanding is not None or not self.queue.empty() or self._last_submitted != self.last_acknowledged_block_id:
            raise RuntimeError("live consumer must be drained before a joint checkpoint")
        cursor = self.last_acknowledged_block_id
        return {"version": 1, "last_consumed_block_id": cursor,
                "gradient_accumulation_position": 0,
                "consumer": {"kind": "live_schema_v2", "split": self.decoder.expected_split,
                             "last_consumed_block_id": cursor}}

    def load_pipeline_state(self, state: Mapping[str, object]) -> None:
        if self._outstanding is not None or not self.queue.empty():
            raise RuntimeError("cannot restore a non-empty live consumer")
        cursor = state.get("last_consumed_block_id")
        if isinstance(cursor, bool) or not isinstance(cursor, int) or cursor < -1:
            raise ValueError("pipeline state has an invalid consumed cursor")
        self.last_acknowledged_block_id = self._last_submitted = cursor
