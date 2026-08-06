"""Deterministic target-token mixture and atomic-block construction."""

from __future__ import annotations

import random
from typing import Iterable, Iterator, Mapping

from .schema import SFTBlock, TokenizedSFTRecord


class BufferedShuffle:
    """Bounded-memory deterministic shuffle without replacement."""

    def __init__(
        self,
        records: Iterable[TokenizedSFTRecord],
        *,
        seed: int,
        buffer_size: int,
    ) -> None:
        if buffer_size <= 0:
            raise ValueError("buffer_size must be positive")
        self.records = records
        self.seed = seed
        self.buffer_size = buffer_size

    def __iter__(self) -> Iterator[TokenizedSFTRecord]:
        rng = random.Random(self.seed)
        iterator = iter(self.records)
        buffer: list[TokenizedSFTRecord] = []
        for _ in range(self.buffer_size):
            try:
                buffer.append(next(iterator))
            except StopIteration:
                break
        while buffer:
            index = rng.randrange(len(buffer))
            selected = buffer[index]
            try:
                buffer[index] = next(iterator)
            except StopIteration:
                buffer.pop(index)
            yield selected


class TargetTokenMixer:
    """Choose the source furthest behind its configured active-token share."""

    def __init__(
        self,
        sources: Mapping[str, Iterable[TokenizedSFTRecord]],
        shares: Mapping[str, float],
        *,
        seed: int,
        target_loss_tokens: int,
    ) -> None:
        if target_loss_tokens <= 0:
            raise ValueError("target_loss_tokens must be positive")
        if set(sources) != set(shares):
            raise ValueError("sources and shares must have identical keys")
        if not sources:
            raise ValueError("at least one source is required")
        if abs(sum(float(value) for value in shares.values()) - 1.0) > 1e-9:
            raise ValueError("source shares must sum to one")
        if any(float(value) <= 0 for value in shares.values()):
            raise ValueError("every configured source share must be positive")
        self.sources = dict(sources)
        self.shares = {key: float(value) for key, value in shares.items()}
        self.seed = seed
        self.target_loss_tokens = target_loss_tokens

    def __iter__(self) -> Iterator[TokenizedSFTRecord]:
        rng = random.Random(self.seed)
        iterators = {name: iter(records) for name, records in self.sources.items()}
        pending: dict[str, TokenizedSFTRecord] = {}
        exhausted: set[str] = set()
        selected_tokens = {name: 0 for name in iterators}
        total = 0

        def fill(name: str) -> None:
            if name in pending or name in exhausted:
                return
            try:
                record = next(iterators[name])
            except StopIteration:
                exhausted.add(name)
                return
            if record.source != name:
                raise ValueError(
                    f"source iterator {name!r} yielded record from {record.source!r}"
                )
            pending[name] = record

        while total < self.target_loss_tokens:
            for name in iterators:
                fill(name)
            eligible = [
                name
                for name, record in pending.items()
                if record.target_token_count <= self.target_loss_tokens - total
            ]
            if not eligible:
                break

            progress = {
                name: selected_tokens[name] / self.shares[name]
                for name in eligible
            }
            minimum = min(progress.values())
            tied = sorted(
                name
                for name, value in progress.items()
                if abs(value - minimum) <= 1e-12
            )
            name = tied[rng.randrange(len(tied))]
            record = pending.pop(name)
            selected_tokens[name] += record.target_token_count
            total += record.target_token_count
            yield record

        if total == 0:
            raise RuntimeError("SFT mixture produced no records")


def build_atomic_blocks(
    records: Iterable[TokenizedSFTRecord],
    *,
    target_tokens_per_block: int,
) -> Iterator[SFTBlock]:
    """Build complete-record atomic blocks without crossing the target ceiling."""

    if target_tokens_per_block <= 0:
        raise ValueError("target_tokens_per_block must be positive")
    block_id = 0
    current: list[TokenizedSFTRecord] = []
    current_targets = 0
    split: str | None = None

    for record in records:
        if record.target_token_count > target_tokens_per_block:
            raise ValueError(
                f"record {record.record_id!r} exceeds atomic block target capacity"
            )
        if split is None:
            split = record.split
        if record.split != split:
            raise ValueError("one atomic block stream cannot mix data splits")
        if current and current_targets + record.target_token_count > target_tokens_per_block:
            yield SFTBlock(block_id, split, tuple(current))  # type: ignore[arg-type]
            block_id += 1
            current = []
            current_targets = 0
        current.append(record)
        current_targets += record.target_token_count

    if current:
        assert split is not None
        yield SFTBlock(block_id, split, tuple(current))  # type: ignore[arg-type]


__all__ = ["BufferedShuffle", "TargetTokenMixer", "build_atomic_blocks"]
