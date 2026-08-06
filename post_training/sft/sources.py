"""Pluggable instruction and replay sources for SFT dataset production."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, Mapping

from trainer.shards import SchemaV2ShardReader

from .schema import ConversationRecord, TokenizedSFTRecord


class JsonlConversationSource:
    """Read a deterministic local JSONL export of chat records."""

    def __init__(
        self,
        path: Path | str,
        *,
        expected_source: str | None = None,
        default_source: str | None = None,
    ) -> None:
        self.path = Path(path)
        self.expected_source = expected_source
        self.default_source = default_source

    def __iter__(self) -> Iterator[ConversationRecord]:
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"invalid JSON at {self.path}:{line_number}"
                    ) from error
                if not isinstance(payload, Mapping):
                    raise ValueError(
                        f"record at {self.path}:{line_number} must be an object"
                    )
                record = ConversationRecord.from_mapping(
                    payload,
                    default_source=self.default_source,
                )
                if (
                    self.expected_source is not None
                    and record.source != self.expected_source
                ):
                    continue
                yield record


class HuggingFaceConversationSource:
    """Optional streaming adapter for a pinned Hugging Face dataset revision.

    The heavy ``datasets`` package is intentionally optional. Production can
    either install it explicitly or export immutable JSONL and use the standard
    library source above.
    """

    def __init__(
        self,
        dataset_name: str,
        *,
        revision: str,
        split: str = "train",
        configuration: str | None = None,
        expected_source: str | None = None,
    ) -> None:
        if not dataset_name or not revision:
            raise ValueError("dataset_name and exact revision are required")
        self.dataset_name = dataset_name
        self.revision = revision
        self.split = split
        self.configuration = configuration
        self.expected_source = expected_source

    def __iter__(self) -> Iterator[ConversationRecord]:
        try:
            from datasets import load_dataset
        except ImportError as error:  # pragma: no cover - optional production path
            raise RuntimeError(
                "HuggingFaceConversationSource requires the optional 'datasets' package; "
                "otherwise export pinned rows to JSONL and use JsonlConversationSource"
            ) from error
        dataset = load_dataset(
            self.dataset_name,
            self.configuration,
            split=self.split,
            revision=self.revision,
            streaming=True,
        )
        for index, payload in enumerate(dataset):
            if not isinstance(payload, Mapping):
                raise ValueError("Hugging Face row must be an object")
            normalized = dict(payload)
            normalized.setdefault(
                "conversation_id",
                f"{self.dataset_name}@{self.revision}:{self.split}:{index}",
            )
            record = ConversationRecord.from_mapping(normalized)
            if (
                self.expected_source is not None
                and record.source != self.expected_source
            ):
                continue
            yield record


def iter_schema_v2_replay(
    root: Path | str,
    *,
    split: str = "train",
    context_length: int = 2_048,
    semantic_vocab_size: int = 50_257,
) -> Iterable[TokenizedSFTRecord]:
    """Expose existing immutable pretraining sequences as full-loss replay."""

    reader = SchemaV2ShardReader(
        root,
        split=split,
        context_length=context_length,
        semantic_vocab_size=semantic_vocab_size,
    )
    for batch in reader.iter_from_start():
        for row in range(batch.sequence_count):
            inputs = batch.input_ids[row].tolist()
            final_target = int(batch.labels[row, -1])
            tokens = tuple(int(value) for value in inputs) + (final_target,)
            yield TokenizedSFTRecord(
                record_id=f"climbmix:{batch.block_id}:{row}",
                source="climbmix-replay",
                split="train" if split == "train" else "validation",
                token_ids=tokens,
                target_mask=tuple(True for _ in range(len(tokens) - 1)),
                metadata={
                    "parent_block_id": batch.block_id,
                    "parent_sequence_index": row,
                    "source_dataset": "climbmix-schema-v2",
                },
            )


__all__ = [
    "HuggingFaceConversationSource",
    "JsonlConversationSource",
    "iter_schema_v2_replay",
]
