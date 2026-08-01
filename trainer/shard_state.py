"""Checkpoint state for immutable schema-v2 readers."""
from __future__ import annotations
from typing import Mapping

def reader_state(reader: object) -> dict[str, object]:
    if reader._outstanding is not None:
        raise RuntimeError("cannot checkpoint with an unacknowledged block")
    return {"version": 1, "kind": "immutable_schema_v2",
        "manifest_identity": reader.manifest_identity, "split": reader.split,
        "context_length": reader.context_length, "sequences_per_block": reader.sequences_per_block,
        "last_consumed_block_id": reader.last_acknowledged_block_id}

def load_reader_state(reader: object, state: Mapping[str, object]) -> None:
    expected = {"version": 1, "kind": "immutable_schema_v2",
        "manifest_identity": reader.manifest_identity, "split": reader.split,
        "context_length": reader.context_length, "sequences_per_block": reader.sequences_per_block}
    if reader._outstanding is not None:
        raise RuntimeError("cannot restore with an outstanding block")
    for key, value in expected.items():
        if state.get(key) != value:
            raise ValueError(f"shard-reader checkpoint mismatch for {key}")
    cursor = state.get("last_consumed_block_id")
    if isinstance(cursor, bool) or not isinstance(cursor, int) or cursor < -1:
        raise ValueError("shard-reader checkpoint has an invalid cursor")
    if cursor >= 0 and cursor not in reader._index:
        raise ValueError("shard-reader checkpoint cursor is outside this dataset")
    reader.last_acknowledged_block_id = cursor

def pipeline_state(reader: object) -> dict[str, object]:
    return {"version": 1, "last_consumed_block_id": reader.last_acknowledged_block_id,
            "gradient_accumulation_position": 0, "consumer": reader_state(reader)}

def load_pipeline_state(reader: object, state: Mapping[str, object]) -> None:
    if state.get("gradient_accumulation_position", 0) != 0:
        raise ValueError("cannot resume partial gradient accumulation")
    consumer = state.get("consumer")
    if isinstance(consumer, Mapping) and consumer.get("kind") == "immutable_schema_v2":
        load_reader_state(reader, consumer)
        return
    if isinstance(consumer, Mapping) and consumer.get("kind") not in {None, "live_schema_v2"}:
        raise ValueError("pipeline state names an unsupported consumer kind")
    load_reader_state(reader, {**reader_state(reader),
        "last_consumed_block_id": state.get("last_consumed_block_id")})
