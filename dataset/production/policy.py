"""Frozen production policy and stable identity hashes."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Mapping

from dataset import config
from dataset.src.storage import canonical_json_bytes
from dataset.src.streaming import STREAM_CACHE_SCHEMA_VERSION, StreamCacheConfig, StreamCacheProducer
from dataset.src.workplan import WorkPlan

PRODUCTION_STATE_VERSION = 1
DEFAULT_CHECKPOINT_SOURCE_TOKENS = 1_000_000_000


@dataclass(frozen=True)
class ProductionPolicy:
    run_id: str
    target_source_tokens: int = config.TARGET_ACCEPTED_SOURCE_TOKENS
    minimum_source_tokens: int = config.MINIMUM_ACCEPTED_SOURCE_TOKENS
    maximum_source_tokens: int = config.MAXIMUM_ACCEPTED_SOURCE_TOKENS
    checkpoint_source_tokens: int = DEFAULT_CHECKPOINT_SOURCE_TOKENS
    remote_required: bool = True

    def __post_init__(self) -> None:
        if not self.run_id or "/" in self.run_id or "\\" in self.run_id or self.run_id in {".", ".."}:
            raise ValueError("run_id must be a non-empty safe path component")
        for name in (
            "target_source_tokens", "minimum_source_tokens",
            "maximum_source_tokens", "checkpoint_source_tokens",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not self.minimum_source_tokens <= self.target_source_tokens <= self.maximum_source_tokens:
            raise ValueError("production token limits must satisfy minimum <= target <= maximum")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def stream_config_dict(value: StreamCacheConfig) -> dict[str, object]:
    return {
        "context_length": value.context_length,
        "sequences_per_block": value.sequences_per_block,
        "target_shard_bytes": value.target_shard_bytes,
        "reader_workers": value.reader_workers,
        "max_in_flight_work_items": value.max_in_flight_work_items,
        "per_cluster_queue_limit": value.per_cluster_queue_limit,
        "prepared_block_queue_limit": value.prepared_block_queue_limit,
        "prefetch_head_start": value.prefetch_head_start,
        "weights": {
            str(key): str(weight)
            for key, weight in sorted(value.weights.items(), key=lambda item: int(item[0]))
        },
        "scheduler_tie_break_seed": value.scheduler_tie_break_seed,
        "rolling_mixture_windows": list(value.rolling_mixture_windows),
        "final_partial_sequence_policy": value.final_partial_sequence_policy,
        "minimum_prefetched_source_tokens": value.minimum_prefetched_source_tokens,
        "minimum_populated_cluster_queues": value.minimum_populated_cluster_queues,
        "maximum_rolling_mixture_error": value.maximum_rolling_mixture_error,
        "maximum_waiting_documents": value.maximum_waiting_documents,
        "reader_batch_source_tokens": value.reader_batch_source_tokens,
        "reader_batch_documents": value.reader_batch_documents,
        "reader_batch_max_bytes": value.reader_batch_max_bytes,
    }


def stable_hash(value: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_json_bytes(dict(value))).hexdigest()


def configuration_hash(policy: ProductionPolicy, stream: StreamCacheConfig, plan: WorkPlan) -> str:
    return stable_hash({
        "production_state_version": PRODUCTION_STATE_VERSION,
        "policy": policy.as_dict(),
        "stream_cache": stream_config_dict(stream),
        "work_plan_hash": plan.hash,
    })


def schema_hash(stream: StreamCacheConfig) -> str:
    return stable_hash({
        "stream_cache_schema_version": STREAM_CACHE_SCHEMA_VERSION,
        "sequence_format": "context_plus_one",
        "context_length": stream.context_length,
        "stored_sequence_tokens": stream.context_length + 1,
        "sequences_per_block": stream.sequences_per_block,
        "int_type": config.INT_TYPE,
        "byte_order": config.BYTE_ORDER,
        "eod_token_id": config.EOD_TOKEN_ID,
    })


def incorporated_source_tokens(producer: StreamCacheProducer) -> int:
    return (
        producer.scheduler.total_emitted_source_tokens
        + producer.validation_source_tokens
        + producer.queued_source_tokens
    )


def reader_configuration(stream: StreamCacheConfig) -> dict[str, int]:
    return {
        "reader_workers": stream.reader_workers,
        "max_in_flight_work_items": stream.max_in_flight_work_items,
        "reader_batch_source_tokens": stream.reader_batch_source_tokens,
        "reader_batch_documents": stream.reader_batch_documents,
        "reader_batch_max_bytes": stream.reader_batch_max_bytes,
    }
