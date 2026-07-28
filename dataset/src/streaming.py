"""Deterministic first-pass streaming cache primitives.

This module is deliberately independent from a model framework.  It turns
validated source documents into fixed-geometry blocks, makes every block durable
in an immutable shard, and then exposes the *same bytes* to a consumer.  The
legacy ``build`` command remains available for the old monolithic format; this
is the schema-v1 streaming-cache path.

The scheduler never uses floating point.  Supplied decimal weights are converted
to :class:`fractions.Fraction`, reduced to a common integer scale, and compared
with ``units[c] * total - emitted[c] * sum(units)``.  That is exactly the
largest-deficit rule without cumulative rounding drift.
"""

from __future__ import annotations

import hashlib
import math
import os
import queue
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Callable, Iterable, Iterator, Mapping, Protocol, Sequence

from dataset import config

from .bitio import tokens_to_uint16_le_bytes
from .bytesource import RangeReader, SourceFile
from .manifest import sha256_file
from .records import iter_owned_records, validate_record
from .split import is_validation
from .storage import write_json_atomic
from .workplan import WorkItem, WorkPlan


STREAM_CACHE_SCHEMA_VERSION = 1
SEQUENCE_FORMAT = "context_plus_one"


def _lcm(left: int, right: int) -> int:
    return left // math.gcd(left, right) * right


def normalize_cluster_weights(weights: Mapping[int | str, object]) -> tuple[dict[int, str], dict[int, int]]:
    """Validate all accepted-cluster weights and return supplied and exact units.

    A mapping must contain *only* the accepted clusters.  In particular, cluster
    11 is not a harmless zero: including it is rejected so an accidental
    production configuration cannot conceal a policy error.
    """

    supplied: dict[int, str] = {}
    fractions: dict[int, Fraction] = {}
    for raw_cluster, raw_weight in weights.items():
        cluster = int(raw_cluster)
        if cluster in supplied:
            raise ValueError(f"duplicate cluster weight {cluster}")
        if isinstance(raw_weight, bool):
            raise ValueError(f"cluster {cluster} weight must be a positive finite number")
        try:
            value = Fraction(str(raw_weight))
        except (ValueError, ZeroDivisionError) as error:
            raise ValueError(f"cluster {cluster} has invalid weight {raw_weight!r}") from error
        if value <= 0:
            raise ValueError(f"cluster {cluster} weight must be positive")
        supplied[cluster] = str(raw_weight)
        fractions[cluster] = value
    expected = set(config.ACCEPTED_CLUSTER_IDS)
    actual = set(fractions)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            "cluster weights must contain exactly accepted clusters "
            f"(missing={missing}, extra={extra}; cluster 11 is excluded)"
        )
    denominator = 1
    for value in fractions.values():
        denominator = _lcm(denominator, value.denominator)
    units = {cluster: int(value * denominator) for cluster, value in fractions.items()}
    divisor = 0
    for value in units.values():
        divisor = math.gcd(divisor, value)
    units = {cluster: value // divisor for cluster, value in sorted(units.items())}
    return dict(sorted(supplied.items())), units


def synthetic_test_weights() -> dict[int, int]:
    """Equal test-only weights.  Never used as a production default."""

    return {cluster: 1 for cluster in config.ACCEPTED_CLUSTER_IDS}


@dataclass(frozen=True)
class StreamCacheConfig:
    context_length: int
    sequences_per_block: int
    target_shard_bytes: int
    reader_workers: int
    max_in_flight_work_items: int
    per_cluster_queue_limit: int
    prepared_block_queue_limit: int
    prefetch_head_start: int
    weights: Mapping[int | str, object]
    scheduler_tie_break_seed: str
    rolling_mixture_windows: tuple[int, ...] = (1_000_000, 10_000_000, 100_000_000)
    final_partial_sequence_policy: str = "pad_eod"
    # These are deliberately source-token/document bounds, rather than "all
    # clusters must be ready" gates.  A source can temporarily be sparse or
    # exhausted without stalling the producer forever.
    minimum_prefetched_source_tokens: int = 1
    minimum_populated_cluster_queues: int = 1
    maximum_rolling_mixture_error: float = 1.0
    maximum_waiting_documents: int = 32
    reader_batch_source_tokens: int = 1_000_000
    reader_batch_documents: int = 1_000
    reader_batch_max_bytes: int = 16 * 1024 * 1024

    def __post_init__(self) -> None:
        for name in (
            "context_length", "sequences_per_block", "target_shard_bytes",
            "reader_workers", "max_in_flight_work_items", "per_cluster_queue_limit",
            "prepared_block_queue_limit",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.prefetch_head_start < 0:
            raise ValueError("prefetch_head_start cannot be negative")
        if not self.scheduler_tie_break_seed:
            raise ValueError("scheduler_tie_break_seed cannot be empty")
        if self.final_partial_sequence_policy not in {"pad_eod", "error"}:
            raise ValueError("final_partial_sequence_policy must be 'pad_eod' or 'error'")
        if any(window <= 0 for window in self.rolling_mixture_windows):
            raise ValueError("rolling mixture windows must be positive")
        if self.minimum_prefetched_source_tokens < 0:
            raise ValueError("minimum_prefetched_source_tokens cannot be negative")
        if self.minimum_populated_cluster_queues <= 0:
            raise ValueError("minimum_populated_cluster_queues must be positive")
        if self.maximum_rolling_mixture_error < 0:
            raise ValueError("maximum_rolling_mixture_error cannot be negative")
        if self.maximum_waiting_documents < 0:
            raise ValueError("maximum_waiting_documents cannot be negative")
        if min(self.reader_batch_source_tokens, self.reader_batch_documents, self.reader_batch_max_bytes) <= 0:
            raise ValueError("reader batch limits must be positive")
        normalize_cluster_weights(self.weights)

    @property
    def stored_sequence_tokens(self) -> int:
        """Each record has context tokens plus the next-token label."""

        return self.context_length + 1

    @property
    def block_bytes(self) -> int:
        return self.stored_sequence_tokens * self.sequences_per_block * 2


@dataclass(frozen=True)
class SourceDocument:
    """A validated accepted document with a source identity stable across runs."""

    source_id: str
    cluster_id: int
    tokens: tuple[int, ...]
    work_item_index: int = 0
    record_start: int = 0

    @property
    def source_token_count(self) -> int:
        return len(self.tokens)


class TokenDeficitScheduler:
    """Whole-document, exact-integer largest-deficit scheduler."""

    def __init__(self, weights: Mapping[int | str, object], tie_break_seed: str) -> None:
        self.supplied_weights, self.weight_units = normalize_cluster_weights(weights)
        self.weight_total = sum(self.weight_units.values())
        self.tie_break_seed = tie_break_seed
        self.emitted_source_tokens = {cluster: 0 for cluster in self.weight_units}
        self.total_emitted_source_tokens = 0
        self.tie_break_counter = 0

    def deficit_numerator(self, cluster_id: int) -> int:
        return (
            self.weight_units[cluster_id] * self.total_emitted_source_tokens
            - self.emitted_source_tokens[cluster_id] * self.weight_total
        )

    def choose(self, available_clusters: Iterable[int]) -> int | None:
        available = sorted(set(available_clusters))
        if not available:
            return None
        unknown = set(available) - set(self.weight_units)
        if unknown:
            raise ValueError(f"scheduler received unaccepted clusters {sorted(unknown)}")
        highest = max(self.deficit_numerator(cluster) for cluster in available)
        tied = [cluster for cluster in available if self.deficit_numerator(cluster) == highest]
        if len(tied) == 1:
            return tied[0]
        return min(
            tied,
            key=lambda cluster: hashlib.sha256(
                f"{self.tie_break_seed}\0{self.tie_break_counter}\0{cluster}".encode("utf-8")
            ).digest(),
        )

    def emit(self, document: SourceDocument) -> None:
        if document.cluster_id not in self.weight_units:
            raise ValueError(f"cannot schedule excluded cluster {document.cluster_id}")
        tokens = document.source_token_count
        if tokens <= 0:
            raise ValueError("scheduled document must have source tokens")
        self.emitted_source_tokens[document.cluster_id] += tokens
        self.total_emitted_source_tokens += tokens
        # A scheduling decision is committed only when its whole document is
        # emitted.  Keeping ``choose`` pure makes repeated availability probes
        # deterministic and makes this counter safe to checkpoint.
        self.tie_break_counter += 1

    def state_dict(self) -> dict[str, object]:
        return {
            "weight_units": {str(k): v for k, v in self.weight_units.items()},
            "emitted_source_tokens": {str(k): v for k, v in self.emitted_source_tokens.items()},
            "total_emitted_source_tokens": self.total_emitted_source_tokens,
            "tie_break_seed": self.tie_break_seed,
            "tie_break_counter": self.tie_break_counter,
        }

    @classmethod
    def from_state(cls, weights: Mapping[int | str, object], state: Mapping[str, object]) -> "TokenDeficitScheduler":
        instance = cls(weights, str(state["tie_break_seed"]))
        expected = {str(k): v for k, v in instance.weight_units.items()}
        if state.get("weight_units") != expected:
            raise ValueError("scheduler state weights do not match this run")
        raw = state.get("emitted_source_tokens")
        if not isinstance(raw, Mapping):
            raise ValueError("scheduler state has no emitted_source_tokens")
        instance.emitted_source_tokens = {cluster: int(raw[str(cluster)]) for cluster in instance.weight_units}
        instance.total_emitted_source_tokens = int(state["total_emitted_source_tokens"])
        instance.tie_break_counter = int(state["tie_break_counter"])
        if sum(instance.emitted_source_tokens.values()) != instance.total_emitted_source_tokens:
            raise ValueError("scheduler state total does not equal per-cluster counters")
        return instance


@dataclass(frozen=True)
class PackedSequence:
    tokens: tuple[int, ...]
    first_source_id: str
    last_source_id: str
    cluster_source_tokens: dict[int, int]
    # One entry per physical stored token.  Values are source, inserted_eod,
    # overlap_source, overlap_eod, or padding.  This is intentionally retained
    # in prepared state (not in .bin) so attribution is auditable and resumable.
    token_kinds: tuple[str, ...] = ()
    token_clusters: tuple[int | None, ...] = ()


@dataclass(frozen=True)
class _PackedToken:
    value: int
    kind: str
    cluster_id: int | None
    source_id: str


class SequencePacker:
    """Append EOD, concatenate documents, and emit context+1 fixed records."""

    def __init__(self, context_length: int, *, final_partial_sequence_policy: str = "pad_eod") -> None:
        if context_length <= 0:
            raise ValueError("context_length must be positive")
        if final_partial_sequence_policy not in {"pad_eod", "error"}:
            raise ValueError("unsupported final partial sequence policy")
        self.context_length = context_length
        self.sequence_tokens = context_length + 1
        self.final_partial_sequence_policy = final_partial_sequence_policy
        self._tokens: list[_PackedToken] = []
        self._first_source_id: str | None = None
        self._last_source_id: str | None = None
        self._cluster_counts: dict[int, int] = {}
        self._has_overlap = False

    def push(self, document: SourceDocument) -> list[PackedSequence]:
        self._tokens.extend(
            _PackedToken(token, "source", document.cluster_id, document.source_id)
            for token in document.tokens
        )
        # Existing EOD is an original source token.  Only an EOD we insert is
        # non-source provenance; both cases remain exactly one document boundary.
        if not document.tokens or document.tokens[-1] != config.EOD_TOKEN_ID:
            self._tokens.append(_PackedToken(config.EOD_TOKEN_ID, "inserted_eod", None, document.source_id))
        self._first_source_id = self._first_source_id or document.source_id
        self._last_source_id = document.source_id
        self._cluster_counts[document.cluster_id] = self._cluster_counts.get(document.cluster_id, 0) + document.source_token_count
        return self._drain()

    def _drain(self) -> list[PackedSequence]:
        result: list[PackedSequence] = []
        while len(self._tokens) >= self.sequence_tokens:
            chunk = tuple(self._tokens[:self.sequence_tokens])
            counts: dict[int, int] = {}
            # The first token of every later record is the prior target.  It is
            # physically duplicated but is not a new source-token contribution.
            first_new = 1 if self._has_overlap else 0
            kinds: list[str] = []
            for index, entry in enumerate(chunk):
                kind = entry.kind
                if index == 0 and self._has_overlap:
                    kind = "overlap_source" if entry.kind == "source" else "overlap_eod"
                kinds.append(kind)
                if index >= first_new and entry.kind == "source" and entry.cluster_id is not None:
                    counts[entry.cluster_id] = counts.get(entry.cluster_id, 0) + 1
            # Advance by context length, retaining the target as the next input.
            del self._tokens[:self.context_length]
            assert self._first_source_id is not None and self._last_source_id is not None
            result.append(PackedSequence(
                tuple(entry.value for entry in chunk), self._first_source_id, self._last_source_id,
                counts, tuple(kinds), tuple(entry.cluster_id for entry in chunk),
            ))
            self._has_overlap = bool(self._tokens)
            self._cluster_counts = {}
            self._first_source_id = self._tokens[0].source_id if self._tokens else None
            self._last_source_id = self._tokens[-1].source_id if self._tokens else None
        return result

    def finish(self) -> list[PackedSequence]:
        if not self._tokens:
            return []
        if self.final_partial_sequence_policy == "error":
            raise RuntimeError("final partial sequence exists; choose pad_eod to preserve it")
        assert self._first_source_id is not None and self._last_source_id is not None
        self._tokens.extend(
            _PackedToken(config.EOD_TOKEN_ID, "padding", None, "<padding>")
            for _ in range(self.sequence_tokens - len(self._tokens))
        )
        result = self._drain()
        # The final emitted record retains its target by the normal stride rule,
        # but there is no following record in a completed cache.  Dropping only
        # this synthetic carry cannot remove a trainable source transition.
        self._tokens = []
        self._has_overlap = False
        self._first_source_id = self._last_source_id = None
        return result

    def state_dict(self) -> dict[str, object]:
        return {
            "context_length": self.context_length,
            "final_partial_sequence_policy": self.final_partial_sequence_policy,
            "carry": [entry.__dict__ for entry in self._tokens],
            # Kept for readers of pre-overlap checkpoint state; it is no longer
            # sufficient on its own because provenance is exact now.
            "carry_tokens": [entry.value for entry in self._tokens],
            "has_overlap": self._has_overlap,
            "first_source_id": self._first_source_id,
            "last_source_id": self._last_source_id,
            "cluster_source_tokens": {str(k): v for k, v in self._cluster_counts.items()},
        }

    @classmethod
    def from_state(cls, state: Mapping[str, object]) -> "SequencePacker":
        instance = cls(int(state["context_length"]), final_partial_sequence_policy=str(state["final_partial_sequence_policy"]))
        raw_carry = state.get("carry")
        if isinstance(raw_carry, list):
            instance._tokens = [
                _PackedToken(int(item["value"]), str(item["kind"]),
                             int(item["cluster_id"]) if item.get("cluster_id") is not None else None,
                             str(item["source_id"]))
                for item in raw_carry if isinstance(item, Mapping)
            ]
        else:
            instance._tokens = [_PackedToken(int(token), "source", None, "<legacy>") for token in state.get("carry_tokens", [])]
        if len(instance._tokens) >= instance.sequence_tokens:
            raise ValueError("invalid packer carry: it already contains a full sequence")
        instance._first_source_id = state.get("first_source_id") if isinstance(state.get("first_source_id"), str) else None
        instance._last_source_id = state.get("last_source_id") if isinstance(state.get("last_source_id"), str) else None
        raw = state.get("cluster_source_tokens", {})
        instance._cluster_counts = {int(k): int(v) for k, v in dict(raw).items()}
        instance._has_overlap = bool(state.get("has_overlap", False))
        return instance


@dataclass(frozen=True)
class PreparedBlock:
    block_id: int
    split: str
    sequence_count: int
    token_count: int
    payload: bytes
    cumulative_source_tokens: int
    per_cluster_source_tokens: dict[int, int]
    first_source_id: str
    last_source_id: str
    schema_version: int = STREAM_CACHE_SCHEMA_VERSION


class PreparedBlockBuilder:
    def __init__(self, sequences_per_block: int, block_id_counter: list[int] | None = None) -> None:
        self.sequences_per_block = sequences_per_block
        self._sequences: list[PackedSequence] = []
        self._block_id_counter = block_id_counter if block_id_counter is not None else [0]

    def push(self, sequence: PackedSequence, *, split: str, cumulative_source_tokens: int) -> PreparedBlock | None:
        self._sequences.append(sequence)
        if len(self._sequences) < self.sequences_per_block:
            return None
        return self._make(split, cumulative_source_tokens)

    def finish(self, *, split: str, cumulative_source_tokens: int) -> PreparedBlock | None:
        return self._make(split, cumulative_source_tokens) if self._sequences else None

    def _make(self, split: str, cumulative_source_tokens: int) -> PreparedBlock:
        sequences, self._sequences = self._sequences, []
        per_cluster: dict[int, int] = {}
        for sequence in sequences:
            for cluster, count in sequence.cluster_source_tokens.items():
                per_cluster[cluster] = per_cluster.get(cluster, 0) + count
        block_id = self._block_id_counter[0]
        self._block_id_counter[0] += 1
        block = PreparedBlock(
            block_id=block_id,
            split=split,
            sequence_count=len(sequences),
            token_count=sum(len(sequence.tokens) for sequence in sequences),
            payload=b"".join(tokens_to_uint16_le_bytes(sequence.tokens) for sequence in sequences),
            cumulative_source_tokens=cumulative_source_tokens,
            per_cluster_source_tokens=per_cluster,
            first_source_id=sequences[0].first_source_id,
            last_source_id=sequences[-1].last_source_id,
        )
        return block


@dataclass(frozen=True)
class ShardMetadata:
    filename: str
    split: str
    byte_size: int
    token_count: int
    sequence_count: int
    checksum: str
    first_block_id: int
    last_block_id: int
    context_length: int
    int_type: str
    byte_order: str
    cumulative_cluster_source_tokens: dict[int, int]
    shard_cluster_source_tokens: dict[int, int]

    def as_dict(self) -> dict[str, object]:
        return {
            **self.__dict__, "cumulative_cluster_source_tokens": {str(k): v for k, v in self.cumulative_cluster_source_tokens.items()},
            "shard_cluster_source_tokens": {str(k): v for k, v in self.shard_cluster_source_tokens.items()},
        }


class ImmutableShardWriter:
    """Write only temporary active files; finalized shards are never reopened."""

    def __init__(self, output_dir: Path, *, split: str, target_bytes: int, context_length: int, start_index: int | None = None) -> None:
        if split not in {"train", "validation"}:
            raise ValueError("split must be train or validation")
        self.directory = output_dir / split
        self.directory.mkdir(parents=True, exist_ok=True)
        self.split, self.target_bytes, self.context_length = split, target_bytes, context_length
        self.shards: list[ShardMetadata] = []
        if start_index is not None:
            self._index = start_index
        else:
            existing = [
                int(p.stem.split("-")[-1])
                for p in self.directory.glob(f"{split}-*.bin")
                if p.stem.split("-")[-1].isdigit()
            ]
            self._index = max(existing) + 1 if existing else 0
        self._handle = None
        self._blocks: list[PreparedBlock] = []
        self._cluster_counts: dict[int, int] = {}
        self._cumulative_counts: dict[int, int] = {}

    @property
    def active_path(self) -> Path:
        return self.directory / f".{self.split}-{self._index:06d}.bin.tmp"

    def write_block(self, block: PreparedBlock) -> list[ShardMetadata]:
        if block.split != self.split:
            raise ValueError("block split does not match shard writer")
        finalized: list[ShardMetadata] = []
        if self._blocks and self._byte_size() + len(block.payload) > self.target_bytes:
            finalized.append(self.finalize_active())
        if self._handle is None:
            if self.active_path.exists():
                self.active_path.unlink()
            self._handle = self.active_path.open("xb")
        written = self._handle.write(block.payload)
        if written != len(block.payload):
            raise OSError("short shard write")
        # The trainer must never receive bytes that are merely sitting in a
        # Python or kernel buffer.  The temp name is still active/mutable, but
        # this complete block is durable before ``StreamCacheProducer`` exposes
        # it to a consumer.  Finalization later supplies the immutable rename.
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._blocks.append(block)
        for cluster, count in block.per_cluster_source_tokens.items():
            self._cluster_counts[cluster] = self._cluster_counts.get(cluster, 0) + count
            self._cumulative_counts[cluster] = self._cumulative_counts.get(cluster, 0) + count
        return finalized

    def _byte_size(self) -> int:
        return sum(len(block.payload) for block in self._blocks)

    def finalize_active(self) -> ShardMetadata:
        if self._handle is None or not self._blocks:
            raise RuntimeError("no active shard to finalize")
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.close()
        final_name = f"{self.split}-{self._index:06d}.bin"
        final_path = self.directory / final_name
        os.replace(self.active_path, final_path)
        directory_fd = os.open(self.directory, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        metadata = ShardMetadata(
            filename=f"{self.split}/{final_name}", split=self.split, byte_size=final_path.stat().st_size,
            token_count=final_path.stat().st_size // 2,
            sequence_count=sum(block.sequence_count for block in self._blocks),
            checksum=sha256_file(final_path), first_block_id=self._blocks[0].block_id,
            last_block_id=self._blocks[-1].block_id, context_length=self.context_length,
            int_type=config.INT_TYPE, byte_order=config.BYTE_ORDER,
            cumulative_cluster_source_tokens=dict(self._cumulative_counts),
            shard_cluster_source_tokens=dict(self._cluster_counts),
        )
        self.shards.append(metadata)
        self._index += 1
        self._handle = None
        self._blocks = []
        self._cluster_counts = {}
        return metadata

    def close(self) -> None:
        if self._handle is not None and not self._handle.closed:
            self._handle.close()
            self._handle = None

    def __del__(self) -> None:
        self.close()

    def finish(self) -> list[ShardMetadata]:
        return [self.finalize_active()] if self._blocks else []


class BlockConsumer(Protocol):
    def submit(self, block: PreparedBlock) -> None: ...


class QueueConsumer:
    """Bounded trainer-facing queue.  Consumers acknowledge stable block IDs."""

    def __init__(self, maxsize: int) -> None:
        self.queue: queue.Queue[PreparedBlock] = queue.Queue(maxsize=maxsize)
        self.last_acknowledged_block_id = -1

    def submit(self, block: PreparedBlock) -> None:
        self.queue.put(block)  # intentional backpressure after cache durability

    def acknowledge(self, block_id: int) -> None:
        if block_id < self.last_acknowledged_block_id:
            raise ValueError("consumer acknowledgements must be monotonic")
        self.last_acknowledged_block_id = block_id


@dataclass(frozen=True)
class DocumentBatch:
    """A bounded, deterministic unit emitted by the parallel source reader."""

    work_item_index: int
    records: tuple[tuple[bool, SourceDocument], ...]
    accepted_source_tokens: int
    estimated_bytes: int


class StreamCacheProducer:
    """Schedule documents, pack them, durably shard, then hand blocks to a consumer."""

    def __init__(
        self,
        output_dir: Path,
        stream_config: StreamCacheConfig,
        consumer: BlockConsumer | None = None,
        block_id_counter: list[int] | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.config = stream_config
        self.scheduler = TokenDeficitScheduler(stream_config.weights, stream_config.scheduler_tie_break_seed)
        self.consumer = consumer or QueueConsumer(stream_config.prepared_block_queue_limit)
        self.train_packer = SequencePacker(stream_config.context_length, final_partial_sequence_policy=stream_config.final_partial_sequence_policy)
        self.validation_packer = SequencePacker(stream_config.context_length, final_partial_sequence_policy=stream_config.final_partial_sequence_policy)
        self.block_id_counter = block_id_counter if block_id_counter is not None else [0]
        self.train_blocks = PreparedBlockBuilder(stream_config.sequences_per_block, block_id_counter=self.block_id_counter)
        self.validation_blocks = PreparedBlockBuilder(stream_config.sequences_per_block, block_id_counter=self.block_id_counter)
        self.train_writer = ImmutableShardWriter(output_dir, split="train", target_bytes=stream_config.target_shard_bytes, context_length=stream_config.context_length)
        self.validation_writer = ImmutableShardWriter(output_dir, split="validation", target_bytes=stream_config.target_shard_bytes, context_length=stream_config.context_length)
        self.last_durable_block_id = -1
        self._queues = {cluster: deque() for cluster in config.ACCEPTED_CLUSTER_IDS}
        self._documents_since_last_schedule = 0
        self._rolling_documents: deque[SourceDocument] = deque()
        self.validation_source_tokens = 0

    def add_training_document(self, document: SourceDocument) -> None:
        if document.cluster_id not in self._queues:
            raise ValueError("training document has excluded cluster")
        bucket = self._queues[document.cluster_id]
        if len(bucket) >= self.config.per_cluster_queue_limit:
            raise RuntimeError(f"cluster {document.cluster_id} queue is full")
        bucket.append(document)
        self._documents_since_last_schedule += 1

    def _queue_source_tokens(self) -> int:
        return sum(document.source_token_count for documents in self._queues.values() for document in documents)

    def _mixture_error(self, documents: Iterable[SourceDocument]) -> float:
        counts = {cluster: 0 for cluster in self.scheduler.weight_units}
        total = 0
        for document in documents:
            counts[document.cluster_id] += document.source_token_count
            total += document.source_token_count
        if not total:
            return 0.0
        return max(abs(counts[c] / total - self.scheduler.weight_units[c] / self.scheduler.weight_total) for c in counts)

    def mixture_measurements(self) -> dict[str, object]:
        recent = list(self._rolling_documents)
        return {
            "cumulative_error": self._mixture_error(
                [SourceDocument("<count>", c, tuple([0]) * n)
                 for c, n in self.scheduler.emitted_source_tokens.items() if n]
            ),
            "rolling_error": self._mixture_error(recent),
            "rolling_windows": {
                str(window): self._mixture_error(_tail_documents_by_tokens(recent, window))
                for window in self.config.rolling_mixture_windows
            },
        }

    def _ready_to_schedule(self, *, force: bool) -> bool:
        populated = sum(bool(documents) for documents in self._queues.values())
        if not populated:
            return False
        if force:
            return True
        enough_coverage = populated >= self.config.minimum_populated_cluster_queues
        enough_tokens = self._queue_source_tokens() >= self.config.minimum_prefetched_source_tokens
        waited_long_enough = self._documents_since_last_schedule >= self.config.maximum_waiting_documents
        return (enough_coverage and enough_tokens) or waited_long_enough

    def drain_training(self, *, force: bool = True, maximum_documents: int | None = None) -> int:
        """Incrementally drain ready queues without requiring every cluster.

        A normal producer call emits at most one document once the prefetch head
        start is satisfied.  A full drain is reserved for finalization or an
        explicit queue-pressure escape hatch.  This prevents an early, lone
        queue from being emptied into a long one-cluster run.
        """
        emitted = 0
        while self._ready_to_schedule(force=force):
            cluster = self.scheduler.choose(cluster for cluster, docs in self._queues.items() if docs)
            if cluster is None:
                return emitted
            document = self._queues[cluster].popleft()
            self.scheduler.emit(document)
            self._rolling_documents.append(document)
            # Keep sufficient history for the largest configured measurement.
            while sum(d.source_token_count for d in self._rolling_documents) > max(self.config.rolling_mixture_windows):
                self._rolling_documents.popleft()
            self._documents_since_last_schedule = 0
            emitted += 1
            for sequence in self.train_packer.push(document):
                block = self.train_blocks.push(sequence, split="train", cumulative_source_tokens=self.scheduler.total_emitted_source_tokens)
                if block is not None:
                    self._publish(block)
            if maximum_documents is not None and emitted >= maximum_documents:
                break
            if not force:
                break
        return emitted

    def add_validation_document(self, document: SourceDocument) -> None:
        self.validation_source_tokens += document.source_token_count
        for sequence in self.validation_packer.push(document):
            block = self.validation_blocks.push(sequence, split="validation", cumulative_source_tokens=self.scheduler.total_emitted_source_tokens)
            if block is not None:
                self._publish(block)

    def _publish(self, block: PreparedBlock) -> None:
        writer = self.train_writer if block.split == "train" else self.validation_writer
        writer.write_block(block)
        # ``write_block`` flushes and fsyncs the complete block before this
        # queue operation.  It is not yet immutable until shard finalization.
        self.consumer.submit(block)
        self.last_durable_block_id = block.block_id

    def finalize_active_shards_for_checkpoint(self) -> list[ShardMetadata]:
        """Turn both active tails into immutable, remotely publishable shards.

        This is intentionally legal at a trainer optimizer boundary only; the
        joint coordinator calls it before constructing a checkpoint manifest so
        that a remote checkpoint never references a mutable `.tmp` tail.
        """
        return self.train_writer.finish() + self.validation_writer.finish()

    def finish(self) -> dict[str, object]:
        self.drain_training(force=True)
        for sequence in self.train_packer.finish():
            block = self.train_blocks.push(sequence, split="train", cumulative_source_tokens=self.scheduler.total_emitted_source_tokens)
            if block is not None:
                self._publish(block)
        for sequence in self.validation_packer.finish():
            block = self.validation_blocks.push(sequence, split="validation", cumulative_source_tokens=self.scheduler.total_emitted_source_tokens)
            if block is not None:
                self._publish(block)
        for builder, split in ((self.train_blocks, "train"), (self.validation_blocks, "validation")):
            block = builder.finish(split=split, cumulative_source_tokens=self.scheduler.total_emitted_source_tokens)
            if block is not None:
                self._publish(block)
        self.finalize_active_shards_for_checkpoint()
        manifest = {
            "schema_version": STREAM_CACHE_SCHEMA_VERSION,
            "sequence_format": SEQUENCE_FORMAT,
            "context_length": self.config.context_length,
            "stored_tokens_per_sequence": self.config.stored_sequence_tokens,
            "final_partial_sequence_policy": self.config.final_partial_sequence_policy,
            "weights": {"supplied": {str(k): v for k, v in self.scheduler.supplied_weights.items()}, "normalized_integer_units": {str(k): v for k, v in self.scheduler.weight_units.items()}},
            "shards": [item.as_dict() for item in self.train_writer.shards + self.validation_writer.shards],
            "scheduler": self.scheduler.state_dict(),
            "accepted_source_tokens": self.scheduler.total_emitted_source_tokens + self.validation_source_tokens,
            "validation_source_tokens": self.validation_source_tokens,
            "mixture": self.mixture_measurements(),
            "last_durable_block_id": self.last_durable_block_id,
            "consumer_visibility": "after the complete active-shard block is flush+fsync durable; finalization is fsync+atomic rename",
            "resume_replay": "Joint checkpoints replay only work after last_consumed_block_id; pipeline state is serialized separately.",
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(self.output_dir / config.MANIFEST_FILENAME, manifest)
        write_json_atomic(self.output_dir / config.PROGRESS_FILENAME, self.checkpoint_state())
        return manifest

    def checkpoint_state(self) -> dict[str, object]:
        return {
            "schema_version": STREAM_CACHE_SCHEMA_VERSION,
            "scheduler": self.scheduler.state_dict(),
            "train_packer": self.train_packer.state_dict(),
            "validation_packer": self.validation_packer.state_dict(),
            "per_cluster_queue_source_ids": {str(k): [d.source_id for d in v] for k, v in self._queues.items()},
            "per_cluster_queues": {
                str(k): [_document_to_dict(d) for d in v] for k, v in self._queues.items()
            },
            "rolling_documents": [_document_to_dict(d) for d in self._rolling_documents],
            "documents_since_last_schedule": self._documents_since_last_schedule,
            "validation_source_tokens": self.validation_source_tokens,
            "last_durable_block_id": self.last_durable_block_id,
            "last_consumer_acknowledged_block_id": getattr(self.consumer, "last_acknowledged_block_id", -1),
            "finalized_shards": [item.as_dict() for item in self.train_writer.shards + self.validation_writer.shards],
        }

    def close(self) -> None:
        self.train_writer.close()
        self.validation_writer.close()

    @classmethod
    def from_state(
        cls,
        output_dir: Path,
        stream_config: StreamCacheConfig,
        state: Mapping[str, object],
        consumer: BlockConsumer | None = None,
    ) -> StreamCacheProducer:
        last_durable = int(state["last_durable_block_id"])
        next_block_id = last_durable + 1 if last_durable >= 0 else 0
        instance = cls(
            output_dir,
            stream_config,
            consumer=consumer,
            block_id_counter=[next_block_id],
        )
        if "scheduler" in state and isinstance(state["scheduler"], Mapping):
            instance.scheduler = TokenDeficitScheduler.from_state(stream_config.weights, state["scheduler"])
        if "train_packer" in state and isinstance(state["train_packer"], Mapping):
            instance.train_packer = SequencePacker.from_state(state["train_packer"])
        if "validation_packer" in state and isinstance(state["validation_packer"], Mapping):
            instance.validation_packer = SequencePacker.from_state(state["validation_packer"])
        instance.last_durable_block_id = last_durable
        raw_queues = state.get("per_cluster_queues", {})
        if isinstance(raw_queues, Mapping):
            for raw_cluster, raw_documents in raw_queues.items():
                cluster = int(raw_cluster)
                if cluster in instance._queues and isinstance(raw_documents, list):
                    instance._queues[cluster].extend(_document_from_dict(item) for item in raw_documents if isinstance(item, Mapping))
        raw_rolling = state.get("rolling_documents", [])
        if isinstance(raw_rolling, list):
            instance._rolling_documents.extend(_document_from_dict(item) for item in raw_rolling if isinstance(item, Mapping))
        instance._documents_since_last_schedule = int(state.get("documents_since_last_schedule", 0))
        instance.validation_source_tokens = int(state.get("validation_source_tokens", 0))
        last_ack = state.get("last_consumer_acknowledged_block_id")
        if last_ack is not None and hasattr(instance.consumer, "acknowledge") and int(last_ack) >= 0:
            instance.consumer.acknowledge(int(last_ack))
        return instance


def _document_to_dict(document: SourceDocument) -> dict[str, object]:
    return {"source_id": document.source_id, "cluster_id": document.cluster_id,
            "tokens": list(document.tokens), "work_item_index": document.work_item_index,
            "record_start": document.record_start}


def _document_from_dict(data: Mapping[str, object]) -> SourceDocument:
    return SourceDocument(str(data["source_id"]), int(data["cluster_id"]),
                          tuple(int(token) for token in data["tokens"]),
                          int(data.get("work_item_index", 0)), int(data.get("record_start", 0)))


def _tail_documents_by_tokens(documents: Sequence[SourceDocument], window: int) -> list[SourceDocument]:
    total = 0
    selected: list[SourceDocument] = []
    for document in reversed(documents):
        selected.append(document)
        total += document.source_token_count
        if total >= window:
            break
    return list(reversed(selected))


def parallel_read_document_batches(
    plan: WorkPlan,
    *,
    reader_factory: Callable[[SourceFile], RangeReader],
    workers: int,
    max_in_flight: int,
    validation_probability: float = config.VALIDATION_PROBABILITY,
    maximum_source_tokens_per_batch: int = 1_000_000,
    maximum_documents_per_batch: int = 1_000,
    maximum_bytes_per_batch: int = 16 * 1024 * 1024,
) -> Iterator[DocumentBatch]:
    """Yield bounded batches in work-plan order with bounded worker channels."""

    if workers <= 0 or max_in_flight <= 0:
        raise ValueError("workers and max_in_flight must be positive")
    files = {source.path: source for source in plan.source_files}

    if min(maximum_source_tokens_per_batch, maximum_documents_per_batch, maximum_bytes_per_batch) <= 0:
        raise ValueError("batch limits must be positive")
    stop = threading.Event()
    finished = object()

    def put(channel: queue.Queue[object], item: object) -> bool:
        while not stop.is_set():
            try:
                channel.put(item, timeout=0.05)
                return True
            except queue.Full:
                continue
        return False

    def read_item(item: WorkItem, channel: queue.Queue[object]) -> None:
        reader = reader_factory(files[item.filename])
        records: list[tuple[bool, SourceDocument]] = []
        source_tokens = estimated_bytes = 0
        try:
            for record in iter_owned_records(item, reader):
                validated = validate_record(record)
                if not validated.valid or validated.cluster_id not in config.ACCEPTED_CLUSTER_IDS:
                    continue
                assert validated.tokens is not None
                document = SourceDocument(
                    f"{plan.revision}:{item.filename}:{record.record_start}", validated.cluster_id,
                    tuple(validated.tokens), item.index, record.record_start,
                )
                projected_tokens = source_tokens + document.source_token_count
                projected_bytes = estimated_bytes + len(record.raw) + document.source_token_count * 2
                if records and (len(records) >= maximum_documents_per_batch or
                                projected_tokens > maximum_source_tokens_per_batch or
                                projected_bytes > maximum_bytes_per_batch):
                    if not put(channel, DocumentBatch(item.index, tuple(records), source_tokens, estimated_bytes)):
                        return
                    records, source_tokens, estimated_bytes = [], 0, 0
                records.append((is_validation(seed=config.SELECTION_SEED, revision=plan.revision,
                                              filename=item.filename, record_start=record.record_start,
                                              probability=validation_probability), document))
                source_tokens += document.source_token_count
                estimated_bytes += len(record.raw) + document.source_token_count * 2
            if records:
                put(channel, DocumentBatch(item.index, tuple(records), source_tokens, estimated_bytes))
            put(channel, finished)
        except BaseException as error:  # future consumer must see reader failures
            put(channel, error)

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="source-reader") as pool:
        futures: dict[int, tuple[Future[None], queue.Queue[object]]] = {}
        next_submit = 0
        next_yield = 0
        while next_yield < len(plan.work_items):
            while next_submit < len(plan.work_items) and len(futures) < max_in_flight:
                item = plan.work_items[next_submit]
                channel: queue.Queue[object] = queue.Queue(maxsize=1)
                futures[next_submit] = (pool.submit(read_item, item, channel), channel)
                next_submit += 1
            future, channel = futures.pop(next_yield)
            while True:
                item = channel.get()
                if item is finished:
                    future.result()
                    break
                if isinstance(item, BaseException):
                    stop.set()
                    for pending, _ in futures.values():
                        pending.cancel()
                    raise item
                assert isinstance(item, DocumentBatch)
                yield item
            next_yield += 1
        stop.set()


def parallel_read_documents(
    plan: WorkPlan,
    *,
    reader_factory: Callable[[SourceFile], RangeReader],
    workers: int,
    max_in_flight: int,
    validation_probability: float = config.VALIDATION_PROBABILITY,
    maximum_source_tokens_per_batch: int = 1_000_000,
    maximum_documents_per_batch: int = 1_000,
    maximum_bytes_per_batch: int = 16 * 1024 * 1024,
) -> Iterator[tuple[bool, SourceDocument]]:
    """Compatibility flattening wrapper over the memory-bounded batch API."""
    for batch in parallel_read_document_batches(
        plan, reader_factory=reader_factory, workers=workers, max_in_flight=max_in_flight,
        validation_probability=validation_probability,
        maximum_source_tokens_per_batch=maximum_source_tokens_per_batch,
        maximum_documents_per_batch=maximum_documents_per_batch,
        maximum_bytes_per_batch=maximum_bytes_per_batch,
    ):
        yield from batch.records


def build_stream_cache(
    output_dir: Path | str,
    stream_config: StreamCacheConfig,
    plan: WorkPlan,
    reader_factory: Callable[[SourceFile], RangeReader],
    consumer: BlockConsumer | None = None,
) -> dict[str, object]:
    """Execute the bounded streaming-cache adapter layer over a WorkPlan.

    Reads source documents concurrently in plan order using parallel_read_documents,
    routes validation documents directly, applies bounded per-cluster queues with
    deterministic draining/backpressure for training documents, finishes cache shards,
    and returns the finalized manifest.
    """

    output_dir = Path(output_dir)
    producer = StreamCacheProducer(
        output_dir,
        stream_config,
        consumer=consumer,
    )
    try:
        for is_validation_doc, document in parallel_read_documents(
            plan,
            reader_factory=reader_factory,
            workers=stream_config.reader_workers,
            max_in_flight=stream_config.max_in_flight_work_items,
            maximum_source_tokens_per_batch=stream_config.reader_batch_source_tokens,
            maximum_documents_per_batch=stream_config.reader_batch_documents,
            maximum_bytes_per_batch=stream_config.reader_batch_max_bytes,
        ):
            if is_validation_doc:
                producer.add_validation_document(document)
            else:
                if len(producer._queues[document.cluster_id]) >= stream_config.per_cluster_queue_limit:
                    producer.drain_training(force=True, maximum_documents=1)
                producer.add_training_document(document)
                producer.drain_training(force=False, maximum_documents=1)

        return producer.finish()
    finally:
        producer.close()
