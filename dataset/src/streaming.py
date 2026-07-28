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
        self._tokens: list[int] = []
        self._first_source_id: str | None = None
        self._last_source_id: str | None = None
        self._cluster_counts: dict[int, int] = {}

    def push(self, document: SourceDocument) -> list[PackedSequence]:
        tokens = list(document.tokens)
        if not tokens or tokens[-1] != config.EOD_TOKEN_ID:
            tokens.append(config.EOD_TOKEN_ID)
        self._tokens.extend(tokens)
        self._first_source_id = self._first_source_id or document.source_id
        self._last_source_id = document.source_id
        self._cluster_counts[document.cluster_id] = self._cluster_counts.get(document.cluster_id, 0) + document.source_token_count
        return self._drain()

    def _drain(self) -> list[PackedSequence]:
        result: list[PackedSequence] = []
        while len(self._tokens) >= self.sequence_tokens:
            chunk = tuple(self._tokens[:self.sequence_tokens])
            del self._tokens[:self.sequence_tokens]
            assert self._first_source_id is not None and self._last_source_id is not None
            result.append(PackedSequence(chunk, self._first_source_id, self._last_source_id, dict(self._cluster_counts)))
            self._cluster_counts = {}
            # Provenance is intentionally coarse for a carry that spans a sequence:
            # first/last identities remain conservative and never lose coverage.
            self._first_source_id = self._last_source_id if self._tokens else None
            if not self._tokens:
                self._last_source_id = None
        return result

    def finish(self) -> list[PackedSequence]:
        if not self._tokens:
            return []
        if self.final_partial_sequence_policy == "error":
            raise RuntimeError("final partial sequence exists; choose pad_eod to preserve it")
        assert self._first_source_id is not None and self._last_source_id is not None
        self._tokens.extend([config.EOD_TOKEN_ID] * (self.sequence_tokens - len(self._tokens)))
        result = self._drain()
        assert not self._tokens
        return result

    def state_dict(self) -> dict[str, object]:
        return {
            "context_length": self.context_length,
            "final_partial_sequence_policy": self.final_partial_sequence_policy,
            "carry_tokens": list(self._tokens),
            "first_source_id": self._first_source_id,
            "last_source_id": self._last_source_id,
            "cluster_source_tokens": {str(k): v for k, v in self._cluster_counts.items()},
        }

    @classmethod
    def from_state(cls, state: Mapping[str, object]) -> "SequencePacker":
        instance = cls(int(state["context_length"]), final_partial_sequence_policy=str(state["final_partial_sequence_policy"]))
        instance._tokens = [int(token) for token in state.get("carry_tokens", [])]
        if len(instance._tokens) >= instance.sequence_tokens:
            raise ValueError("invalid packer carry: it already contains a full sequence")
        instance._first_source_id = state.get("first_source_id") if isinstance(state.get("first_source_id"), str) else None
        instance._last_source_id = state.get("last_source_id") if isinstance(state.get("last_source_id"), str) else None
        raw = state.get("cluster_source_tokens", {})
        instance._cluster_counts = {int(k): int(v) for k, v in dict(raw).items()}
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

    def add_training_document(self, document: SourceDocument) -> None:
        if document.cluster_id not in self._queues:
            raise ValueError("training document has excluded cluster")
        bucket = self._queues[document.cluster_id]
        if len(bucket) >= self.config.per_cluster_queue_limit:
            raise RuntimeError(f"cluster {document.cluster_id} queue is full")
        bucket.append(document)

    def drain_training(self) -> None:
        while True:
            cluster = self.scheduler.choose(cluster for cluster, docs in self._queues.items() if docs)
            if cluster is None:
                return
            document = self._queues[cluster].popleft()
            self.scheduler.emit(document)
            for sequence in self.train_packer.push(document):
                block = self.train_blocks.push(sequence, split="train", cumulative_source_tokens=self.scheduler.total_emitted_source_tokens)
                if block is not None:
                    self._publish(block)

    def add_validation_document(self, document: SourceDocument) -> None:
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

    def finish(self) -> dict[str, object]:
        self.drain_training()
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
        self.train_writer.finish()
        self.validation_writer.finish()
        manifest = {
            "schema_version": STREAM_CACHE_SCHEMA_VERSION,
            "sequence_format": SEQUENCE_FORMAT,
            "context_length": self.config.context_length,
            "stored_tokens_per_sequence": self.config.stored_sequence_tokens,
            "final_partial_sequence_policy": self.config.final_partial_sequence_policy,
            "weights": {"supplied": {str(k): v for k, v in self.scheduler.supplied_weights.items()}, "normalized_integer_units": {str(k): v for k, v in self.scheduler.weight_units.items()}},
            "shards": [item.as_dict() for item in self.train_writer.shards + self.validation_writer.shards],
            "scheduler": self.scheduler.state_dict(),
            "last_durable_block_id": self.last_durable_block_id,
            "consumer_visibility": "after the complete active-shard block is flush+fsync durable; finalization is fsync+atomic rename",
            "resume_replay": "A future joint checkpoint may replay only blocks after last_durable_block_id; no GPU atomicity is claimed.",
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
        last_ack = state.get("last_consumer_acknowledged_block_id")
        if last_ack is not None and hasattr(instance.consumer, "acknowledge") and int(last_ack) >= 0:
            instance.consumer.acknowledge(int(last_ack))
        return instance


def parallel_read_documents(
    plan: WorkPlan,
    *,
    reader_factory: Callable[[SourceFile], RangeReader],
    workers: int,
    max_in_flight: int,
    validation_probability: float = config.VALIDATION_PROBABILITY,
) -> Iterator[tuple[bool, SourceDocument]]:
    """Fetch bounded work items concurrently but yield deterministic plan order.

    Results are held only for the configured in-flight window.  Future
    completion order cannot leak into document order because every future is
    consumed by increasing work-plan index.
    """

    if workers <= 0 or max_in_flight <= 0:
        raise ValueError("workers and max_in_flight must be positive")
    files = {source.path: source for source in plan.source_files}

    def read_item(item: WorkItem) -> list[tuple[bool, SourceDocument]]:
        reader = reader_factory(files[item.filename])
        output: list[tuple[bool, SourceDocument]] = []
        for record in iter_owned_records(item, reader):
            validated = validate_record(record)
            if not validated.valid or validated.cluster_id not in config.ACCEPTED_CLUSTER_IDS:
                continue
            assert validated.tokens is not None
            source_id = f"{plan.revision}:{item.filename}:{record.record_start}"
            document = SourceDocument(source_id, validated.cluster_id, tuple(validated.tokens), item.index, record.record_start)
            output.append((is_validation(seed=config.SELECTION_SEED, revision=plan.revision, filename=item.filename, record_start=record.record_start, probability=validation_probability), document))
        return output

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="source-reader") as pool:
        futures: dict[int, Future[list[tuple[bool, SourceDocument]]]] = {}
        next_submit = 0
        next_yield = 0
        while next_yield < len(plan.work_items):
            while next_submit < len(plan.work_items) and len(futures) < max_in_flight:
                item = plan.work_items[next_submit]
                futures[next_submit] = pool.submit(read_item, item)
                next_submit += 1
            future = futures.pop(next_yield)
            # result() deliberately propagates the original reader exception and
            # executor shutdown cancels not-yet-started work.
            yield from future.result()
            next_yield += 1


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
        ):
            if is_validation_doc:
                producer.add_validation_document(document)
            else:
                if len(producer._queues[document.cluster_id]) >= stream_config.per_cluster_queue_limit:
                    producer.drain_training()
                producer.add_training_document(document)

        return producer.finish()
    finally:
        producer.close()

