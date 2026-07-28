"""Deterministic first-pass streaming cache primitives.

This module is deliberately independent from a model framework.  It turns
validated source documents into fixed-geometry blocks, makes every block durable
in an immutable shard, and then exposes the *same bytes* to a consumer.  The
legacy ``build`` command remains available for the old monolithic format; this
is the schema-v2 streaming-cache path.

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
from typing import Callable, Iterable, Iterator, Mapping, Protocol

from dataset import config

from .bitio import tokens_to_uint16_le_bytes
from .bytesource import RangeReader, SourceFile
from .manifest import sha256_file
from .records import iter_owned_records, validate_record
from .split import is_validation
from .storage import write_json_atomic
from .workplan import WorkItem, WorkPlan


STREAM_CACHE_SCHEMA_VERSION = 2
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
        if not math.isfinite(self.maximum_rolling_mixture_error) or self.maximum_rolling_mixture_error < 0:
            raise ValueError("maximum_rolling_mixture_error must be a finite non-negative number")
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


def _packed_sequence_to_dict(sequence: PackedSequence) -> dict[str, object]:
    """Return the JSON-safe representation used by prepared-block state."""

    return {
        "tokens": list(sequence.tokens),
        "first_source_id": sequence.first_source_id,
        "last_source_id": sequence.last_source_id,
        "cluster_source_tokens": {
            str(cluster): count for cluster, count in sequence.cluster_source_tokens.items()
        },
        "token_kinds": list(sequence.token_kinds),
        "token_clusters": list(sequence.token_clusters),
    }


def _packed_sequence_from_dict(data: Mapping[str, object]) -> PackedSequence:
    """Decode and validate one pending prepared sequence."""

    raw_tokens = data.get("tokens")
    if not isinstance(raw_tokens, (list, tuple)) or not raw_tokens:
        raise ValueError("prepared-block state has invalid sequence tokens")
    tokens: list[int] = []
    for raw_token in raw_tokens:
        if isinstance(raw_token, bool) or not isinstance(raw_token, int):
            raise ValueError("prepared-block sequence tokens must be integers")
        token = raw_token
        if token < 0 or token > config.TOKEN_MAX:
            raise ValueError("prepared-block sequence token is outside uint16 range")
        tokens.append(token)

    first_source_id = data.get("first_source_id")
    last_source_id = data.get("last_source_id")
    if not isinstance(first_source_id, str) or not isinstance(last_source_id, str):
        raise ValueError("prepared-block state has invalid source identities")

    raw_counts = data.get("cluster_source_tokens")
    if not isinstance(raw_counts, Mapping):
        raise ValueError("prepared-block state has invalid cluster counts")
    cluster_counts: dict[int, int] = {}
    for raw_cluster, raw_count in raw_counts.items():
        if isinstance(raw_cluster, bool) or isinstance(raw_count, bool):
            raise ValueError("prepared-block cluster counts must be integers")
        try:
            cluster = int(raw_cluster)
            count = int(raw_count)
        except (TypeError, ValueError) as error:
            raise ValueError("prepared-block cluster counts must be integers") from error
        if cluster in cluster_counts:
            raise ValueError("prepared-block state repeats a cluster count")
        if count < 0:
            raise ValueError("prepared-block cluster counts cannot be negative")
        cluster_counts[cluster] = count

    raw_kinds = data.get("token_kinds", [])
    raw_clusters = data.get("token_clusters", [])
    if not isinstance(raw_kinds, (list, tuple)) or not isinstance(raw_clusters, (list, tuple)):
        raise ValueError("prepared-block state has invalid token provenance")
    if bool(raw_kinds) != bool(raw_clusters):
        raise ValueError("prepared-block token provenance fields must be present together")
    if raw_kinds and len(raw_kinds) != len(tokens):
        raise ValueError("prepared-block token kinds do not match token count")
    if raw_clusters and len(raw_clusters) != len(tokens):
        raise ValueError("prepared-block token clusters do not match token count")
    allowed_kinds = {"source", "inserted_eod", "overlap_source", "overlap_eod", "padding"}
    if any(not isinstance(kind, str) or kind not in allowed_kinds for kind in raw_kinds):
        raise ValueError("prepared-block state has an invalid token kind")
    token_kinds = tuple(raw_kinds)
    token_clusters: list[int | None] = []
    for raw_cluster in raw_clusters:
        if raw_cluster is None:
            token_clusters.append(None)
        elif isinstance(raw_cluster, bool) or not isinstance(raw_cluster, int):
            raise ValueError("prepared-block token clusters must be integers or null")
        else:
            token_clusters.append(raw_cluster)
    return PackedSequence(
        tuple(tokens), first_source_id, last_source_id, cluster_counts,
        token_kinds, tuple(token_clusters),
    )


class PreparedBlockBuilder:
    def __init__(self, sequences_per_block: int, block_id_counter: list[int] | None = None) -> None:
        if isinstance(sequences_per_block, bool) or sequences_per_block <= 0:
            raise ValueError("sequences_per_block must be positive")
        self.sequences_per_block = sequences_per_block
        self._sequences: list[PackedSequence] = []
        self._block_id_counter = block_id_counter if block_id_counter is not None else [0]
        if (
            not isinstance(self._block_id_counter, list)
            or len(self._block_id_counter) != 1
            or isinstance(self._block_id_counter[0], bool)
            or not isinstance(self._block_id_counter[0], int)
            or self._block_id_counter[0] < 0
        ):
            raise ValueError("block_id_counter must contain one non-negative integer")

    def push(self, sequence: PackedSequence, *, split: str, cumulative_source_tokens: int) -> PreparedBlock | None:
        self._sequences.append(sequence)
        if len(self._sequences) < self.sequences_per_block:
            return None
        return self._make(split, cumulative_source_tokens)

    def finish(self, *, split: str, cumulative_source_tokens: int) -> PreparedBlock | None:
        return self._make(split, cumulative_source_tokens) if self._sequences else None

    def state_dict(self) -> dict[str, object]:
        next_block_id = self._block_id_counter[0]
        if isinstance(next_block_id, bool) or not isinstance(next_block_id, int) or next_block_id < 0:
            raise ValueError("prepared-block builder has an invalid next block ID")
        if len(self._sequences) >= self.sequences_per_block:
            raise ValueError("prepared-block builder has too many pending sequences")
        return {
            "sequences_per_block": self.sequences_per_block,
            "pending_sequences": [_packed_sequence_to_dict(sequence) for sequence in self._sequences],
            "next_block_id": next_block_id,
        }

    @classmethod
    def from_state(cls, state: Mapping[str, object]) -> "PreparedBlockBuilder":
        if not isinstance(state, Mapping):
            raise ValueError("prepared-block builder state must be a mapping")
        raw_size = state.get("sequences_per_block")
        raw_next = state.get("next_block_id")
        if isinstance(raw_size, bool) or not isinstance(raw_size, int) or raw_size <= 0:
            raise ValueError("prepared-block builder state has invalid block size")
        if isinstance(raw_next, bool) or not isinstance(raw_next, int) or raw_next < 0:
            raise ValueError("prepared-block builder state has invalid next block ID")
        raw_pending = state.get("pending_sequences")
        if not isinstance(raw_pending, list) or len(raw_pending) >= raw_size:
            raise ValueError("prepared-block builder state has invalid pending sequence count")
        instance = cls(raw_size, block_id_counter=[raw_next])
        for raw_sequence in raw_pending:
            if not isinstance(raw_sequence, Mapping):
                raise ValueError("prepared-block builder state has an invalid pending sequence")
            instance._sequences.append(_packed_sequence_from_dict(raw_sequence))
        return instance

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
                raise RuntimeError(
                    f"mutable active shard already exists and is not recoverable: {self.active_path}"
                )
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


class NullBlockConsumer:
    """No-op sink for cache-only builds; submitted blocks are auto-acknowledged."""

    def __init__(self) -> None:
        self.last_acknowledged_block_id = -1

    def submit(self, block: PreparedBlock) -> None:
        self.last_acknowledged_block_id = block.block_id

    def acknowledge(self, block_id: int) -> None:
        if block_id < self.last_acknowledged_block_id:
            raise ValueError("consumer acknowledgements must be monotonic")
        self.last_acknowledged_block_id = block_id


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


@dataclass(frozen=True)
class _RollingContribution:
    """One compact source-token contribution in the rolling history."""

    cluster_id: int
    token_count: int


class StreamCacheProducer:
    """Schedule documents, pack them, durably shard, then hand blocks to a consumer."""

    def __init__(
        self,
        output_dir: Path,
        stream_config: StreamCacheConfig,
        consumer: BlockConsumer | None = None,
        block_id_counter: list[int] | None = None,
        validation_consumer: BlockConsumer | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.config = stream_config
        self.scheduler = TokenDeficitScheduler(stream_config.weights, stream_config.scheduler_tie_break_seed)
        self.consumer = consumer if consumer is not None else NullBlockConsumer()
        self.validation_consumer = validation_consumer
        self.train_packer = SequencePacker(stream_config.context_length, final_partial_sequence_policy=stream_config.final_partial_sequence_policy)
        self.validation_packer = SequencePacker(stream_config.context_length, final_partial_sequence_policy=stream_config.final_partial_sequence_policy)
        self.train_block_id_counter = block_id_counter if block_id_counter is not None else [0]
        self.validation_block_id_counter = [0]
        # Retain the old attribute as the trainer-facing counter.  Validation
        # blocks deliberately live in their own ID namespace.
        self.block_id_counter = self.train_block_id_counter
        self.train_blocks = PreparedBlockBuilder(
            stream_config.sequences_per_block, block_id_counter=self.train_block_id_counter
        )
        self.validation_blocks = PreparedBlockBuilder(
            stream_config.sequences_per_block, block_id_counter=self.validation_block_id_counter
        )
        self.train_writer = ImmutableShardWriter(output_dir, split="train", target_bytes=stream_config.target_shard_bytes, context_length=stream_config.context_length)
        self.validation_writer = ImmutableShardWriter(output_dir, split="validation", target_bytes=stream_config.target_shard_bytes, context_length=stream_config.context_length)
        self.last_durable_block_id = -1
        self.last_durable_validation_block_id = -1
        self._queues = {cluster: deque() for cluster in config.ACCEPTED_CLUSTER_IDS}
        self._queued_source_tokens = 0
        self._documents_since_last_schedule = 0
        self._rolling_contributions: deque[_RollingContribution] = deque()
        self._rolling_source_tokens = 0
        self._rolling_cluster_source_tokens = {cluster: 0 for cluster in config.ACCEPTED_CLUSTER_IDS}
        self.validation_source_tokens = 0

    @property
    def queued_source_tokens(self) -> int:
        return self._queued_source_tokens

    @property
    def rolling_source_tokens(self) -> int:
        return self._rolling_source_tokens

    @property
    def rolling_cluster_source_tokens(self) -> dict[int, int]:
        return dict(self._rolling_cluster_source_tokens)

    def add_training_document(self, document: SourceDocument) -> None:
        if document.cluster_id not in self._queues:
            raise ValueError("training document has excluded cluster")
        bucket = self._queues[document.cluster_id]
        if len(bucket) >= self.config.per_cluster_queue_limit:
            raise RuntimeError(f"cluster {document.cluster_id} queue is full")
        bucket.append(document)
        self._queued_source_tokens += document.source_token_count
        self._documents_since_last_schedule += 1

    def _queue_source_tokens(self) -> int:
        return self._queued_source_tokens

    def _apply_rolling_contribution(
        self,
        contributions: deque[_RollingContribution],
        counts: dict[int, int],
        total: int,
        cluster_id: int,
        token_count: int,
    ) -> int:
        """Append one contribution and retain exactly the newest window tokens."""

        if token_count <= 0:
            return total
        contributions.append(_RollingContribution(cluster_id, token_count))
        counts[cluster_id] = counts.get(cluster_id, 0) + token_count
        total += token_count
        for oldest, trimmed in self._rolling_trim_plan(contributions, total):
            actual_oldest = contributions.popleft()
            if actual_oldest != oldest:
                raise RuntimeError("rolling contribution order changed during trim")
            remaining = oldest.token_count - trimmed
            if remaining:
                # A partial oldest contribution retains its newest tail.  If
                # the new document itself is oversized, this same operation
                # leaves its newest largest-window tokens in the deque.
                contributions.appendleft(_RollingContribution(oldest.cluster_id, remaining))
            counts[oldest.cluster_id] = counts.get(oldest.cluster_id, 0) - trimmed
            total -= trimmed
        return total

    def _rolling_trim_plan(
        self,
        contributions: Iterable[_RollingContribution],
        total: int,
    ) -> list[tuple[_RollingContribution, int]]:
        """Return the compact oldest-to-newest trims needed for one transition."""

        excess = max(0, total - max(self.config.rolling_mixture_windows))
        plan: list[tuple[_RollingContribution, int]] = []
        for contribution in contributions:
            if excess <= 0:
                break
            trimmed = min(contribution.token_count, excess)
            plan.append((contribution, trimmed))
            excess -= trimmed
        if excess:
            raise RuntimeError("rolling contribution trim exceeded available tokens")
        return plan

    def _append_rolling_document(self, document: SourceDocument) -> None:
        self._rolling_source_tokens = self._apply_rolling_contribution(
            self._rolling_contributions,
            self._rolling_cluster_source_tokens,
            self._rolling_source_tokens,
            document.cluster_id,
            document.source_token_count,
        )

    def _mixture_error_fraction(self, counts: Mapping[int, int], total: int) -> Fraction:
        if total <= 0:
            return Fraction(0)
        return max(
            (
                abs(
                    Fraction(int(counts.get(cluster, 0)), total)
                    - Fraction(self.scheduler.weight_units[cluster], self.scheduler.weight_total)
                )
                for cluster in self.scheduler.weight_units
            ),
            default=Fraction(0),
        )

    def _mixture_error_from_counts(self, counts: Mapping[int, int], total: int) -> float:
        return float(self._mixture_error_fraction(counts, total))

    def _rolling_counts_for_windows(self) -> dict[int, tuple[dict[int, int], int]]:
        """Measure all configured windows with one reverse compact-history scan."""

        windows = tuple(dict.fromkeys(self.config.rolling_mixture_windows))
        counts_by_window = {
            window: {cluster: 0 for cluster in self.scheduler.weight_units}
            for window in windows
        }
        totals = {window: 0 for window in windows}
        unfinished = set(windows)
        for contribution in reversed(self._rolling_contributions):
            if not unfinished:
                break
            source_tokens = contribution.token_count
            for window in unfinished:
                take = min(source_tokens, window - totals[window])
                counts_by_window[window][contribution.cluster_id] += take
                totals[window] += take
            unfinished = {window for window in unfinished if totals[window] < window}
        return {window: (counts_by_window[window], totals[window]) for window in windows}

    def _predicted_rolling_error(self, document: SourceDocument) -> Fraction:
        """Return the rolling error after appending and trimming ``document``."""

        candidate = _RollingContribution(document.cluster_id, document.source_token_count)
        counts = dict(self._rolling_cluster_source_tokens)
        counts[candidate.cluster_id] = counts.get(candidate.cluster_id, 0) + candidate.token_count
        total = self._rolling_source_tokens + candidate.token_count
        def virtual_contributions() -> Iterator[_RollingContribution]:
            yield from self._rolling_contributions
            yield candidate
        for contribution, trimmed in self._rolling_trim_plan(virtual_contributions(), total):
            counts[contribution.cluster_id] = counts.get(contribution.cluster_id, 0) - trimmed
            total -= trimmed
        return self._mixture_error_fraction(counts, total)

    def _candidate_respects_mixture_bound(self, document: SourceDocument) -> bool:
        if self._rolling_source_tokens == 0:
            return True
        current = self._mixture_error_fraction(
            self._rolling_cluster_source_tokens, self._rolling_source_tokens
        )
        predicted = self._predicted_rolling_error(document)
        bound = Fraction(str(self.config.maximum_rolling_mixture_error))
        if current <= bound:
            return predicted <= bound
        # Once the stream is outside the bound, normal scheduling must make
        # strict corrective progress.  Equality is still refused, so a
        # repeated same-cluster head cannot consume the queue indefinitely.
        return predicted < current

    def mixture_measurements(self) -> dict[str, object]:
        window_measurements = self._rolling_counts_for_windows()
        return {
            "cumulative_error": self._mixture_error_from_counts(
                self.scheduler.emitted_source_tokens,
                self.scheduler.total_emitted_source_tokens,
            ),
            "rolling_error": self._mixture_error_from_counts(
                self._rolling_cluster_source_tokens,
                self._rolling_source_tokens,
            ),
            "rolling_windows": {
                str(window): self._mixture_error_from_counts(counts, total)
                for window, (counts, total) in window_measurements.items()
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

    def _acceptable_training_clusters(self, *, force: bool, cluster_id: int | None) -> list[int]:
        if cluster_id is not None:
            if cluster_id not in self._queues:
                raise ValueError("training drain requested an excluded cluster")
            candidates = [cluster_id] if self._queues[cluster_id] else []
        else:
            candidates = [cluster for cluster, documents in self._queues.items() if documents]
        if force:
            return candidates
        return [
            cluster for cluster in candidates
            if self._candidate_respects_mixture_bound(self._queues[cluster][0])
        ]

    def drain_training(
        self,
        *,
        force: bool = True,
        maximum_documents: int | None = None,
        cluster_id: int | None = None,
    ) -> int:
        """Incrementally drain ready queues without requiring every cluster.

        A normal producer call emits at most one document once the prefetch head
        start is satisfied.  A full drain is reserved for finalization or an
        explicit queue-pressure escape hatch.  This prevents an early, lone
        queue from being emptied into a long one-cluster run.
        """
        emitted = 0
        while self._ready_to_schedule(force=force):
            acceptable = self._acceptable_training_clusters(force=force, cluster_id=cluster_id)
            if not acceptable:
                return emitted
            cluster = self.scheduler.choose(acceptable)
            if cluster is None:
                return emitted
            document = self._queues[cluster].popleft()
            self._queued_source_tokens -= document.source_token_count
            if self._queued_source_tokens < 0:
                raise RuntimeError("queued source-token counter became negative")
            self.scheduler.emit(document)
            self._append_rolling_document(document)
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
            if cluster_id is not None and not self._queues[cluster_id]:
                break
        return emitted

    def add_validation_document(self, document: SourceDocument) -> None:
        if document.cluster_id not in self.scheduler.weight_units:
            raise ValueError("validation document has excluded cluster")
        self.validation_source_tokens += document.source_token_count
        for sequence in self.validation_packer.push(document):
            block = self.validation_blocks.push(sequence, split="validation", cumulative_source_tokens=self.validation_source_tokens)
            if block is not None:
                self._publish(block)

    def _publish(self, block: PreparedBlock) -> None:
        writer = self.train_writer if block.split == "train" else self.validation_writer
        writer.write_block(block)
        # ``write_block`` flushes and fsyncs the complete block before this
        # queue operation.  It is not yet immutable until shard finalization.
        if block.split == "train":
            self.consumer.submit(block)
            self.last_durable_block_id = block.block_id
        else:
            if self.validation_consumer is not None:
                self.validation_consumer.submit(block)
            self.last_durable_validation_block_id = block.block_id

    def finalize_active_shards_for_checkpoint(self) -> list[ShardMetadata]:
        """Turn both active tails into immutable, remotely publishable shards.

        This is intentionally legal at a trainer optimizer boundary only; the
        joint coordinator calls it before constructing a checkpoint manifest so
        that a remote checkpoint never references a mutable `.tmp` tail.
        """
        finalized = self.train_writer.finish() + self.validation_writer.finish()
        active_paths = list(self.train_writer.directory.glob(".train-*.bin.tmp"))
        active_paths += list(self.validation_writer.directory.glob(".validation-*.bin.tmp"))
        if active_paths:
            raise RuntimeError(
                "checkpoint boundary found mutable shard state: "
                + ", ".join(str(path) for path in active_paths)
            )
        return finalized

    def finish(self) -> dict[str, object]:
        self.drain_training(force=True)
        for sequence in self.train_packer.finish():
            block = self.train_blocks.push(sequence, split="train", cumulative_source_tokens=self.scheduler.total_emitted_source_tokens)
            if block is not None:
                self._publish(block)
        for sequence in self.validation_packer.finish():
            block = self.validation_blocks.push(sequence, split="validation", cumulative_source_tokens=self.validation_source_tokens)
            if block is not None:
                self._publish(block)
        for builder, split in ((self.train_blocks, "train"), (self.validation_blocks, "validation")):
            cumulative = self.scheduler.total_emitted_source_tokens if split == "train" else self.validation_source_tokens
            block = builder.finish(split=split, cumulative_source_tokens=cumulative)
            if block is not None:
                self._publish(block)
        self._assert_consumers_acknowledged()
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
            "last_durable_train_block_id": self.last_durable_block_id,
            "last_durable_validation_block_id": self.last_durable_validation_block_id,
            "consumer_visibility": "after the complete active-shard block is flush+fsync durable; finalization is fsync+atomic rename",
            "resume_replay": "Joint checkpoints replay only work after last_consumed_block_id; pipeline state is serialized separately.",
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(self.output_dir / config.MANIFEST_FILENAME, manifest)
        write_json_atomic(self.output_dir / config.PROGRESS_FILENAME, self.checkpoint_state())
        return manifest

    @staticmethod
    def _consumer_acknowledged_block_id(consumer: BlockConsumer) -> int:
        raw_acknowledged = getattr(consumer, "last_acknowledged_block_id", -1)
        try:
            return int(raw_acknowledged)
        except (TypeError, ValueError) as error:
            raise RuntimeError("consumer has an invalid acknowledgement watermark") from error

    def _assert_consumers_acknowledged(self) -> None:
        train_acknowledged = self._consumer_acknowledged_block_id(self.consumer)
        if train_acknowledged < self.last_durable_block_id:
            raise RuntimeError(
                "cannot checkpoint while durable training blocks are unacknowledged: "
                f"acknowledged={train_acknowledged}, durable={self.last_durable_block_id}"
            )
        if self.validation_consumer is not None:
            validation_acknowledged = self._consumer_acknowledged_block_id(self.validation_consumer)
            if validation_acknowledged < self.last_durable_validation_block_id:
                raise RuntimeError(
                    "cannot checkpoint while durable validation blocks are unacknowledged: "
                    f"acknowledged={validation_acknowledged}, durable={self.last_durable_validation_block_id}"
                )

    def checkpoint_state(self) -> dict[str, object]:
        # A checkpoint may only point at immutable shard names.  Finalizing
        # here also makes a checkpoint taken between prepared blocks safe: the
        # pending builder state is serialized below while every published
        # block is already represented by finalized metadata.
        self._assert_consumers_acknowledged()
        self.finalize_active_shards_for_checkpoint()
        train_writer_state = _writer_state_dict(self.train_writer)
        validation_writer_state = _writer_state_dict(self.validation_writer)
        return {
            "schema_version": STREAM_CACHE_SCHEMA_VERSION,
            "scheduler": self.scheduler.state_dict(),
            "train_packer": self.train_packer.state_dict(),
            "validation_packer": self.validation_packer.state_dict(),
            "per_cluster_queues": {
                str(k): [_document_to_dict(d) for d in v] for k, v in self._queues.items()
            },
            "rolling_contributions": [
                {"cluster_id": contribution.cluster_id, "token_count": contribution.token_count}
                for contribution in self._rolling_contributions
            ],
            "queued_source_tokens": self._queued_source_tokens,
            "rolling_source_tokens": self._rolling_source_tokens,
            "rolling_cluster_source_tokens": {
                str(cluster): count
                for cluster, count in self._rolling_cluster_source_tokens.items()
                if count
            },
            "documents_since_last_schedule": self._documents_since_last_schedule,
            "validation_source_tokens": self.validation_source_tokens,
            "last_durable_block_id": self.last_durable_block_id,
            "last_durable_train_block_id": self.last_durable_block_id,
            "last_durable_validation_block_id": self.last_durable_validation_block_id,
            "last_consumer_acknowledged_block_id": getattr(self.consumer, "last_acknowledged_block_id", -1),
            "last_validation_consumer_acknowledged_block_id": (
                getattr(self.validation_consumer, "last_acknowledged_block_id", -1)
                if self.validation_consumer is not None else -1
            ),
            "train_blocks": self.train_blocks.state_dict(),
            "validation_blocks": self.validation_blocks.state_dict(),
            "train_writer": train_writer_state,
            "validation_writer": validation_writer_state,
            # Keep one complete list for manifest/checkpoint consumers that do
            # not need to distinguish the two namespaces.
            "finalized_shards": [
                item.as_dict() for item in self.train_writer.shards + self.validation_writer.shards
            ],
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
        validation_consumer: BlockConsumer | None = None,
    ) -> StreamCacheProducer:
        if not isinstance(state, Mapping):
            raise ValueError("stream-cache state must be a mapping")
        if state.get("schema_version") != STREAM_CACHE_SCHEMA_VERSION:
            raise ValueError("unsupported stream-cache state schema")
        last_durable = int(state["last_durable_train_block_id"])
        if "last_durable_block_id" in state and int(state["last_durable_block_id"]) != last_durable:
            raise ValueError("stream-cache train durable block IDs disagree")
        if last_durable < -1:
            raise ValueError("stream-cache state has an invalid train durable block ID")
        instance = cls(
            output_dir,
            stream_config,
            consumer=consumer,
            block_id_counter=[0],
            validation_consumer=validation_consumer,
        )
        if "scheduler" in state and isinstance(state["scheduler"], Mapping):
            instance.scheduler = TokenDeficitScheduler.from_state(stream_config.weights, state["scheduler"])
        if "train_packer" in state and isinstance(state["train_packer"], Mapping):
            instance.train_packer = SequencePacker.from_state(state["train_packer"])
        if "validation_packer" in state and isinstance(state["validation_packer"], Mapping):
            instance.validation_packer = SequencePacker.from_state(state["validation_packer"])
        raw_train_blocks = state.get("train_blocks")
        raw_validation_blocks = state.get("validation_blocks")
        if not isinstance(raw_train_blocks, Mapping) or not isinstance(raw_validation_blocks, Mapping):
            raise ValueError("stream-cache state is missing prepared-block builder state")
        instance.train_blocks = PreparedBlockBuilder.from_state(raw_train_blocks)
        instance.validation_blocks = PreparedBlockBuilder.from_state(raw_validation_blocks)
        if (
            instance.train_blocks.sequences_per_block != stream_config.sequences_per_block
            or instance.validation_blocks.sequences_per_block != stream_config.sequences_per_block
        ):
            raise ValueError("prepared-block builder geometry does not match this run")
        if (
            instance.train_packer.context_length != stream_config.context_length
            or instance.validation_packer.context_length != stream_config.context_length
            or instance.train_packer.final_partial_sequence_policy != stream_config.final_partial_sequence_policy
            or instance.validation_packer.final_partial_sequence_policy != stream_config.final_partial_sequence_policy
        ):
            raise ValueError("sequence packer geometry does not match this run")
        instance.train_block_id_counter = instance.train_blocks._block_id_counter
        instance.validation_block_id_counter = instance.validation_blocks._block_id_counter
        instance.block_id_counter = instance.train_block_id_counter
        if instance.train_block_id_counter[0] != last_durable + 1:
            raise ValueError("train builder next ID does not follow the durable train ID")
        _restore_writer_from_state(instance.train_writer, state.get("train_writer"), output_dir, stream_config)
        _restore_writer_from_state(instance.validation_writer, state.get("validation_writer"), output_dir, stream_config)
        raw_finalized = state.get("finalized_shards")
        expected_finalized = [
            item.as_dict() for item in instance.train_writer.shards + instance.validation_writer.shards
        ]
        if not isinstance(raw_finalized, list) or raw_finalized != expected_finalized:
            raise ValueError("stream-cache finalized shard metadata is inconsistent")
        instance.last_durable_block_id = last_durable
        instance.last_durable_validation_block_id = int(state["last_durable_validation_block_id"])
        if instance.last_durable_validation_block_id < -1:
            raise ValueError("stream-cache state has an invalid validation durable block ID")
        if instance.validation_block_id_counter[0] != instance.last_durable_validation_block_id + 1:
            raise ValueError("validation builder next ID does not follow the durable validation ID")
        train_shard_last = instance.train_writer.shards[-1].last_block_id if instance.train_writer.shards else -1
        validation_shard_last = (
            instance.validation_writer.shards[-1].last_block_id
            if instance.validation_writer.shards else -1
        )
        if train_shard_last != last_durable or validation_shard_last != instance.last_durable_validation_block_id:
            raise ValueError("writer shard tails do not match durable block IDs")
        raw_queues = state.get("per_cluster_queues", {})
        if not isinstance(raw_queues, Mapping):
            raise ValueError("stream-cache state has invalid training queues")
        for raw_cluster, raw_documents in raw_queues.items():
            cluster = int(raw_cluster)
            if cluster not in instance._queues:
                raise ValueError(f"stream-cache state references excluded queue {cluster}")
            if not isinstance(raw_documents, list):
                raise ValueError("stream-cache state has an invalid training queue")
            for raw_document in raw_documents:
                if not isinstance(raw_document, Mapping):
                    raise ValueError("stream-cache state has an invalid queued document")
                document = _document_from_dict(raw_document)
                if document.cluster_id != cluster:
                    raise ValueError("queued document cluster does not match its queue")
                instance._queues[cluster].append(document)
                instance._queued_source_tokens += document.source_token_count
            if len(instance._queues[cluster]) > stream_config.per_cluster_queue_limit:
                raise ValueError("stream-cache state exceeds a per-cluster queue limit")
        raw_rolling = state.get("rolling_contributions")
        if not isinstance(raw_rolling, list):
            raise ValueError("stream-cache state has invalid rolling contributions")
        largest_window = max(stream_config.rolling_mixture_windows)
        for raw_contribution in raw_rolling:
            if not isinstance(raw_contribution, Mapping):
                raise ValueError("stream-cache state has an invalid rolling contribution")
            cluster = int(raw_contribution["cluster_id"])
            token_count = int(raw_contribution["token_count"])
            if cluster not in instance.scheduler.weight_units or token_count <= 0:
                raise ValueError("stream-cache state has an invalid rolling contribution")
            instance._rolling_contributions.append(_RollingContribution(cluster, token_count))
            instance._rolling_source_tokens += token_count
            instance._rolling_cluster_source_tokens[cluster] += token_count
        if instance._rolling_source_tokens > largest_window:
            raise ValueError("stream-cache rolling contributions exceed the configured window")
        raw_queued_counter = int(state["queued_source_tokens"])
        if raw_queued_counter != instance._queued_source_tokens:
            raise ValueError("stream-cache queued-token counter does not match queued documents")
        raw_rolling_counter = int(state["rolling_source_tokens"])
        if raw_rolling_counter != instance._rolling_source_tokens:
            raise ValueError("stream-cache rolling-token counter does not match rolling documents")
        raw_rolling_clusters = state["rolling_cluster_source_tokens"]
        if not isinstance(raw_rolling_clusters, Mapping):
            raise ValueError("stream-cache state has invalid rolling cluster counters")
        expected = {
            str(cluster): count
            for cluster, count in instance._rolling_cluster_source_tokens.items()
            if count
        }
        actual = {
            str(cluster): int(count)
            for cluster, count in raw_rolling_clusters.items()
            if int(count)
        }
        if actual != expected:
            raise ValueError("stream-cache rolling cluster counters do not match rolling documents")
        instance._documents_since_last_schedule = int(state.get("documents_since_last_schedule", 0))
        if instance._documents_since_last_schedule < 0:
            raise ValueError("stream-cache state has invalid waiting-document count")
        instance.validation_source_tokens = int(state.get("validation_source_tokens", 0))
        if instance.validation_source_tokens < 0:
            raise ValueError("stream-cache state has invalid validation token count")
        last_ack = int(state["last_consumer_acknowledged_block_id"])
        if last_ack < instance.last_durable_block_id:
            raise ValueError("stream-cache state drops unacknowledged training blocks")
        current_ack = instance._consumer_acknowledged_block_id(instance.consumer)
        if current_ack < last_ack and hasattr(instance.consumer, "acknowledge"):
            instance.consumer.acknowledge(last_ack)
        validation_ack = state["last_validation_consumer_acknowledged_block_id"]
        if instance.validation_consumer is not None:
            validation_acknowledged = int(validation_ack)
            if validation_acknowledged < instance.last_durable_validation_block_id:
                raise ValueError("stream-cache state drops unacknowledged validation blocks")
            current_validation_ack = instance._consumer_acknowledged_block_id(instance.validation_consumer)
            if (
                current_validation_ack < validation_acknowledged
                and hasattr(instance.validation_consumer, "acknowledge")
            ):
                instance.validation_consumer.acknowledge(validation_acknowledged)
        return instance


def _writer_state_dict(writer: ImmutableShardWriter) -> dict[str, object]:
    return {
        "split": writer.split,
        "next_index": writer._index,
        "cumulative_cluster_source_tokens": {
            str(cluster): count for cluster, count in writer._cumulative_counts.items()
        },
        "shards": [item.as_dict() for item in writer.shards],
    }


def _shard_metadata_from_dict(
    raw: Mapping[str, object],
    *,
    output_dir: Path,
    expected_split: str,
    context_length: int,
) -> tuple[ShardMetadata, int]:
    if not isinstance(raw, Mapping):
        raise ValueError("shard metadata must be a mapping")
    filename = raw.get("filename")
    split = raw.get("split")
    if not isinstance(filename, str) or split != expected_split:
        raise ValueError("shard metadata has an invalid split or filename")
    relative = Path(filename)
    if (
        relative.is_absolute()
        or relative.parent != Path(expected_split)
        or not relative.name.startswith(f"{expected_split}-")
        or relative.suffix != ".bin"
    ):
        raise ValueError("shard metadata references a path outside its split")
    suffix = relative.stem[len(expected_split) + 1:]
    if not suffix.isdigit():
        raise ValueError("shard metadata has an invalid shard index")
    index = int(suffix)
    path = output_dir / relative
    if not path.is_file():
        raise FileNotFoundError(f"Referenced finalized shard does not exist: {path}")

    try:
        byte_size = int(raw["byte_size"])
        token_count = int(raw["token_count"])
        sequence_count = int(raw["sequence_count"])
        first_block_id = int(raw["first_block_id"])
        last_block_id = int(raw["last_block_id"])
        stored_context_length = int(raw["context_length"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("shard metadata has invalid numeric fields") from error
    checksum = raw.get("checksum")
    if not isinstance(checksum, str) or len(checksum) != 64:
        raise ValueError("shard metadata has an invalid checksum")
    if byte_size < 0 or byte_size % 2 or token_count != byte_size // 2 or sequence_count <= 0:
        raise ValueError("shard metadata has inconsistent size or sequence counts")
    if first_block_id < 0 or last_block_id < first_block_id:
        raise ValueError("shard metadata has an invalid block range")
    if stored_context_length != context_length:
        raise ValueError("shard context length does not match this run")
    if raw.get("int_type") != config.INT_TYPE or raw.get("byte_order") != config.BYTE_ORDER:
        raise ValueError("shard integer representation does not match this run")
    if path.stat().st_size != byte_size:
        raise ValueError(f"Referenced shard has the wrong size: {path}")
    if sha256_file(path) != checksum:
        raise ValueError(f"Referenced shard checksum mismatch: {path}")

    def parse_counts(field: str) -> dict[int, int]:
        raw_counts = raw.get(field)
        if not isinstance(raw_counts, Mapping):
            raise ValueError(f"shard metadata has invalid {field}")
        counts: dict[int, int] = {}
        for raw_cluster, raw_count in raw_counts.items():
            if isinstance(raw_count, bool):
                raise ValueError(f"shard metadata has invalid {field}")
            count = int(raw_count)
            if count < 0:
                raise ValueError(f"shard metadata has negative {field}")
            counts[int(raw_cluster)] = count
        return counts

    metadata = ShardMetadata(
        filename=filename,
        split=expected_split,
        byte_size=byte_size,
        token_count=token_count,
        sequence_count=sequence_count,
        checksum=checksum,
        first_block_id=first_block_id,
        last_block_id=last_block_id,
        context_length=stored_context_length,
        int_type=str(raw["int_type"]),
        byte_order=str(raw["byte_order"]),
        cumulative_cluster_source_tokens=parse_counts("cumulative_cluster_source_tokens"),
        shard_cluster_source_tokens=parse_counts("shard_cluster_source_tokens"),
    )
    return metadata, index


def _restore_writer_from_state(
    writer: ImmutableShardWriter,
    raw_state: object,
    output_dir: Path,
    stream_config: StreamCacheConfig,
) -> None:
    if not isinstance(raw_state, Mapping):
        raise ValueError(f"missing {writer.split} writer state")
    if raw_state.get("split") != writer.split:
        raise ValueError(f"{writer.split} writer state has the wrong split")
    raw_shards = raw_state.get("shards")
    if not isinstance(raw_shards, list):
        raise ValueError(f"{writer.split} writer state has invalid shard metadata")
    parsed: list[ShardMetadata] = []
    indexes: list[int] = []
    for raw_shard in raw_shards:
        if not isinstance(raw_shard, Mapping):
            raise ValueError(f"{writer.split} writer state has invalid shard metadata")
        metadata, index = _shard_metadata_from_dict(
            raw_shard,
            output_dir=output_dir,
            expected_split=writer.split,
            context_length=stream_config.context_length,
        )
        parsed.append(metadata)
        indexes.append(index)
    if indexes != sorted(indexes) or len(set(indexes)) != len(indexes):
        raise ValueError(f"{writer.split} writer shard indexes are not ordered")

    previous: dict[int, int] = {}
    for metadata in parsed:
        expected = dict(previous)
        for cluster, count in metadata.shard_cluster_source_tokens.items():
            expected[cluster] = expected.get(cluster, 0) + count
        expected = {cluster: count for cluster, count in expected.items() if count}
        if metadata.cumulative_cluster_source_tokens != expected:
            raise ValueError(f"{writer.split} writer cumulative shard counters are inconsistent")
        previous = expected

    raw_next_index = raw_state.get("next_index")
    if isinstance(raw_next_index, bool) or not isinstance(raw_next_index, int) or raw_next_index < 0:
        raise ValueError(f"{writer.split} writer state has an invalid next index")
    if indexes and raw_next_index <= indexes[-1]:
        raise ValueError(f"{writer.split} writer next index would reuse a finalized shard")
    existing_indexes = [
        int(path.stem.split("-")[-1])
        for path in writer.directory.glob(f"{writer.split}-*.bin")
        if path.stem.split("-")[-1].isdigit()
    ]
    if existing_indexes and raw_next_index <= max(existing_indexes):
        raise ValueError(f"{writer.split} writer next index conflicts with an existing shard")
    active_paths = list(writer.directory.glob(f".{writer.split}-*.bin.tmp"))
    if active_paths:
        raise ValueError(
            f"{writer.split} writer has mutable active shard state that is not in the checkpoint"
        )

    raw_cumulative = raw_state.get("cumulative_cluster_source_tokens")
    if not isinstance(raw_cumulative, Mapping):
        raise ValueError(f"{writer.split} writer state has invalid cumulative counters")
    cumulative = {int(cluster): int(count) for cluster, count in raw_cumulative.items()}
    if any(count < 0 for count in cumulative.values()) or cumulative != previous:
        raise ValueError(f"{writer.split} writer cumulative counters do not match shards")
    writer.shards = parsed
    writer._index = raw_next_index
    writer._cumulative_counts = cumulative
    writer._blocks = []
    writer._cluster_counts = {}
    writer._handle = None


def _document_to_dict(document: SourceDocument) -> dict[str, object]:
    return {"source_id": document.source_id, "cluster_id": document.cluster_id,
            "tokens": list(document.tokens), "work_item_index": document.work_item_index,
            "record_start": document.record_start}


def _document_from_dict(data: Mapping[str, object]) -> SourceDocument:
    return SourceDocument(str(data["source_id"]), int(data["cluster_id"]),
                          tuple(int(token) for token in data["tokens"]),
                          int(data.get("work_item_index", 0)), int(data.get("record_start", 0)))


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
    """Yield bounded batches in deterministic work-item cycles."""

    for cycle in _iter_parallel_document_batch_cycles(
        plan,
        reader_factory=reader_factory,
        workers=workers,
        max_in_flight=max_in_flight,
        validation_probability=validation_probability,
        maximum_source_tokens_per_batch=maximum_source_tokens_per_batch,
        maximum_documents_per_batch=maximum_documents_per_batch,
        maximum_bytes_per_batch=maximum_bytes_per_batch,
    ):
        yield from cycle


def _document_batches_for_item(
    plan: WorkPlan,
    item: WorkItem,
    *,
    reader_factory: Callable[[SourceFile], RangeReader],
    files: Mapping[str, SourceFile],
    validation_probability: float,
    maximum_source_tokens_per_batch: int,
    maximum_documents_per_batch: int,
    maximum_bytes_per_batch: int,
) -> Iterator[DocumentBatch]:
    """Produce bounded batches for one work item when advanced by a worker."""

    reader = reader_factory(files[item.filename])
    records: list[tuple[bool, SourceDocument]] = []
    source_tokens = estimated_bytes = 0
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
        if records and (
            len(records) >= maximum_documents_per_batch
            or projected_tokens > maximum_source_tokens_per_batch
            or projected_bytes > maximum_bytes_per_batch
        ):
            batch = DocumentBatch(item.index, tuple(records), source_tokens, estimated_bytes)
            records, source_tokens, estimated_bytes = [], 0, 0
            yield batch
        records.append((
            is_validation(
                seed=config.SELECTION_SEED,
                revision=plan.revision,
                filename=item.filename,
                record_start=record.record_start,
                probability=validation_probability,
            ),
            document,
        ))
        source_tokens += document.source_token_count
        estimated_bytes += len(record.raw) + document.source_token_count * 2
    if records:
        batch = DocumentBatch(item.index, tuple(records), source_tokens, estimated_bytes)
        records, source_tokens, estimated_bytes = [], 0, 0
        yield batch


def _iter_parallel_document_batch_cycles(
    plan: WorkPlan,
    *,
    reader_factory: Callable[[SourceFile], RangeReader],
    workers: int,
    max_in_flight: int,
    validation_probability: float,
    maximum_source_tokens_per_batch: int,
    maximum_documents_per_batch: int,
    maximum_bytes_per_batch: int,
) -> Iterator[tuple[DocumentBatch, ...]]:
    """Advance one bounded batch iterator per active work item per cycle."""

    if workers <= 0 or max_in_flight <= 0:
        raise ValueError("workers and max_in_flight must be positive")
    if min(maximum_source_tokens_per_batch, maximum_documents_per_batch, maximum_bytes_per_batch) <= 0:
        raise ValueError("batch limits must be positive")
    files = {source.path: source for source in plan.source_files}
    finished = object()

    def next_batch(iterator: Iterator[DocumentBatch]) -> object:
        try:
            return next(iterator)
        except StopIteration:
            return finished

    active: list[tuple[int, Iterator[DocumentBatch]]] = []
    next_submit = 0
    in_cycle_futures: list[Future[object]] = []
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="source-reader") as pool:
        try:
            while next_submit < len(plan.work_items) and len(active) < max_in_flight:
                item = plan.work_items[next_submit]
                active.append((next_submit, _document_batches_for_item(
                    plan,
                    item,
                    reader_factory=reader_factory,
                    files=files,
                    validation_probability=validation_probability,
                    maximum_source_tokens_per_batch=maximum_source_tokens_per_batch,
                    maximum_documents_per_batch=maximum_documents_per_batch,
                    maximum_bytes_per_batch=maximum_bytes_per_batch,
                )))
                next_submit += 1

            while active:
                cycle = tuple(active)
                in_cycle_futures = [pool.submit(next_batch, iterator) for _, iterator in cycle]
                batches: list[DocumentBatch] = []
                finished_slots: set[int] = set()
                for (slot, _), future in zip(cycle, in_cycle_futures):
                    result = future.result()
                    if result is finished:
                        finished_slots.add(slot)
                    elif isinstance(result, DocumentBatch):
                        batches.append(result)
                    else:
                        raise RuntimeError("source reader emitted an invalid batch result")
                if finished_slots:
                    active = [entry for entry in active if entry[0] not in finished_slots]
                while next_submit < len(plan.work_items) and len(active) < max_in_flight:
                    item = plan.work_items[next_submit]
                    active.append((next_submit, _document_batches_for_item(
                        plan,
                        item,
                        reader_factory=reader_factory,
                        files=files,
                        validation_probability=validation_probability,
                        maximum_source_tokens_per_batch=maximum_source_tokens_per_batch,
                        maximum_documents_per_batch=maximum_documents_per_batch,
                        maximum_bytes_per_batch=maximum_bytes_per_batch,
                    )))
                    next_submit += 1
                if batches:
                    yield tuple(batches)
        except BaseException:
            for future in in_cycle_futures:
                future.cancel()
            raise
        finally:
            for future in in_cycle_futures:
                future.cancel()


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
    """Yield one document at a time, interleaved across each batch cycle."""
    for cycle in _iter_parallel_document_batch_cycles(
        plan,
        reader_factory=reader_factory,
        workers=workers,
        max_in_flight=max_in_flight,
        validation_probability=validation_probability,
        maximum_source_tokens_per_batch=maximum_source_tokens_per_batch,
        maximum_documents_per_batch=maximum_documents_per_batch,
        maximum_bytes_per_batch=maximum_bytes_per_batch,
    ):
        for record_index in range(max(len(batch.records) for batch in cycle)):
            for batch in cycle:
                if record_index < len(batch.records):
                    yield batch.records[record_index]


def build_stream_cache(
    output_dir: Path | str,
    stream_config: StreamCacheConfig,
    plan: WorkPlan,
    reader_factory: Callable[[SourceFile], RangeReader],
    consumer: BlockConsumer | None = None,
    validation_consumer: BlockConsumer | None = None,
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
        validation_consumer=validation_consumer,
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
                    drained = producer.drain_training(
                        force=True,
                        maximum_documents=1,
                        cluster_id=document.cluster_id,
                    )
                    if drained != 1 or len(producer._queues[document.cluster_id]) >= stream_config.per_cluster_queue_limit:
                        raise RuntimeError(
                            f"could not free training queue for cluster {document.cluster_id}"
                        )
                producer.add_training_document(document)
                producer.drain_training(force=False, maximum_documents=1)

        return producer.finish()
    finally:
        producer.close()
