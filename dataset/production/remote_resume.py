"""Resume a production cache whose older finalized shards were evicted locally.

This path is intentionally narrower than ``StreamCacheProducer.from_state``.
The ordinary local-cache resume continues to require every finalized shard on
local disk.  Remote-only resume is legal only after the caller has independently
verified a durability manifest whose immutable shard metadata exactly covers the
producer checkpoint.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

from dataset import config
from dataset.src.streaming import (
    STREAM_CACHE_SCHEMA_VERSION,
    PreparedBlockBuilder,
    SequencePacker,
    ShardMetadata,
    StreamCacheConfig,
    StreamCacheProducer,
    TokenDeficitScheduler,
    _RollingContribution,
    _document_from_dict,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _counts(raw: object, *, field: str) -> dict[int, int]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"shard metadata has invalid {field}")
    result: dict[int, int] = {}
    for cluster, count in raw.items():
        if isinstance(count, bool):
            raise ValueError(f"shard metadata has invalid {field}")
        parsed = int(count)
        if parsed < 0:
            raise ValueError(f"shard metadata has negative {field}")
        result[int(cluster)] = parsed
    return result


def _parse_remote_shard(
    raw: Mapping[str, object],
    *,
    expected_split: str,
    context_length: int,
    verified: Mapping[str, Mapping[str, object]],
) -> tuple[ShardMetadata, int]:
    filename = raw.get("filename")
    if not isinstance(filename, str) or raw.get("split") != expected_split:
        raise ValueError("shard metadata has an invalid split or filename")
    relative = Path(filename)
    if (
        relative.is_absolute()
        or relative.parent != Path(expected_split)
        or relative.suffix != ".bin"
        or not relative.name.startswith(f"{expected_split}-")
    ):
        raise ValueError("shard metadata references a path outside its split")
    suffix = relative.stem[len(expected_split) + 1 :]
    if not suffix.isdigit():
        raise ValueError("shard metadata has an invalid shard index")
    index = int(suffix)

    try:
        byte_size = int(raw["byte_size"])
        token_count = int(raw["token_count"])
        sequence_count = int(raw["sequence_count"])
        first = int(raw["first_block_id"])
        last = int(raw["last_block_id"])
        stored_context = int(raw["context_length"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("shard metadata has invalid numeric fields") from error
    checksum = raw.get("checksum")
    if not isinstance(checksum, str) or not _SHA256.fullmatch(checksum):
        raise ValueError("shard metadata has an invalid checksum")
    if byte_size <= 0 or byte_size % 2 or token_count != byte_size // 2 or sequence_count <= 0:
        raise ValueError("shard metadata has inconsistent size or sequence counts")
    if first < 0 or last < first or stored_context != context_length:
        raise ValueError("shard metadata has invalid block range or context length")
    if raw.get("int_type") != config.INT_TYPE or raw.get("byte_order") != config.BYTE_ORDER:
        raise ValueError("shard integer representation does not match this run")

    remote = verified.get(filename)
    if remote is None or remote.get("remote_durable") is not True:
        raise RuntimeError(f"evicted shard has no verified remote durability entry: {filename}")
    required_equal = (
        "filename", "split", "byte_size", "token_count", "sequence_count",
        "checksum", "first_block_id", "last_block_id", "context_length",
        "int_type", "byte_order", "cumulative_cluster_source_tokens",
        "shard_cluster_source_tokens",
    )
    for field in required_equal:
        if remote.get(field) != raw.get(field):
            raise RuntimeError(f"remote durability metadata disagrees with producer state: {filename} {field}")
    if remote.get("local_sha256") != checksum:
        raise RuntimeError(f"remote durability checksum disagrees with producer state: {filename}")

    return ShardMetadata(
        filename=filename,
        split=expected_split,
        byte_size=byte_size,
        token_count=token_count,
        sequence_count=sequence_count,
        checksum=checksum,
        first_block_id=first,
        last_block_id=last,
        context_length=stored_context,
        int_type=str(raw["int_type"]),
        byte_order=str(raw["byte_order"]),
        cumulative_cluster_source_tokens=_counts(
            raw.get("cumulative_cluster_source_tokens"),
            field="cumulative_cluster_source_tokens",
        ),
        shard_cluster_source_tokens=_counts(
            raw.get("shard_cluster_source_tokens"),
            field="shard_cluster_source_tokens",
        ),
    ), index


def _restore_writer(
    writer: object,
    raw_state: object,
    *,
    stream: StreamCacheConfig,
    verified: Mapping[str, Mapping[str, object]],
) -> None:
    if not isinstance(raw_state, Mapping) or raw_state.get("split") != writer.split:
        raise ValueError(f"missing or invalid {writer.split} writer state")
    raw_shards = raw_state.get("shards")
    if not isinstance(raw_shards, list):
        raise ValueError(f"{writer.split} writer state has invalid shard metadata")
    parsed: list[ShardMetadata] = []
    indexes: list[int] = []
    for raw in raw_shards:
        if not isinstance(raw, Mapping):
            raise ValueError(f"{writer.split} writer state has invalid shard metadata")
        metadata, index = _parse_remote_shard(
            raw,
            expected_split=writer.split,
            context_length=stream.context_length,
            verified=verified,
        )
        parsed.append(metadata)
        indexes.append(index)
    if indexes != sorted(indexes) or len(indexes) != len(set(indexes)):
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

    next_index = raw_state.get("next_index")
    if isinstance(next_index, bool) or not isinstance(next_index, int) or next_index < 0:
        raise ValueError(f"{writer.split} writer state has an invalid next index")
    if indexes and next_index <= indexes[-1]:
        raise ValueError(f"{writer.split} writer next index would reuse a finalized shard")
    cumulative = _counts(
        raw_state.get("cumulative_cluster_source_tokens"),
        field=f"{writer.split} cumulative_cluster_source_tokens",
    )
    if cumulative != previous:
        raise ValueError(f"{writer.split} writer cumulative counters do not match shards")
    active = list(writer.directory.glob(f".{writer.split}-*.bin.tmp"))
    if active:
        raise RuntimeError(f"{writer.split} writer has uncheckpointed mutable shard state")

    writer.shards = parsed
    writer._index = next_index
    writer._cumulative_counts = cumulative
    writer._blocks = []
    writer._cluster_counts = {}
    writer._handle = None


def verified_remote_entries(manifest: object) -> dict[str, Mapping[str, object]]:
    if not isinstance(manifest, Mapping) or manifest.get("version") != 1:
        raise RuntimeError("remote durability manifest has an unsupported structure")
    rows = manifest.get("shards")
    if not isinstance(rows, list):
        raise RuntimeError("remote durability manifest has no shard list")
    result: dict[str, Mapping[str, object]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise RuntimeError("remote durability manifest has an invalid shard entry")
        name = row.get("filename")
        if not isinstance(name, str) or name in result:
            raise RuntimeError("remote durability manifest has an invalid or duplicate filename")
        if row.get("remote_durable") is not True:
            raise RuntimeError(f"remote durability entry is not verified: {name}")
        result[name] = row
    return result


def restore_remote_evicted_producer(
    output_dir: Path,
    stream: StreamCacheConfig,
    state: Mapping[str, object],
    *,
    durability_manifest: Mapping[str, object],
) -> StreamCacheProducer:
    """Restore producer metadata only after remote shard read-back was verified."""

    verified = verified_remote_entries(durability_manifest)
    if not isinstance(state, Mapping) or state.get("schema_version") != STREAM_CACHE_SCHEMA_VERSION:
        raise ValueError("unsupported stream-cache state schema")
    last_durable = int(state["last_durable_train_block_id"])
    if "last_durable_block_id" in state and int(state["last_durable_block_id"]) != last_durable:
        raise ValueError("stream-cache train durable block IDs disagree")
    if last_durable < -1:
        raise ValueError("stream-cache state has an invalid train durable block ID")

    instance = StreamCacheProducer(output_dir, stream, block_id_counter=[0])
    if isinstance(state.get("scheduler"), Mapping):
        instance.scheduler = TokenDeficitScheduler.from_state(stream.weights, state["scheduler"])
    if isinstance(state.get("train_packer"), Mapping):
        instance.train_packer = SequencePacker.from_state(state["train_packer"])
    if isinstance(state.get("validation_packer"), Mapping):
        instance.validation_packer = SequencePacker.from_state(state["validation_packer"])
    raw_train_blocks = state.get("train_blocks")
    raw_validation_blocks = state.get("validation_blocks")
    if not isinstance(raw_train_blocks, Mapping) or not isinstance(raw_validation_blocks, Mapping):
        raise ValueError("stream-cache state is missing prepared-block builder state")
    instance.train_blocks = PreparedBlockBuilder.from_state(raw_train_blocks)
    instance.validation_blocks = PreparedBlockBuilder.from_state(raw_validation_blocks)
    if (
        instance.train_blocks.sequences_per_block != stream.sequences_per_block
        or instance.validation_blocks.sequences_per_block != stream.sequences_per_block
    ):
        raise ValueError("prepared-block builder geometry does not match this run")
    if (
        instance.train_packer.context_length != stream.context_length
        or instance.validation_packer.context_length != stream.context_length
        or instance.train_packer.final_partial_sequence_policy != stream.final_partial_sequence_policy
        or instance.validation_packer.final_partial_sequence_policy != stream.final_partial_sequence_policy
    ):
        raise ValueError("sequence packer geometry does not match this run")

    instance.train_block_id_counter = instance.train_blocks._block_id_counter
    instance.validation_block_id_counter = instance.validation_blocks._block_id_counter
    instance.block_id_counter = instance.train_block_id_counter
    if instance.train_block_id_counter[0] != last_durable + 1:
        raise ValueError("train builder next ID does not follow the durable train ID")
    _restore_writer(instance.train_writer, state.get("train_writer"), stream=stream, verified=verified)
    _restore_writer(
        instance.validation_writer,
        state.get("validation_writer"),
        stream=stream,
        verified=verified,
    )
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
    train_tail = instance.train_writer.shards[-1].last_block_id if instance.train_writer.shards else -1
    validation_tail = (
        instance.validation_writer.shards[-1].last_block_id
        if instance.validation_writer.shards else -1
    )
    if train_tail != last_durable or validation_tail != instance.last_durable_validation_block_id:
        raise ValueError("writer shard tails do not match durable block IDs")

    raw_queues = state.get("per_cluster_queues", {})
    if not isinstance(raw_queues, Mapping):
        raise ValueError("stream-cache state has invalid training queues")
    for raw_cluster, raw_documents in raw_queues.items():
        cluster = int(raw_cluster)
        if cluster not in instance._queues or not isinstance(raw_documents, list):
            raise ValueError("stream-cache state has an invalid training queue")
        for raw_document in raw_documents:
            if not isinstance(raw_document, Mapping):
                raise ValueError("stream-cache state has an invalid queued document")
            document = _document_from_dict(raw_document)
            if document.cluster_id != cluster:
                raise ValueError("queued document cluster does not match its queue")
            instance._queues[cluster].append(document)
            instance._queued_source_tokens += document.source_token_count
        if len(instance._queues[cluster]) > stream.per_cluster_queue_limit:
            raise ValueError("stream-cache state exceeds a per-cluster queue limit")

    raw_rolling = state.get("rolling_contributions")
    if not isinstance(raw_rolling, list):
        raise ValueError("stream-cache state has invalid rolling contributions")
    largest_window = max(stream.rolling_mixture_windows)
    for raw in raw_rolling:
        if not isinstance(raw, Mapping):
            raise ValueError("stream-cache state has an invalid rolling contribution")
        cluster = int(raw["cluster_id"])
        count = int(raw["token_count"])
        if cluster not in instance.scheduler.weight_units or count <= 0:
            raise ValueError("stream-cache state has an invalid rolling contribution")
        instance._rolling_contributions.append(_RollingContribution(cluster, count))
        instance._rolling_source_tokens += count
        instance._rolling_cluster_source_tokens[cluster] += count
    if instance._rolling_source_tokens > largest_window:
        raise ValueError("stream-cache rolling contributions exceed the configured window")
    if int(state["queued_source_tokens"]) != instance._queued_source_tokens:
        raise ValueError("stream-cache queued-token counter does not match queued documents")
    if int(state["rolling_source_tokens"]) != instance._rolling_source_tokens:
        raise ValueError("stream-cache rolling-token counter does not match rolling documents")
    raw_clusters = state.get("rolling_cluster_source_tokens")
    if not isinstance(raw_clusters, Mapping):
        raise ValueError("stream-cache state has invalid rolling cluster counters")
    expected_clusters = {
        str(cluster): count
        for cluster, count in instance._rolling_cluster_source_tokens.items()
        if count
    }
    actual_clusters = {str(cluster): int(count) for cluster, count in raw_clusters.items() if int(count)}
    if actual_clusters != expected_clusters:
        raise ValueError("stream-cache rolling cluster counters do not match rolling documents")

    instance._documents_since_last_schedule = int(state.get("documents_since_last_schedule", 0))
    instance.validation_source_tokens = int(state.get("validation_source_tokens", 0))
    if instance._documents_since_last_schedule < 0 or instance.validation_source_tokens < 0:
        raise ValueError("stream-cache state has invalid counters")
    last_ack = int(state["last_consumer_acknowledged_block_id"])
    if last_ack < instance.last_durable_block_id:
        raise ValueError("stream-cache state drops unacknowledged training blocks")
    instance.consumer.acknowledge(last_ack)
    return instance


__all__ = ["restore_remote_evicted_producer", "verified_remote_entries"]
