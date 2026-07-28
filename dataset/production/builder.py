"""Crash-safe production builder for the schema-v2 streaming cache."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Mapping

from dataset import config
from dataset.src.bytesource import RangeReader, SourceFile
from dataset.src.remote import RemoteShardStore, ensure_safe_directory
from dataset.src.storage import read_json, write_json_atomic
from dataset.src.streaming import StreamCacheConfig, StreamCacheProducer, parallel_read_documents
from dataset.src.workplan import WorkPlan

from .policy import (
    PRODUCTION_STATE_VERSION,
    ProductionPolicy,
    configuration_hash,
    incorporated_source_tokens,
    reader_configuration,
    schema_hash,
)
from .remote import mirror_shards
from .safety import (
    PROGRESS_BACKUP_FILENAME,
    RunLock,
    discard_uncheckpointed_artifacts,
    recover_progress_backup,
    unlink_durable,
)

LOGGER = logging.getLogger(__name__)


def _validate_cursor(raw: object, *, plan: WorkPlan, stream: StreamCacheConfig) -> dict[str, object]:
    if not isinstance(raw, Mapping) or raw.get("version") != 1:
        raise ValueError("production progress is missing a supported source-reader cursor")
    if raw.get("work_plan_hash") != plan.hash:
        raise ValueError("production source-reader cursor belongs to a different work plan")
    if raw.get("reader_configuration") != reader_configuration(stream):
        raise ValueError("production source-reader configuration does not match this run")
    documents = raw.get("documents_consumed")
    positions = raw.get("last_incorporated_record_start")
    if isinstance(documents, bool) or not isinstance(documents, int) or documents < 0:
        raise ValueError("production source-reader document count is invalid")
    if not isinstance(positions, Mapping):
        raise ValueError("production source-reader offsets are invalid")
    parsed: dict[str, int] = {}
    for work_item, record_start in positions.items():
        if not isinstance(work_item, str) or not work_item.isdigit():
            raise ValueError("production source-reader work-item key is invalid")
        if isinstance(record_start, bool) or not isinstance(record_start, int) or record_start < 0:
            raise ValueError("production source-reader record offset is invalid")
        parsed[work_item] = record_start
    if documents == 0 and parsed:
        raise ValueError("empty production source-reader cursor has record offsets")
    return {"documents_consumed": documents, "last_incorporated_record_start": parsed}


def _production_identity(policy: ProductionPolicy, cfg_hash: str, format_hash: str) -> dict[str, object]:
    return {
        "version": PRODUCTION_STATE_VERSION,
        "configuration_hash": cfg_hash,
        "schema_hash": format_hash,
        "policy": policy.as_dict(),
    }


def _decorate_state(
    state: dict[str, object],
    *,
    policy: ProductionPolicy,
    cfg_hash: str,
    format_hash: str,
    plan: WorkPlan,
    stream: StreamCacheConfig,
    documents_consumed: int,
    last_record_start: Mapping[str, int],
    accepted: int,
    complete: bool,
) -> None:
    state["source_reader"] = {
        "version": 1,
        "work_plan_hash": plan.hash,
        "reader_configuration": reader_configuration(stream),
        "documents_consumed": documents_consumed,
        "last_incorporated_record_start": dict(sorted(last_record_start.items())),
    }
    state["production"] = _production_identity(policy, cfg_hash, format_hash)
    state["accepted_source_tokens_incorporated"] = accepted
    state["complete"] = complete


def build_production_cache(
    output_dir: Path | str,
    stream: StreamCacheConfig,
    policy: ProductionPolicy,
    plan: WorkPlan,
    reader_factory: Callable[[SourceFile], RangeReader],
    *,
    remote_store: RemoteShardStore | None,
    resume: bool = False,
    simulate_crash_after_documents: int | None = None,
) -> dict[str, object]:
    """Build or resume a whole-document bounded and remotely durable cache."""

    if policy.remote_required and remote_store is None:
        raise RuntimeError("production policy requires a configured remote shard store")
    if simulate_crash_after_documents is not None and simulate_crash_after_documents <= 0:
        raise ValueError("simulate_crash_after_documents must be positive")

    output_dir = Path(output_dir)
    ensure_safe_directory(output_dir)
    progress_path = output_dir / config.PROGRESS_FILENAME
    manifest_path = output_dir / config.MANIFEST_FILENAME
    cfg_hash = configuration_hash(policy, stream, plan)
    format_hash = schema_hash(stream)

    with RunLock(output_dir):
        if resume:
            recover_progress_backup(output_dir)
            state = read_json(progress_path)
            if not isinstance(state, Mapping):
                raise ValueError("production progress.json must be an object")
            expected = _production_identity(policy, cfg_hash, format_hash)
            if state.get("production") != expected:
                raise ValueError("production progress configuration does not match this invocation")
            cursor = _validate_cursor(state.get("source_reader"), plan=plan, stream=stream)
            if state.get("complete") is True:
                manifest = read_json(manifest_path)
                if not isinstance(manifest, dict):
                    raise ValueError("completed production cache has an invalid manifest")
                if remote_store is not None:
                    mirror_shards(
                        remote_store,
                        output_dir=output_dir,
                        run_id=policy.run_id,
                        shard_entries=list(manifest.get("shards", [])),
                        configuration_hash=cfg_hash,
                        schema_hash=format_hash,
                        verify_existing=True,
                        prune_unreferenced=True,
                    )
                return manifest
            discard_uncheckpointed_artifacts(output_dir, state)
            producer = StreamCacheProducer.from_state(output_dir, stream, state)
            if remote_store is not None:
                mirror_shards(
                    remote_store,
                    output_dir=output_dir,
                    run_id=policy.run_id,
                    shard_entries=list(state.get("finalized_shards", [])),
                    configuration_hash=cfg_hash,
                    schema_hash=format_hash,
                    verify_existing=True,
                )
        else:
            if progress_path.exists() or manifest_path.exists():
                raise FileExistsError(
                    f"production state already exists in {output_dir}; use --resume or a new output directory"
                )
            producer = StreamCacheProducer(output_dir, stream)
            cursor = {"documents_consumed": 0, "last_incorporated_record_start": {}}

        documents_consumed = int(cursor["documents_consumed"])
        last_record_start = dict(cursor["last_incorporated_record_start"])
        last_checkpoint_tokens = incorporated_source_tokens(producer)
        replay_positions: dict[str, int] = {}
        replayed = 0
        replay_verified = documents_consumed == 0
        completion_reason = "source_exhausted"

        def durable_state(*, complete: bool = False) -> dict[str, object]:
            nonlocal last_checkpoint_tokens
            state = producer.checkpoint_state()
            accepted = incorporated_source_tokens(producer)
            _decorate_state(
                state,
                policy=policy,
                cfg_hash=cfg_hash,
                format_hash=format_hash,
                plan=plan,
                stream=stream,
                documents_consumed=documents_consumed,
                last_record_start=last_record_start,
                accepted=accepted,
                complete=complete,
            )
            if remote_store is not None:
                mirror_shards(
                    remote_store,
                    output_dir=output_dir,
                    run_id=policy.run_id,
                    shard_entries=list(state.get("finalized_shards", [])),
                    configuration_hash=cfg_hash,
                    schema_hash=format_hash,
                )
            write_json_atomic(progress_path, state)
            last_checkpoint_tokens = accepted
            LOGGER.info(
                "durable dataset checkpoint: documents=%d accepted=%d shards=%d",
                documents_consumed, accepted, len(state.get("finalized_shards", [])),
            )
            return state

        try:
            for is_validation_doc, document in parallel_read_documents(
                plan,
                reader_factory=reader_factory,
                workers=stream.reader_workers,
                max_in_flight=stream.max_in_flight_work_items,
                maximum_source_tokens_per_batch=stream.reader_batch_source_tokens,
                maximum_documents_per_batch=stream.reader_batch_documents,
                maximum_bytes_per_batch=stream.reader_batch_max_bytes,
            ):
                if not replay_verified and replayed < documents_consumed:
                    replay_positions[str(document.work_item_index)] = document.record_start
                    replayed += 1
                    continue
                if not replay_verified and replayed == documents_consumed:
                    if replay_positions != last_record_start:
                        raise RuntimeError("source reader replay does not match the durable cursor")
                    replay_verified = True

                incorporated = incorporated_source_tokens(producer)
                if incorporated >= policy.target_source_tokens:
                    completion_reason = "target_reached"
                    break
                if incorporated + document.source_token_count > policy.maximum_source_tokens:
                    completion_reason = "hard_maximum_guard"
                    break

                if is_validation_doc:
                    producer.add_validation_document(document)
                else:
                    queue_for_cluster = producer._queues[document.cluster_id]
                    if len(queue_for_cluster) >= stream.per_cluster_queue_limit:
                        drained = producer.drain_training(
                            force=True, maximum_documents=1, cluster_id=document.cluster_id
                        )
                        if drained != 1 or len(queue_for_cluster) >= stream.per_cluster_queue_limit:
                            raise RuntimeError(f"could not free training queue for cluster {document.cluster_id}")
                    producer.add_training_document(document)
                    producer.drain_training(force=False, maximum_documents=1)

                documents_consumed += 1
                last_record_start[str(document.work_item_index)] = document.record_start
                accepted = incorporated_source_tokens(producer)
                if documents_consumed % 10_000 == 0:
                    LOGGER.info(
                        "dataset progress: documents=%d accepted=%d queued=%d",
                        documents_consumed, accepted, producer.queued_source_tokens,
                    )
                if accepted - last_checkpoint_tokens >= policy.checkpoint_source_tokens:
                    durable_state()
                if simulate_crash_after_documents == documents_consumed:
                    durable_state()
                    raise RuntimeError("simulated production interruption after durable checkpoint")

            if not replay_verified and replay_positions != last_record_start:
                raise RuntimeError("source reader replay ended before matching the durable cursor")

            accepted_before_finish = incorporated_source_tokens(producer)
            if accepted_before_finish < policy.minimum_source_tokens:
                durable_state()
                raise RuntimeError(
                    "source ended before the minimum corpus size: "
                    f"accepted={accepted_before_finish}, minimum={policy.minimum_source_tokens}"
                )
            if accepted_before_finish >= policy.target_source_tokens:
                completion_reason = "target_reached"

            safe_state = durable_state()
            backup_path = output_dir / PROGRESS_BACKUP_FILENAME
            write_json_atomic(backup_path, safe_state)
            try:
                manifest = producer.finish()
                accepted = int(manifest["accepted_source_tokens"])
                if accepted != accepted_before_finish:
                    raise RuntimeError(
                        "accepted-source-token accounting changed during finalization: "
                        f"before={accepted_before_finish}, after={accepted}"
                    )
                if accepted > policy.maximum_source_tokens:
                    raise RuntimeError("completed cache exceeds the production hard maximum")

                manifest["work_plan_hash"] = plan.hash
                manifest["production"] = {
                    "version": PRODUCTION_STATE_VERSION,
                    "run_id": policy.run_id,
                    "configuration_hash": cfg_hash,
                    "schema_hash": format_hash,
                    "target_source_tokens": policy.target_source_tokens,
                    "minimum_source_tokens": policy.minimum_source_tokens,
                    "maximum_source_tokens": policy.maximum_source_tokens,
                    "checkpoint_source_tokens": policy.checkpoint_source_tokens,
                    "target_reached": accepted >= policy.target_source_tokens,
                    "completion_reason": completion_reason,
                    "remote_required": policy.remote_required,
                }
                final_state = producer.checkpoint_state()
                _decorate_state(
                    final_state,
                    policy=policy,
                    cfg_hash=cfg_hash,
                    format_hash=format_hash,
                    plan=plan,
                    stream=stream,
                    documents_consumed=documents_consumed,
                    last_record_start=last_record_start,
                    accepted=accepted,
                    complete=True,
                )
                if remote_store is not None:
                    mirror_shards(
                        remote_store,
                        output_dir=output_dir,
                        run_id=policy.run_id,
                        shard_entries=list(final_state.get("finalized_shards", [])),
                        configuration_hash=cfg_hash,
                        schema_hash=format_hash,
                        prune_unreferenced=True,
                    )
                write_json_atomic(manifest_path, manifest)
                write_json_atomic(progress_path, final_state)
                unlink_durable(backup_path)
                LOGGER.info(
                    "production dataset complete: accepted=%d shards=%d reason=%s",
                    accepted, len(manifest.get("shards", [])), completion_reason,
                )
                return manifest
            except BaseException:
                write_json_atomic(progress_path, safe_state)
                unlink_durable(manifest_path)
                raise
        finally:
            producer.close()
