"""Concurrent-capable production builder for the 10B incremental shard frontier.

The legacy production builder remains unchanged for already-qualified finite
runs.  This path extends the same deterministic producer with one additional
contract: every durable checkpoint publishes a monotonic READY shard frontier,
and source ingestion continues until both the frozen training-block horizon and
the source-token target are satisfied.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Mapping

from dataset import config
from dataset.incremental_frontier import (
    build_run_contract,
    publish_frontier,
    publish_run_contract,
)
from dataset.src.bytesource import RangeReader, SourceFile
from dataset.src.remote import RemoteShardStore, ensure_safe_directory
from dataset.src.storage import read_json, write_json_atomic
from dataset.src.streaming import StreamCacheConfig, StreamCacheProducer, parallel_read_documents
from dataset.src.workplan import WorkPlan

from .builder import (
    _decorate_state,
    _durability_manifest,
    _evict_verified_local_shards,
    _production_identity,
    _validate_cursor,
    _validate_manifest_remote_coverage,
    _validate_remote_inventory,
)
from .policy import (
    ProductionPolicy,
    configuration_hash,
    incorporated_source_tokens,
    schema_hash,
)
from .remote import mirror_shards
from .remote_resume import restore_remote_evicted_producer
from .safety import (
    PROGRESS_BACKUP_FILENAME,
    RunLock,
    discard_uncheckpointed_artifacts,
    recover_progress_backup,
    unlink_durable,
)

LOGGER = logging.getLogger(__name__)


def build_incremental_production_cache(
    output_dir: Path | str,
    stream: StreamCacheConfig,
    policy: ProductionPolicy,
    plan: WorkPlan,
    reader_factory: Callable[[SourceFile], RangeReader],
    *,
    remote_store: RemoteShardStore,
    frontier_store: object,
    nominal_training_tokens: int,
    training_validation_blocks: int,
    resume: bool = False,
    simulate_crash_after_documents: int | None = None,
) -> dict[str, object]:
    """Build a remotely durable prefix while publishing READY shards incrementally."""

    if not policy.remote_required:
        raise RuntimeError("incremental production requires remote durability")
    if nominal_training_tokens <= 0 or training_validation_blocks <= 0:
        raise ValueError("incremental training horizon and validation block count must be positive")
    if simulate_crash_after_documents is not None and simulate_crash_after_documents <= 0:
        raise ValueError("simulate_crash_after_documents must be positive")

    output_dir = Path(output_dir)
    ensure_safe_directory(output_dir)
    progress_path = output_dir / config.PROGRESS_FILENAME
    manifest_path = output_dir / config.MANIFEST_FILENAME
    cfg_hash = configuration_hash(policy, stream, plan)
    format_hash = schema_hash(stream)
    contract = build_run_contract(
        run_id=policy.run_id,
        nominal_training_tokens=nominal_training_tokens,
        target_source_tokens=policy.target_source_tokens,
        minimum_source_tokens=policy.minimum_source_tokens,
        maximum_source_tokens=policy.maximum_source_tokens,
        checkpoint_source_tokens=policy.checkpoint_source_tokens,
        context_length=stream.context_length,
        sequences_per_block=stream.sequences_per_block,
        target_shard_bytes=stream.target_shard_bytes,
        configuration_hash=cfg_hash,
        schema_hash=format_hash,
        work_plan_hash=plan.hash,
        validation_blocks=training_validation_blocks,
    )
    planned_blocks = int(contract["planned_train_blocks"])
    write_json_atomic(output_dir / "run_contract.json", contract)
    publish_run_contract(frontier_store, run_id=policy.run_id, contract=contract)

    with RunLock(output_dir):
        if resume:
            recover_progress_backup(output_dir)
            state = read_json(progress_path)
            if not isinstance(state, Mapping):
                raise ValueError("production progress.json must be an object")
            expected = _production_identity(policy, cfg_hash, format_hash)
            if state.get("production") != expected:
                raise ValueError("production progress configuration does not match this invocation")
            saved_contract_hash = state.get("incremental_contract_sha256")
            if saved_contract_hash not in {None, contract["contract_sha256"]}:
                raise ValueError("production progress belongs to a different incremental run contract")
            cursor = _validate_cursor(state.get("source_reader"), plan=plan, stream=stream)
            durable = _durability_manifest(output_dir)
            _validate_remote_inventory(remote_store, run_id=policy.run_id, durability_manifest=durable)
            if state.get("complete") is True:
                manifest = read_json(manifest_path)
                if not isinstance(manifest, dict):
                    raise ValueError("completed production cache has an invalid manifest")
                _validate_manifest_remote_coverage(manifest, durable)
                _validate_remote_inventory(remote_store, run_id=policy.run_id, durability_manifest=durable)
                return manifest
            discard_uncheckpointed_artifacts(output_dir, state)
            producer = restore_remote_evicted_producer(
                output_dir,
                stream,
                state,
                durability_manifest=durable,
            )
            # Reassert the remote frontier from the exact durable resume state.
            publish_frontier(
                frontier_store,
                run_id=policy.run_id,
                contract=contract,
                durability_manifest=durable,
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

        def training_horizon_reached() -> bool:
            return producer.last_durable_block_id + 1 >= planned_blocks

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
            state["incremental_contract_sha256"] = contract["contract_sha256"]
            state["planned_train_blocks"] = planned_blocks
            state["training_horizon_reached"] = training_horizon_reached()
            durable = mirror_shards(
                remote_store,
                output_dir=output_dir,
                run_id=policy.run_id,
                shard_entries=list(state.get("finalized_shards", [])),
                configuration_hash=cfg_hash,
                schema_hash=format_hash,
            )
            publish_frontier(
                frontier_store,
                run_id=policy.run_id,
                contract=contract,
                durability_manifest=durable,
            )
            # Producer progress is committed before local eviction.  Therefore
            # every READY frontier entry is reconstructable after a crash.
            write_json_atomic(progress_path, state)
            removed = _evict_verified_local_shards(
                output_dir,
                state=state,
                durability_manifest=durable,
            )
            last_checkpoint_tokens = accepted
            LOGGER.info(
                "incremental durable checkpoint: documents=%d accepted=%d train_blocks=%d/%d shards=%d evicted=%d",
                documents_consumed,
                accepted,
                producer.last_durable_block_id + 1,
                planned_blocks,
                len(state.get("finalized_shards", [])),
                removed,
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
                if incorporated >= policy.target_source_tokens and training_horizon_reached():
                    completion_reason = "source_target_and_train_horizon_reached"
                    break
                if incorporated + document.source_token_count > policy.maximum_source_tokens:
                    if not training_horizon_reached():
                        durable_state()
                        raise RuntimeError(
                            "incremental producer hit the source-token hard maximum before the frozen "
                            f"training horizon: blocks={producer.last_durable_block_id + 1}/{planned_blocks}"
                        )
                    completion_reason = "hard_maximum_after_train_horizon"
                    break

                if is_validation_doc:
                    producer.add_validation_document(document)
                else:
                    queue_for_cluster = producer._queues[document.cluster_id]
                    if len(queue_for_cluster) >= stream.per_cluster_queue_limit:
                        drained = producer.drain_training(
                            force=True,
                            maximum_documents=1,
                            cluster_id=document.cluster_id,
                        )
                        if drained != 1 or len(queue_for_cluster) >= stream.per_cluster_queue_limit:
                            raise RuntimeError(
                                f"could not free training queue for cluster {document.cluster_id}"
                            )
                    producer.add_training_document(document)
                    producer.drain_training(force=False, maximum_documents=1)

                documents_consumed += 1
                last_record_start[str(document.work_item_index)] = document.record_start
                accepted = incorporated_source_tokens(producer)
                if documents_consumed % 10_000 == 0:
                    LOGGER.info(
                        "incremental dataset progress: documents=%d accepted=%d train_blocks=%d/%d queued=%d",
                        documents_consumed,
                        accepted,
                        producer.last_durable_block_id + 1,
                        planned_blocks,
                        producer.queued_source_tokens,
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
            if accepted_before_finish < policy.target_source_tokens or not training_horizon_reached():
                durable_state()
                raise RuntimeError(
                    "source ended before satisfying the incremental run contract: "
                    f"accepted={accepted_before_finish}/{policy.target_source_tokens}, "
                    f"train_blocks={producer.last_durable_block_id + 1}/{planned_blocks}"
                )

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
                raw_shards = manifest.get("shards")
                if not isinstance(raw_shards, list):
                    raise RuntimeError("completed incremental manifest has no shard inventory")
                final_train_blocks = max(
                    (
                        int(row["last_block_id"]) + 1
                        for row in raw_shards
                        if isinstance(row, Mapping) and row.get("split") == "train"
                    ),
                    default=0,
                )
                if final_train_blocks < planned_blocks:
                    raise RuntimeError("finalized incremental corpus does not cover the frozen train horizon")

                manifest["work_plan_hash"] = plan.hash
                manifest["production"] = {
                    "version": 1,
                    "run_id": policy.run_id,
                    "configuration_hash": cfg_hash,
                    "schema_hash": format_hash,
                    "target_source_tokens": policy.target_source_tokens,
                    "minimum_source_tokens": policy.minimum_source_tokens,
                    "maximum_source_tokens": policy.maximum_source_tokens,
                    "checkpoint_source_tokens": policy.checkpoint_source_tokens,
                    "target_reached": accepted >= policy.target_source_tokens,
                    "completion_reason": completion_reason,
                    "remote_required": True,
                    "incremental_frontier": True,
                    "contract_sha256": contract["contract_sha256"],
                    "planned_train_blocks": planned_blocks,
                    "planned_train_target_tokens": contract["planned_train_target_tokens"],
                    "final_train_blocks": final_train_blocks,
                    "unused_final_train_tail_blocks": final_train_blocks - planned_blocks,
                    "training_horizon_reached": True,
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
                final_state["incremental_contract_sha256"] = contract["contract_sha256"]
                final_state["planned_train_blocks"] = planned_blocks
                final_state["training_horizon_reached"] = True
                durable = mirror_shards(
                    remote_store,
                    output_dir=output_dir,
                    run_id=policy.run_id,
                    shard_entries=list(final_state.get("finalized_shards", [])),
                    configuration_hash=cfg_hash,
                    schema_hash=format_hash,
                    prune_unreferenced=True,
                )
                # Publish the last active-shard bytes as READY, but leave the
                # frontier active until the CLI writes and read-back verifies the
                # final enriched manifest and can bind its exact SHA-256.
                publish_frontier(
                    frontier_store,
                    run_id=policy.run_id,
                    contract=contract,
                    durability_manifest=durable,
                )
                write_json_atomic(manifest_path, manifest)
                _validate_manifest_remote_coverage(manifest, durable)
                _validate_remote_inventory(
                    remote_store,
                    run_id=policy.run_id,
                    durability_manifest=durable,
                )
                write_json_atomic(progress_path, final_state)
                _evict_verified_local_shards(
                    output_dir,
                    state=final_state,
                    durability_manifest=durable,
                )
                unlink_durable(backup_path)
                LOGGER.info(
                    "incremental production complete: accepted=%d train_blocks=%d planned=%d shards=%d",
                    accepted,
                    final_train_blocks,
                    planned_blocks,
                    len(raw_shards),
                )
                return manifest
            except BaseException:
                write_json_atomic(progress_path, safe_state)
                unlink_durable(manifest_path)
                raise
        finally:
            producer.close()


__all__ = ["build_incremental_production_cache"]
