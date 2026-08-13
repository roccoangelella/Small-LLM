"""Opt-in live Modal/HF smoke for the incremental 10B transport path."""
from __future__ import annotations

import json
import os
import sys
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import modal

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import launch  # noqa: E402
from cpu_supervision import await_stage_with_producer  # noqa: E402
from incremental_smoke_support import (  # noqa: E402
    APPROVED_WEIGHTS_SHA256,
    SMOKE_FIRST_SEGMENT_STEPS,
    SMOKE_MICROBATCH,
    SMOKE_NOMINAL_TRAINING_TOKENS,
    SMOKE_SECOND_SEGMENT_STEPS,
    SMOKE_SEQUENCES_PER_BLOCK,
    SMOKE_TARGET_SHARD_BYTES,
    SMOKE_TRAIN_BLOCKS,
    SMOKE_VALIDATION_PROBABILITY,
    producer_arguments,
    smoke_identity,
    validate_nonce,
    validate_smoke_run_id,
    wire_live_smoke_trainer_command,
)

app = modal.App("small-llm-incremental-live-smoke", image=launch.IMAGE)
REMOTE_REPO = launch.REMOTE_REPO
REMOTE_MODAL = launch.REMOTE_MODAL
RUN_ROOT = launch.RUN_ROOT
CACHE_ROOT = launch.CACHE_ROOT
RUN_VOLUME = launch.RUN_VOLUME
CACHE_VOLUME = launch.CACHE_VOLUME
TRAINING_SECRET = launch.TRAINING_SECRET
SMOKE_PRODUCER_ROOT = CACHE_ROOT / "incremental-smoke-producer"
SMOKE_DATA_ROOT = CACHE_ROOT / "incremental-smoke-data"


def _secret() -> tuple[str, str]:
    token = os.environ.get("HF_TOKEN", "").strip()
    repo_id = os.environ.get("SMALL_LLM_HF_REPO_ID", "").strip()
    if not token or not repo_id:
        raise RuntimeError("HF_TOKEN and SMALL_LLM_HF_REPO_ID are required")
    return token, repo_id


def _dataset_bucket(base_repo: str) -> str:
    return os.environ.get("SMALL_LLM_HF_DATASET_BUCKET_ID", "").strip() or f"{base_repo}-datasets"


def _checkpoint_transport(repo_id: str, source_commit: str | None = None) -> None:
    os.environ["SMALL_LLM_HF_REPO_ID"] = repo_id
    if source_commit:
        os.environ["SMALL_LLM_MODAL_SOURCE_COMMIT"] = source_commit
    sys.path.insert(0, str(REMOTE_MODAL))
    from model_repo_checkpoint import install_model_repo_checkpoint_transport

    install_model_repo_checkpoint_transport()


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise RuntimeError(f"expected JSON object: {path}")
    return dict(value)


@app.function(timeout=10 * 60, single_use_containers=True, secrets=[TRAINING_SECRET])
def preflight_remote(nonce: str) -> dict[str, object]:
    nonce = validate_nonce(nonce)
    token, base_repo = _secret()
    identity = smoke_identity(base_repo, nonce)
    bucket = _dataset_bucket(base_repo)
    from dataset.src.hf_bucket_shards import HuggingFaceBucketShardStore
    from huggingface_hub import HfApi

    store = HuggingFaceBucketShardStore(bucket, token=token, private=True, create_bucket=True)
    if store._list_files(prefix=f"run/{identity.dataset_run_id}/"):
        raise RuntimeError("generated smoke dataset prefix already exists")
    HfApi(token=token).create_repo(
        repo_id=identity.checkpoint_repo_id,
        repo_type="model",
        private=True,
        exist_ok=False,
    )
    return {
        "dataset_run_id": identity.dataset_run_id,
        "training_run_id": identity.training_run_id,
        "checkpoint_repo_id": identity.checkpoint_repo_id,
        "dataset_bucket_id": bucket,
        "h100_allocated": False,
    }


@app.function(
    cpu=4.0,
    memory=8192,
    timeout=60 * 60,
    single_use_containers=True,
    secrets=[TRAINING_SECRET],
    volumes={str(CACHE_ROOT): CACHE_VOLUME},
)
def produce_remote(dataset_run_id: str, dataset_bucket_id: str) -> dict[str, object]:
    validate_smoke_run_id(dataset_run_id)
    weights = REMOTE_REPO / "dataset" / "mixture_weights.json"
    from dataset.src.remote import sha256_path

    if sha256_path(weights) != APPROVED_WEIGHTS_SHA256:
        raise RuntimeError("smoke producer mixture weights differ from production")
    forbidden = {
        "dataset.src.streaming",
        "dataset.production.cli",
        "dataset.production.policy",
        "dataset.production.incremental_builder",
    }
    if forbidden.intersection(sys.modules):
        raise RuntimeError("smoke split override must precede dataset streaming imports")
    from dataset import config as dataset_config

    # Test-only and container-local: production remains at the frozen 0.1% split.
    dataset_config.VALIDATION_PROBABILITY = SMOKE_VALIDATION_PROBABILITY
    from dataset.production.cli import main as production_main
    from dataset.src.hf_bucket_shards import HuggingFaceBucketShardStore

    token, _ = _secret()
    store = HuggingFaceBucketShardStore(
        dataset_bucket_id, token=token, private=True, create_bucket=True
    )
    store._write_json(
        store.object_key(dataset_run_id, "smoke_contract.json"),
        {
            "version": 1,
            "kind": "incremental-infrastructure-smoke",
            "scientific_dataset": False,
            "validation_probability": SMOKE_VALIDATION_PROBABILITY,
            "context_length": 2048,
            "sequences_per_block": SMOKE_SEQUENCES_PER_BLOCK,
            "target_shard_bytes": SMOKE_TARGET_SHARD_BYTES,
            "planned_train_blocks": SMOKE_TRAIN_BLOCKS,
        },
    )
    code = production_main(
        producer_arguments(
            weights_file=weights,
            output_dir=SMOKE_PRODUCER_ROOT / dataset_run_id,
            dataset_run_id=dataset_run_id,
            dataset_bucket_id=dataset_bucket_id,
        ),
        durable_progress_hook=lambda: getattr(CACHE_VOLUME, "commit")(),
    )
    if code != 0:
        raise RuntimeError(f"smoke producer exited with status {code}")
    return {"status": "producer_complete"}


@app.function(
    timeout=30 * 60,
    single_use_containers=True,
    secrets=[TRAINING_SECRET],
    volumes={
        str(RUN_ROOT): RUN_VOLUME.with_mount_options(read_only=True),
        str(CACHE_ROOT): CACHE_VOLUME,
    },
)
def stage_remote(
    dataset_run_id: str,
    training_run_id: str,
    dataset_bucket_id: str,
    checkpoint_repo_id: str,
) -> dict[str, object]:
    validate_smoke_run_id(dataset_run_id)
    validate_smoke_run_id(training_run_id)
    _checkpoint_transport(checkpoint_repo_id)
    import rolling_dataset
    from dataset.incremental_frontier import SHARD_FRONTIER_FILENAME
    from dataset.incremental_stage import stage_incremental_window_when_ready, verify_incremental_stage
    from dataset.src.hf_bucket_shards import HuggingFaceBucketShardStore

    cursor = rolling_dataset.next_unconsumed_block(training_run_id=training_run_id, run_root=RUN_ROOT)
    start = int(cursor["next_block_id"])
    token, _ = _secret()
    store = HuggingFaceBucketShardStore(
        dataset_bucket_id, token=token, private=True, create_bucket=True
    )
    destination = SMOKE_DATA_ROOT / dataset_run_id
    staged = stage_incremental_window_when_ready(
        store=store,
        run_id=dataset_run_id,
        destination=destination,
        start_block_id=start,
        timeout_seconds=30 * 60,
        poll_seconds=2.0,
    )
    getattr(CACHE_VOLUME, "commit")()
    if staged.get("status") != "ready":
        raise RuntimeError(f"smoke CPU stage is not ready: {staged}")
    verified = verify_incremental_stage(
        destination=destination,
        bucket_id=dataset_bucket_id,
        run_id=dataset_run_id,
        required_train_block=start,
    )
    frontier = _json(destination / SHARD_FRONTIER_FILENAME)
    return {
        "status": "ready",
        "h100_dispatch_allowed": True,
        "h100_allocated": False,
        "dataset_dir": str(destination),
        "next_block_id": start,
        "checkpoint_source": cursor.get("source"),
        "producer_complete": bool(frontier.get("producer_complete")),
        "verification": verified,
    }


@app.function(
    gpu="H100",
    timeout=30 * 60,
    single_use_containers=True,
    secrets=[TRAINING_SECRET],
    volumes={str(RUN_ROOT): RUN_VOLUME, str(CACHE_ROOT): CACHE_VOLUME},
)
def train_segment_remote(
    *,
    source_commit: str,
    dataset_run_id: str,
    training_run_id: str,
    dataset_bucket_id: str,
    checkpoint_repo_id: str,
    expected_start_step: int,
    segment_steps: int,
    require_remote_restore: bool,
) -> dict[str, object]:
    validate_smoke_run_id(dataset_run_id)
    validate_smoke_run_id(training_run_id)
    _checkpoint_transport(checkpoint_repo_id, source_commit)
    import profiles
    import rolling_dataset
    import runtime as base_runtime
    from dataset.incremental_frontier import RUN_CONTRACT_FILENAME, SHARD_FRONTIER_FILENAME
    from dataset.incremental_stage import verify_incremental_stage

    dataset = SMOKE_DATA_ROOT / dataset_run_id
    run_dir = RUN_ROOT / training_run_id
    checkpoints = run_dir / "checkpoints"
    evidence = run_dir / "evidence"
    checkpoints.mkdir(parents=True, exist_ok=True)
    evidence.mkdir(parents=True, exist_ok=True)

    cursor = rolling_dataset.next_unconsumed_block(training_run_id=training_run_id, run_root=RUN_ROOT)
    if int(cursor["next_block_id"]) != expected_start_step:
        raise RuntimeError("smoke checkpoint cursor is not aligned to the requested segment")
    stage_verification = verify_incremental_stage(
        destination=dataset,
        bucket_id=dataset_bucket_id,
        run_id=dataset_run_id,
        required_train_block=expected_start_step,
    )

    restored: dict[str, Any] | None = None
    if require_remote_restore:
        restored = base_runtime._restore_hf_checkpoint_if_needed(training_run_id, run_dir)
        if not isinstance(restored, Mapping):
            raise RuntimeError("remote-only smoke resume did not restore a checkpoint")
        if restored.get("source") != "hf_model_repo" or int(restored.get("step", -1)) != expected_start_step:
            raise RuntimeError(f"smoke restored the wrong HF checkpoint: {restored}")
    elif expected_start_step != 0:
        raise RuntimeError("the first smoke segment must start at zero")

    latest_id, latest_step = base_runtime._latest_checkpoint(checkpoints)
    if latest_step != expected_start_step:
        raise RuntimeError("restored local checkpoint does not match the expected smoke step")
    contract = _json(dataset / RUN_CONTRACT_FILENAME)
    trainer = contract.get("trainer")
    if not isinstance(trainer, Mapping) or int(trainer.get("steps", -1)) != SMOKE_TRAIN_BLOCKS:
        raise RuntimeError("smoke run contract has the wrong trainer horizon")
    plan = {"trainer": dict(trainer)}
    model = profiles.MODEL_PRESETS[20_000_000]
    tokens = profiles.TokenPreset(
        SMOKE_NOMINAL_TRAINING_TOKENS,
        "SMOKE",
        "incremental-live-smoke",
        "hf_rolling_shards",
    )
    transport = run_dir / "hf_checkpoint_transport.json"
    base_runtime._write_hf_transport_manifest(
        transport,
        run_id=training_run_id,
        dataset=dataset,
        dataset_profile="incremental-live-smoke",
        source_commit=source_commit,
        microbatch_size=SMOKE_MICROBATCH,
        resume_parent_source_commit=None,
        bucket_id=checkpoint_repo_id,
    )
    command = base_runtime._trainer_command(
        model=model,
        tokens=tokens,
        dataset=dataset,
        plan=plan,
        checkpoint_dir=checkpoints,
        steps=segment_steps,
        microbatch=SMOKE_MICROBATCH,
        precision="fp16",
        wandb_run_id="smoke-disabled",
        gpu_tag="h100-smoke",
        online=False,
        resume=latest_id,
    )
    command = wire_live_smoke_trainer_command(
        command,
        dataset_bucket_id=dataset_bucket_id,
        dataset_run_id=dataset_run_id,
        remote_manifest=transport,
        checkpoint_repo_id=checkpoint_repo_id,
    )
    started = time.perf_counter()
    base_runtime._run(
        command,
        cwd=REMOTE_REPO,
        log_path=evidence / f"segment-{expected_start_step:08d}.log",
    )
    getattr(RUN_VOLUME, "commit")()
    getattr(CACHE_VOLUME, "commit")()

    final_id, final_step = base_runtime._latest_checkpoint(checkpoints)
    if final_step != expected_start_step + segment_steps:
        raise RuntimeError("smoke segment ended at the wrong durable checkpoint")
    pointer = base_runtime._hf_model_repo_store().read_json(f"run/{training_run_id}/latest.json")
    if not isinstance(pointer, Mapping) or pointer.get("checkpoint_id") != final_id:
        raise RuntimeError("HF latest pointer does not match the smoke checkpoint")

    eviction: dict[str, object] = {}
    if expected_start_step == 0:
        rows = _json(dataset / SHARD_FRONTIER_FILENAME).get("ready_train_shards")
        if not isinstance(rows, list) or len(rows) < 2:
            raise RuntimeError("smoke frontier lacks current+successor train shards")
        first, second = rows[0], rows[1]
        if not isinstance(first, Mapping) or not isinstance(second, Mapping):
            raise RuntimeError("smoke frontier train rows are malformed")
        if (dataset / str(first["filename"])).exists():
            raise RuntimeError("the consumed smoke shard was not evicted")
        if not (dataset / str(second["filename"])).is_file():
            raise RuntimeError("the smoke successor shard is not locally verified")
        eviction = {
            "consumed_shard": first["filename"],
            "consumed_shard_evicted": True,
            "successor_shard": second["filename"],
            "successor_present": True,
        }
    return {
        "status": "segment_complete",
        "checkpoint_id": final_id,
        "completed_steps": final_step,
        "elapsed_seconds": time.perf_counter() - started,
        "remote_restore": dict(restored) if isinstance(restored, Mapping) else None,
        "stage_verification": stage_verification,
        "eviction": eviction,
    }


@app.function(
    timeout=10 * 60,
    single_use_containers=True,
    secrets=[TRAINING_SECRET],
    volumes={str(RUN_ROOT): RUN_VOLUME},
)
def move_local_checkpoint_aside_remote(
    *, training_run_id: str, checkpoint_repo_id: str, expected_step: int
) -> dict[str, object]:
    validate_smoke_run_id(training_run_id)
    _checkpoint_transport(checkpoint_repo_id)
    import runtime as base_runtime

    expected_id = f"step-{expected_step:08d}"
    pointer = base_runtime._hf_model_repo_store().read_json(f"run/{training_run_id}/latest.json")
    if not isinstance(pointer, Mapping) or pointer.get("checkpoint_id") != expected_id:
        raise RuntimeError("expected HF pointer is absent before remote-resume probe")
    run_dir = RUN_ROOT / training_run_id
    backup = RUN_ROOT / f"{training_run_id}-local-backup-step-{expected_step:08d}"
    if not run_dir.is_dir() or backup.exists():
        raise RuntimeError("local smoke run cannot be moved into its unique backup path")
    os.replace(run_dir, backup)
    getattr(RUN_VOLUME, "commit")()
    return {
        "status": "local_state_moved_aside",
        "checkpoint_id": expected_id,
        "local_backup": str(backup),
        "remote_pointer_preserved": True,
    }


@app.local_entrypoint()
def main(dry_run: bool = False) -> None:
    nonce = uuid.uuid4().hex[:12]
    source_commit = launch._local_source_commit()
    dataset_run_id = f"smoke-incremental-dataset-{nonce}"
    training_run_id = f"smoke-incremental-train-{nonce}"
    print(json.dumps({
        "smoke": "100m-10b-incremental-live",
        "source_commit": source_commit,
        "dataset_run_id": dataset_run_id,
        "training_run_id": training_run_id,
        "model": "20M infrastructure probe",
        "h100_segments": [SMOKE_FIRST_SEGMENT_STEPS, SMOKE_SECOND_SEGMENT_STEPS],
        "target_shard_bytes": SMOKE_TARGET_SHARD_BYTES,
        "validation_probability": SMOKE_VALIDATION_PROBABILITY,
        "dry_run": dry_run,
    }, indent=2, sort_keys=True))
    if dry_run:
        return

    preflight = preflight_remote.remote(nonce)
    bucket = str(preflight["dataset_bucket_id"])
    checkpoint_repo = str(preflight["checkpoint_repo_id"])
    if preflight.get("dataset_run_id") != dataset_run_id or preflight.get("training_run_id") != training_run_id:
        raise RuntimeError("remote preflight identity differs from the local smoke nonce")

    producer_call = produce_remote.spawn(dataset_run_id, bucket)
    stage_call = stage_remote.spawn(dataset_run_id, training_run_id, bucket, checkpoint_repo)
    try:
        staged, producer_result = await_stage_with_producer(stage_call, producer_call, poll_seconds=2.0)
        if staged.get("h100_dispatch_allowed") is not True or int(staged.get("next_block_id", -1)) != 0:
            raise RuntimeError(f"smoke CPU gate refused the first H100 segment: {staged}")
        if staged.get("producer_complete") is True or producer_result is not None:
            raise RuntimeError("producer completed too early to exercise a live READY frontier")

        first = train_segment_remote.spawn(
            source_commit=source_commit,
            dataset_run_id=dataset_run_id,
            training_run_id=training_run_id,
            dataset_bucket_id=bucket,
            checkpoint_repo_id=checkpoint_repo,
            expected_start_step=0,
            segment_steps=SMOKE_FIRST_SEGMENT_STEPS,
            require_remote_restore=False,
        ).get()
        moved = move_local_checkpoint_aside_remote.remote(
            training_run_id=training_run_id,
            checkpoint_repo_id=checkpoint_repo,
            expected_step=SMOKE_FIRST_SEGMENT_STEPS,
        )
        restaged = stage_remote.remote(dataset_run_id, training_run_id, bucket, checkpoint_repo)
        if int(restaged.get("next_block_id", -1)) != SMOKE_FIRST_SEGMENT_STEPS:
            raise RuntimeError("remote pointer did not drive the second CPU stage")
        second = train_segment_remote.spawn(
            source_commit=source_commit,
            dataset_run_id=dataset_run_id,
            training_run_id=training_run_id,
            dataset_bucket_id=bucket,
            checkpoint_repo_id=checkpoint_repo,
            expected_start_step=SMOKE_FIRST_SEGMENT_STEPS,
            segment_steps=SMOKE_SECOND_SEGMENT_STEPS,
            require_remote_restore=True,
        ).get()
        restored = second.get("remote_restore")
        if not isinstance(restored, Mapping) or restored.get("source") != "hf_model_repo":
            raise RuntimeError("second H100 segment did not prove HF model-repository restore")
        print(json.dumps({
            "status": "passed",
            "dataset_run_id": dataset_run_id,
            "training_run_id": training_run_id,
            "dataset_bucket_id": bucket,
            "checkpoint_repo_id": checkpoint_repo,
            "first_segment": first,
            "local_checkpoint_move": moved,
            "second_stage": restaged,
            "second_segment": second,
            "artifacts": "preserved under isolated smoke identities for audit",
        }, indent=2, sort_keys=True))
    finally:
        try:
            producer_call.cancel(terminate_containers=True)
        except Exception:
            pass
