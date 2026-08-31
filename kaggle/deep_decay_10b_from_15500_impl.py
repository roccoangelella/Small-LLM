#!/usr/bin/env python3
"""Resume the 100M/10B deep-decay continuation on Kaggle 2xT4.

This is an execution-topology migration of ADR 0095. Restore the exact original
uncooled step-15,500 state, fork it into the deep-decay run namespace, stage the
checkpoint-aligned rolling 10B dataset from Hugging Face, and preserve the
64-sequence global optimizer block. The original checkpoint used execution
microbatch four; Kaggle uses microbatch two because the live 100M T4 evidence
ruled out the local 4x2048 shape on a 14.56-GiB T4.

Usage from repository root:

    python kaggle/deep_decay_10b_from_15500.py --dry-run
    python kaggle/deep_decay_10b_from_15500.py
    python kaggle/deep_decay_10b_from_15500.py --max-steps-this-session 250

Rerunning the same command resumes from the newest manifest-verified Kaggle
deep-decay checkpoint in the Hugging Face checkpoint Bucket. A newer legacy
model-repository checkpoint is accepted only as a migration source. If the
Kaggle continuation has never started, the launcher accepts only the exact
source step-00015500 checkpoint.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
KAGGLE = ROOT / "kaggle"
BEAM = ROOT / "beam"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(KAGGLE) not in sys.path:
    sys.path.insert(0, str(KAGGLE))

import dual_t4_runtime

SOURCE_RUN_ID = "100m-10b-data-001"
SOURCE_STEP = 15_500
SOURCE_CHECKPOINT_ID = f"step-{SOURCE_STEP:08d}"
RUN_ID = "100m-10b-deep-decay-from-step15500"
DATASET_PROFILE = "modal-10b-b64"
DATASET_RUN_ID = "modal-10b-b64-dataset-001"
SOURCE_MICROBATCH_SIZE = 4
MICROBATCH_SIZE = 2
SEQUENCES_PER_BLOCK = 64
CONTEXT_LENGTH = 2048
TARGETS_PER_FULL_BLOCK = SEQUENCES_PER_BLOCK * CONTEXT_LENGTH
PEAK_LR = 3e-4
SETTLE_LR = 1e-4
SETTLE_LR_RATIO = SETTLE_LR / PEAK_LR
COOLDOWN_START_LR = 1e-5
FINAL_LR = 5e-6
MINIMUM_LR_RATIO = FINAL_LR / PEAK_LR
TOTAL_TARGETS = 10_000_007_168
FINAL_STEP = 76_294
SOURCE_EXPECTED_TOKENS = SOURCE_STEP * TARGETS_PER_FULL_BLOCK
REQUESTED_SETTLE_TOKENS = 300_000_000
SETTLE_STEPS = math.ceil(REQUESTED_SETTLE_TOKENS / TARGETS_PER_FULL_BLOCK)
SETTLE_TOKENS = SETTLE_STEPS * TARGETS_PER_FULL_BLOCK
SETTLE_END_STEP = SOURCE_STEP + SETTLE_STEPS
SETTLE_END_TOKENS = SOURCE_EXPECTED_TOKENS + SETTLE_TOKENS
COOLDOWN_STEPS = 3_052
COOLDOWN_TOKENS = COOLDOWN_STEPS * TARGETS_PER_FULL_BLOCK
COOLDOWN_START_STEP = FINAL_STEP - COOLDOWN_STEPS
COOLDOWN_START_TOKENS = COOLDOWN_START_STEP * TARGETS_PER_FULL_BLOCK
ADDITIONAL_STEPS = FINAL_STEP - SOURCE_STEP
BASE_POWER = math.log(SETTLE_LR / COOLDOWN_START_LR) / math.log(
    COOLDOWN_START_TOKENS / SETTLE_END_TOKENS
)
REMOTE_EVERY = 250

WORK_ROOT = Path(
    os.environ.get("SMALL_LLM_KAGGLE_WORK_ROOT", "/kaggle/working/small-llm")
).expanduser()
RUN_DIR = WORK_ROOT / "runs" / RUN_ID
CHECKPOINT_DIR = RUN_DIR / "checkpoints"
SOURCE_CACHE_DIR = WORK_ROOT / "source" / SOURCE_RUN_ID
DATA_CACHE_ROOT = WORK_ROOT / "datasets"
CONTRACT_PATH = RUN_DIR / "deep_decay_10b_contract.json"

if FINAL_STEP * TARGETS_PER_FULL_BLOCK != TOTAL_TARGETS:
    raise RuntimeError("frozen 10B endpoint is not block-aligned")
if SETTLE_END_TOKENS >= COOLDOWN_START_TOKENS:
    raise RuntimeError("settling phase overlaps terminal cooldown")
if COOLDOWN_START_TOKENS + COOLDOWN_TOKENS != TOTAL_TARGETS:
    raise RuntimeError("terminal cooldown does not end at exact 10B")


def _beam_runtime() -> Any:
    """Import Beam's provider-neutral runtime helpers without importing Beam SDK."""
    beam = str(BEAM)
    if beam in sys.path:
        sys.path.remove(beam)
    sys.path.insert(0, beam)
    import runtime as runtime_base  # type: ignore

    runtime_path = Path(runtime_base.__file__).resolve()
    expected = (BEAM / "runtime.py").resolve()
    if runtime_path != expected:
        raise RuntimeError(f"expected provider-neutral runtime {expected}, got {runtime_path}")
    return runtime_base


def _expected_lr(tokens: int) -> float:
    if tokens <= SOURCE_EXPECTED_TOKENS:
        return PEAK_LR
    if tokens <= SETTLE_END_TOKENS:
        progress = (tokens - SOURCE_EXPECTED_TOKENS) / SETTLE_TOKENS
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return SETTLE_LR + (PEAK_LR - SETTLE_LR) * cosine
    base_lr = SETTLE_LR * (SETTLE_END_TOKENS / tokens) ** BASE_POWER
    if tokens <= COOLDOWN_START_TOKENS:
        return base_lr
    progress = min(1.0, max(0.0, (tokens - COOLDOWN_START_TOKENS) / COOLDOWN_TOKENS))
    return FINAL_LR + (COOLDOWN_START_LR - FINAL_LR) * (1.0 - progress)


def _contract() -> dict[str, object]:
    return {
        "version": 3,
        "kind": "step15500_deep_decay_settle_power_linear_cooldown",
        "execution": "kaggle_dual_t4_ddp_block64",
        "source_run_id": SOURCE_RUN_ID,
        "source_checkpoint_id": SOURCE_CHECKPOINT_ID,
        "source_step": SOURCE_STEP,
        "source_expected_consumed_tokens": SOURCE_EXPECTED_TOKENS,
        "source_microbatch_size": SOURCE_MICROBATCH_SIZE,
        "run_id": RUN_ID,
        "dataset_profile": DATASET_PROFILE,
        "dataset_run_id": DATASET_RUN_ID,
        "sequences_per_block": SEQUENCES_PER_BLOCK,
        "world_size": 2,
        "sequences_per_rank": SEQUENCES_PER_BLOCK // 2,
        "microbatch_size": MICROBATCH_SIZE,
        "local_microbatches_per_rank": SEQUENCES_PER_BLOCK // 2 // MICROBATCH_SIZE,
        "schedule": "wsqd",
        "schedule_anchor_tokens": SOURCE_EXPECTED_TOKENS,
        "anchor_lr": PEAK_LR,
        "settle_requested_tokens": REQUESTED_SETTLE_TOKENS,
        "settle_steps": SETTLE_STEPS,
        "settle_tokens": SETTLE_TOKENS,
        "settle_end_step": SETTLE_END_STEP,
        "settle_end_tokens": SETTLE_END_TOKENS,
        "settle_lr_ratio": SETTLE_LR_RATIO,
        "settle_lr": SETTLE_LR,
        "base_power": BASE_POWER,
        "cooldown_start_step": COOLDOWN_START_STEP,
        "cooldown_start_tokens": COOLDOWN_START_TOKENS,
        "lr_at_cooldown_start": COOLDOWN_START_LR,
        "cooldown_steps": COOLDOWN_STEPS,
        "decay_tokens": COOLDOWN_TOKENS,
        "minimum_lr_ratio": MINIMUM_LR_RATIO,
        "final_lr": FINAL_LR,
        "final_step": FINAL_STEP,
        "final_targets": TOTAL_TARGETS,
        "additional_steps": ADDITIONAL_STEPS,
        "scientific_change": "ADR0095 scheduler; Kaggle changes execution slicing 4->2 while preserving the exact 64-sequence optimizer block",
    }


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _checkpoint_step(checkpoint_id: str) -> int:
    if not checkpoint_id.startswith("step-") or len(checkpoint_id) != 13:
        raise RuntimeError(f"invalid checkpoint ID: {checkpoint_id!r}")
    try:
        return int(checkpoint_id.removeprefix("step-"))
    except ValueError as error:
        raise RuntimeError(f"invalid checkpoint ID: {checkpoint_id!r}") from error


def _remote_checkpoint_state(runtime_base: Any, *, run_id: str) -> dict[str, object] | None:
    """Select the newest pointer, preferring Bucket latest on a step tie."""

    candidates: list[tuple[int, int, object, Mapping[str, object], str, str]] = []
    for priority, store, source, expected_transport in (
        (
            1,
            runtime_base._hf_bucket_store(),
            "hf_bucket",
            "modal-hf-bucket-checkpoint-v1",
        ),
        (
            0,
            runtime_base._hf_model_repo_store(),
            "legacy_hf_model_repo",
            "modal-hf-checkpoint-v1",
        ),
    ):
        pointer = store.read_json(f"run/{run_id}/latest.json")
        if pointer is None:
            continue
        if not isinstance(pointer, Mapping):
            raise RuntimeError(f"{source} pointer for {run_id} is not an object")
        checkpoint_id = pointer.get("checkpoint_id")
        if not isinstance(checkpoint_id, str):
            raise RuntimeError(f"{source} pointer for {run_id} has no checkpoint_id")
        candidates.append(
            (
                _checkpoint_step(checkpoint_id),
                priority,
                store,
                pointer,
                source,
                expected_transport,
            )
        )
    if not candidates:
        return None
    step, _, store, pointer, source, expected_transport = max(
        candidates,
        key=lambda item: (item[0], item[1]),
    )
    return {
        "checkpoint_id": pointer["checkpoint_id"],
        "step": step,
        "store": store,
        "pointer": pointer,
        "source": source,
        "expected_transport": expected_transport,
    }


def _restore_pointer(runtime_base: Any, *, run_id: str, destination_run_dir: Path, require_checkpoint_id: str | None = None) -> dict[str, object] | None:
    from dataset.src.joint_checkpoint import restore_on_empty_vps
    from dataset.src.remote import TwoPhaseCheckpointPublisher

    remote_state = _remote_checkpoint_state(runtime_base, run_id=run_id)
    if remote_state is None:
        return None
    store = remote_state["store"]
    pointer = remote_state["pointer"]
    checkpoint_id = remote_state["checkpoint_id"]
    assert isinstance(pointer, Mapping) and isinstance(checkpoint_id, str)
    if require_checkpoint_id is not None and checkpoint_id != require_checkpoint_id:
        raise RuntimeError(f"exact source checkpoint required: expected {require_checkpoint_id}, HF latest points to {checkpoint_id}")

    destination_run_dir.mkdir(parents=True, exist_ok=True)
    local_id, local_step = runtime_base._latest_checkpoint(destination_run_dir / "checkpoints")
    if local_id is not None:
        return {"checkpoint_id": local_id, "step": local_step, "source": "local"}

    restored = restore_on_empty_vps(
        publisher=TwoPhaseCheckpointPublisher(store, run_id=run_id),
        store=None,
        run_id=run_id,
        destination=destination_run_dir,
        checkpoint_pointer=pointer,
        prefetch_shards=0,
    )
    metadata = runtime_base._verified_checkpoint_metadata(restored, checkpoint_id)
    transport = json.loads((restored / "drive_manifest.json").read_text(encoding="utf-8"))
    if (
        not isinstance(transport, Mapping)
        or transport.get("transport") != remote_state["expected_transport"]
    ):
        raise RuntimeError("restored Kaggle checkpoint has the wrong HF transport identity")
    metadata.update(source=remote_state["source"])
    return metadata


def _verify_deep_decay_checkpoint(runtime_base: Any, checkpoint_id: str) -> int:
    del runtime_base
    from trainer.state import load_trainer_state_file, release_host_memory

    step = _checkpoint_step(checkpoint_id)
    if not SOURCE_STEP <= step <= FINAL_STEP:
        raise RuntimeError(f"deep-decay checkpoint step {step} is outside the frozen horizon")
    root = CHECKPOINT_DIR / checkpoint_id
    state = load_trainer_state_file(root / "trainer_state.pkl", map_location="cpu")
    try:
        config = state.get("config")
        scheduler = state.get("scheduler")
        if state.get("global_step") != step:
            raise RuntimeError("deep-decay checkpoint global_step disagrees with checkpoint ID")
        if state.get("consumed_tokens") != step * TARGETS_PER_FULL_BLOCK:
            raise RuntimeError("deep-decay checkpoint consumed-token count drifted")
        if not isinstance(config, Mapping) or not isinstance(scheduler, Mapping):
            raise RuntimeError("deep-decay checkpoint lacks config/scheduler mappings")
        expected = {
            "microbatch_size": MICROBATCH_SIZE,
            "schedule": "wsqd",
            "learning_rate": PEAK_LR,
            "warmup_tokens": 0,
            "stable_tokens": 0,
            "decay_tokens": COOLDOWN_TOKENS,
            "schedule_anchor_tokens": SOURCE_EXPECTED_TOKENS,
            "cooldown_start_tokens": COOLDOWN_START_TOKENS,
            "settle_tokens": SETTLE_TOKENS,
            "settle_lr_ratio": SETTLE_LR_RATIO,
            "base_power": BASE_POWER,
        }
        drift = {key: (config.get(key), value) for key, value in expected.items() if config.get(key) != value}
        if drift:
            raise RuntimeError("deep-decay checkpoint scientific/execution config drifted: " + json.dumps(drift, sort_keys=True))
        if scheduler.get("committed_tokens") != step * TARGETS_PER_FULL_BLOCK:
            raise RuntimeError("deep-decay scheduler committed_tokens drifted")
    finally:
        del state
        release_host_memory()
    return step


def _stage_dataset(runtime_base: Any, *, start_block_id: int) -> Path:
    from dataset.incremental_stage import stage_incremental_window_when_ready, verify_incremental_stage
    from dataset.src.hf_bucket_shards import HuggingFaceBucketShardStore

    bucket_id = os.environ.get("SMALL_LLM_HF_DATASET_BUCKET_ID", "").strip() or f"{runtime_base._hf_model_repo_id()}-datasets"
    store = HuggingFaceBucketShardStore(bucket_id, token=runtime_base._hf_token(), private=True, create_bucket=False)
    destination = DATA_CACHE_ROOT / DATASET_RUN_ID / f"from-{start_block_id:08d}"
    staged = stage_incremental_window_when_ready(store=store, run_id=DATASET_RUN_ID, destination=destination, start_block_id=start_block_id)
    if staged.get("status") != "ready":
        raise RuntimeError(f"dataset stage did not become ready: {staged}")
    verify_incremental_stage(destination=destination, bucket_id=bucket_id, run_id=DATASET_RUN_ID, required_train_block=start_block_id)
    return destination


def _fork_source_checkpoint(runtime_base: Any, *, source_root: Path, dataset: Path) -> None:
    del runtime_base
    import torch
    from dataset.src.joint_checkpoint import verify_local_manifest
    from dataset.src.remote import sha256_path
    from trainer.identity import canonical_hash
    from trainer.state import load_trainer_state_file, release_host_memory

    source = source_root / "checkpoints" / SOURCE_CHECKPOINT_ID
    verify_local_manifest(source)
    source_payload = json.loads((source / "checkpoint.json").read_text(encoding="utf-8"))
    if not isinstance(source_payload, Mapping):
        raise RuntimeError("source checkpoint.json is not an object")
    pipeline = source_payload.get("pipeline_state")
    if not isinstance(pipeline, Mapping) or pipeline.get("last_consumed_block_id") != SOURCE_STEP - 1:
        raise RuntimeError("step-15500 source has the wrong data cursor")

    state = load_trainer_state_file(source / "trainer_state.pkl", map_location="cpu")
    try:
        if state.get("global_step") != SOURCE_STEP:
            raise RuntimeError("step-15500 source has the wrong global_step")
        if state.get("consumed_tokens") != SOURCE_EXPECTED_TOKENS:
            raise RuntimeError("step-15500 source has the wrong consumed-token count")
        config = state.get("config")
        scheduler = state.get("scheduler")
        model_config = state.get("model_config")
        if not isinstance(config, Mapping) or not isinstance(scheduler, Mapping):
            raise RuntimeError("source trainer state lacks config/scheduler mappings")
        if not isinstance(model_config, Mapping):
            raise RuntimeError("source trainer state lacks model_config")
        if config.get("microbatch_size") != SOURCE_MICROBATCH_SIZE:
            raise RuntimeError("source checkpoint no longer has the expected microbatch four")
        if config.get("learning_rate") != PEAK_LR or config.get("schedule") != "wsd":
            raise RuntimeError("source checkpoint is not the expected uncooled 3e-4 WSD state")

        patched_config = dict(config)
        patched_config.update(
            microbatch_size=MICROBATCH_SIZE,
            schedule="wsqd",
            warmup_tokens=0,
            stable_tokens=0,
            decay_tokens=COOLDOWN_TOKENS,
            minimum_lr_ratio=MINIMUM_LR_RATIO,
            schedule_anchor_tokens=SOURCE_EXPECTED_TOKENS,
            cooldown_start_tokens=COOLDOWN_START_TOKENS,
            settle_tokens=SETTLE_TOKENS,
            settle_lr_ratio=SETTLE_LR_RATIO,
            base_power=BASE_POWER,
        )
        patched_scheduler = dict(scheduler)
        patched_scheduler["config"] = dict(patched_config)
        patched_scheduler["committed_tokens"] = SOURCE_EXPECTED_TOKENS
        patched_scheduler["last_lr"] = PEAK_LR
        state["config"] = patched_config
        state["scheduler"] = patched_scheduler

        manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
        production = manifest.get("production") if isinstance(manifest, Mapping) else None
        dataset_configuration_hash = production.get("configuration_hash") if isinstance(production, Mapping) else None
        new_configuration_hash = canonical_hash({"version": 1, "model": dict(model_config), "trainer": patched_config, "dataset_configuration_hash": dataset_configuration_hash})

        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        target = CHECKPOINT_DIR / SOURCE_CHECKPOINT_ID
        if target.exists():
            verify_local_manifest(target)
            return
        staging = CHECKPOINT_DIR / f".{SOURCE_CHECKPOINT_ID}.fork"
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True, exist_ok=False)
        trainer_state_path = staging / "trainer_state.pkl"
        torch.save(state, trainer_state_path)
        checkpoint_payload = dict(source_payload)
        checkpoint_payload["configuration_hash"] = new_configuration_hash
        _write_json(staging / "checkpoint.json", checkpoint_payload)
        _write_json(staging / "local_manifest.json", {"files": [{"name": "trainer_state.pkl", "sha256": sha256_path(trainer_state_path)}, {"name": "checkpoint.json", "sha256": sha256_path(staging / "checkpoint.json")}]})
        verify_local_manifest(staging)
        os.replace(staging, target)
    finally:
        del state
        release_host_memory()


def _replace_option(command: list[str], option: str, value: str) -> None:
    try:
        index = command.index(option)
    except ValueError as error:
        raise RuntimeError(f"trainer command lacks required option {option}") from error
    if index + 1 >= len(command):
        raise RuntimeError(f"trainer command option {option} has no value")
    command[index + 1] = value


def _replace_tag(command: list[str], old: str, new: str) -> None:
    try:
        index = command.index("--wandb-tags") + 1
    except ValueError:
        return
    while index < len(command) and not command[index].startswith("--"):
        if command[index] == old:
            command[index] = new
        index += 1


def _dual_t4_command(trainer_command: Sequence[str]) -> list[str]:
    command = list(trainer_command)
    if len(command) < 3 or command[1:3] != ["-m", "trainer"]:
        raise RuntimeError("provider-neutral trainer command no longer begins with `python -m trainer`")
    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError("uv is required for the qualified Kaggle two-T4 runtime")
    return [
        uv,
        "run",
        "--python",
        "3.13",
        "--no-project",
        *dual_t4_runtime.qualified_runtime_uv_args(),
        "python",
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={dual_t4_runtime.WORLD_SIZE}",
        str(KAGGLE / "dual_t4_train_block64.py"),
        "--worktree",
        str(ROOT),
        *command[3:],
    ]


def _publish_latest_to_bucket(
    runtime_base: Any,
    *,
    checkpoint_id: str,
    dataset: Path,
) -> dict[str, object]:
    """Make the verified Kaggle execution state the durable Bucket latest."""

    store = runtime_base._hf_bucket_store()
    pointer = store.read_json(f"run/{RUN_ID}/latest.json")
    if pointer is not None and not isinstance(pointer, Mapping):
        raise RuntimeError("Kaggle checkpoint Bucket latest pointer is not an object")
    current_id = pointer.get("checkpoint_id") if isinstance(pointer, Mapping) else None
    force_id = os.environ.get("SMALL_LLM_KAGGLE_PROVIDER_MIGRATION_CHECKPOINT_ID", "").strip()
    checkpoint_root = CHECKPOINT_DIR / checkpoint_id
    source_commit = os.environ.get("SMALL_LLM_SOURCE_COMMIT", "kaggle-main")
    previous_source_commit: str | None = None
    previous_payload: Mapping[str, object] | None = None
    previous_transport = checkpoint_root / "drive_manifest.json"
    if previous_transport.is_file():
        raw_previous_payload = json.loads(previous_transport.read_text(encoding="utf-8"))
        previous_payload = raw_previous_payload if isinstance(raw_previous_payload, Mapping) else None
        observed_source = (
            previous_payload.get("source_commit")
            if isinstance(previous_payload, Mapping)
            else None
        )
        if isinstance(observed_source, str) and observed_source and observed_source != source_commit:
            previous_source_commit = observed_source
    checkpoint_bucket_id = runtime_base._hf_checkpoint_bucket_id()
    local_bucket_transport_current = (
        isinstance(previous_payload, Mapping)
        and previous_payload.get("transport") == "modal-hf-bucket-checkpoint-v1"
        and previous_payload.get("bucket_id") == checkpoint_bucket_id
        and previous_payload.get("microbatch_size") == MICROBATCH_SIZE
        and previous_payload.get("source_commit") == source_commit
    )
    from dataset.src.remote import build_checkpoint_manifest

    remote_bytes_current = (
        isinstance(pointer, Mapping)
        and pointer.get("checkpoint_manifest")
        == build_checkpoint_manifest(checkpoint_root)
    )
    if (
        current_id == checkpoint_id
        and force_id != checkpoint_id
        and local_bucket_transport_current
        and remote_bytes_current
    ):
        return {"status": "already_current", "checkpoint_id": checkpoint_id}

    manifest = runtime_base._write_hf_transport_manifest(
        RUN_DIR / "hf_checkpoint_transport.json",
        run_id=RUN_ID,
        dataset=dataset,
        dataset_profile=DATASET_PROFILE,
        source_commit=source_commit,
        microbatch_size=MICROBATCH_SIZE,
        resume_parent_source_commit=previous_source_commit,
        bucket_id=checkpoint_bucket_id,
    )
    from dataset.src.remote import TwoPhaseCheckpointPublisher

    publisher = TwoPhaseCheckpointPublisher(store, run_id=RUN_ID)
    publisher.publish(
        checkpoint_root,
        checkpoint_id=checkpoint_id,
        drive_manifest=manifest,
        metric=None,
        best_metric=None,
    )
    cleanup = store.prune_run_checkpoints(run_id=RUN_ID, checkpoint_id=checkpoint_id)
    durable = store.read_json(f"run/{RUN_ID}/latest.json")
    if not isinstance(durable, Mapping) or durable.get("checkpoint_id") != checkpoint_id:
        raise RuntimeError("Kaggle checkpoint Bucket latest changed during migration")
    return dict(cleanup)


def _prepare(runtime_base: Any) -> tuple[str, int, Path]:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    local_id, local_step = runtime_base._latest_checkpoint(CHECKPOINT_DIR)
    if local_id is None:
        restored = _restore_pointer(runtime_base, run_id=RUN_ID, destination_run_dir=RUN_DIR)
        if restored is not None:
            local_id = str(restored["checkpoint_id"])
            local_step = int(restored["step"])

    if local_id is not None:
        step = _verify_deep_decay_checkpoint(runtime_base, local_id)
        if step != local_step:
            raise RuntimeError("verified continuation checkpoint step disagrees with runtime cursor")
        if not CONTRACT_PATH.is_file():
            _write_json(CONTRACT_PATH, _contract())
        existing = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
        if existing != _contract():
            raise RuntimeError("deep-decay Kaggle contract drifted")
        if step == FINAL_STEP:
            return local_id, step, Path()
        dataset = _stage_dataset(runtime_base, start_block_id=step)
        _publish_latest_to_bucket(
            runtime_base,
            checkpoint_id=local_id,
            dataset=dataset,
        )
        return local_id, step, dataset

    source = _restore_pointer(runtime_base, run_id=SOURCE_RUN_ID, destination_run_dir=SOURCE_CACHE_DIR, require_checkpoint_id=SOURCE_CHECKPOINT_ID)
    if source is None:
        raise RuntimeError(
            f"exact source {SOURCE_RUN_ID}/{SOURCE_CHECKPOINT_ID} is unavailable "
            "in the HF checkpoint Bucket or legacy model repository"
        )
    if int(source["step"]) != SOURCE_STEP:
        raise RuntimeError("restored source step is not exactly 15,500")
    dataset = _stage_dataset(runtime_base, start_block_id=SOURCE_STEP)
    _fork_source_checkpoint(runtime_base, source_root=SOURCE_CACHE_DIR, dataset=dataset)
    _write_json(CONTRACT_PATH, _contract())
    local_id, local_step = runtime_base._latest_checkpoint(CHECKPOINT_DIR)
    if local_id != SOURCE_CHECKPOINT_ID or local_step != SOURCE_STEP:
        raise RuntimeError("deep-decay fork did not install exact step-15,500 checkpoint")
    _publish_latest_to_bucket(
        runtime_base,
        checkpoint_id=local_id,
        dataset=dataset,
    )
    return local_id, local_step, dataset


def _build_trainer_command(runtime_base: Any, *, dataset: Path, resume_checkpoint_id: str, steps: int) -> list[str]:
    from model_repo_checkpoint import install_model_repo_checkpoint_transport  # type: ignore
    from profiles import resolve_presets  # type: ignore

    model_preset, token_preset = resolve_presets("100M", "10B")
    if token_preset.dataset_profile != DATASET_PROFILE:
        raise RuntimeError("100M/10B profile drifted")

    install_model_repo_checkpoint_transport()
    os.environ["SMALL_LLM_MODAL_ROLLING_DATASET"] = "1"
    dataset_bucket_id = os.environ.get("SMALL_LLM_HF_DATASET_BUCKET_ID", "").strip() or f"{runtime_base._hf_model_repo_id()}-datasets"
    os.environ["SMALL_LLM_DATASET_SHARD_BUCKET"] = dataset_bucket_id
    os.environ["SMALL_LLM_DATASET_SHARD_RUN_ID"] = DATASET_RUN_ID
    os.environ["SMALL_LLM_DATASET_SHARD_PREFETCH"] = "1"

    checkpoint_bucket_id = runtime_base._hf_checkpoint_bucket_id()
    remote_manifest = RUN_DIR / "hf_checkpoint_transport.json"
    runtime_base._write_hf_transport_manifest(
        remote_manifest,
        run_id=RUN_ID,
        dataset=dataset,
        dataset_profile=DATASET_PROFILE,
        source_commit=os.environ.get("SMALL_LLM_SOURCE_COMMIT", "kaggle-main"),
        microbatch_size=MICROBATCH_SIZE,
        resume_parent_source_commit=None,
        bucket_id=checkpoint_bucket_id,
    )
    plan: dict[str, Any] = {"trainer": {"warmup_tokens": 0, "stable_tokens": 0, "decay_tokens": COOLDOWN_TOKENS, "validation_blocks": 16}}
    command = runtime_base._trainer_command(
        model=model_preset,
        tokens=token_preset,
        dataset=dataset,
        plan=plan,
        checkpoint_dir=CHECKPOINT_DIR,
        steps=steps,
        microbatch=MICROBATCH_SIZE,
        precision="fp16",
        wandb_run_id=RUN_ID,
        gpu_tag="dual-t4",
        online=True,
        resume=resume_checkpoint_id,
        remote_manifest=remote_manifest,
        remote_bucket_id=checkpoint_bucket_id,
    )
    _replace_option(command, "--schedule", "wsqd")
    _replace_option(command, "--warmup-tokens", "0")
    _replace_option(command, "--stable-tokens", "0")
    _replace_option(command, "--decay-tokens", str(COOLDOWN_TOKENS))
    _replace_option(command, "--minimum-lr-ratio", str(MINIMUM_LR_RATIO))
    _replace_option(command, "--remote-publish-every-steps", str(REMOTE_EVERY))
    _replace_option(command, "--wandb-resume", "allow" if resume_checkpoint_id == SOURCE_CHECKPOINT_ID else "must")
    _replace_option(command, "--wandb-run-name", "100M/10B deep-decay continuation from step 15500")
    command += [
        "--schedule-anchor-tokens", str(SOURCE_EXPECTED_TOKENS),
        "--cooldown-start-tokens", str(COOLDOWN_START_TOKENS),
        "--settle-tokens", str(SETTLE_TOKENS),
        "--settle-lr-ratio", str(SETTLE_LR_RATIO),
        "--base-power", str(BASE_POWER),
    ]
    _replace_tag(command, "beam", "kaggle")
    if "--wandb-tags" in command and "dual-t4-ddp" not in command:
        tag_index = command.index("--wandb-tags") + 1
        while tag_index < len(command) and not command[tag_index].startswith("--"):
            tag_index += 1
        command.insert(tag_index, "dual-t4-ddp")
    return command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-steps-this-session", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _dry_run_payload(max_steps_this_session: int | None) -> dict[str, object]:
    return {
        "action": "step15500_deep_decay_10b_continuation",
        "execution": "kaggle_dual_t4_ddp_block64",
        "source_run_id": SOURCE_RUN_ID,
        "source_checkpoint_id": SOURCE_CHECKPOINT_ID,
        "source_microbatch_size": SOURCE_MICROBATCH_SIZE,
        "run_id": RUN_ID,
        "dataset_profile": DATASET_PROFILE,
        "dataset_run_id": DATASET_RUN_ID,
        "world_size": dual_t4_runtime.WORLD_SIZE,
        "sequences_per_block": SEQUENCES_PER_BLOCK,
        "sequences_per_rank": SEQUENCES_PER_BLOCK // dual_t4_runtime.WORLD_SIZE,
        "microbatch_size": MICROBATCH_SIZE,
        "local_microbatches_per_rank": SEQUENCES_PER_BLOCK // dual_t4_runtime.WORLD_SIZE // MICROBATCH_SIZE,
        "runtime": {"torch": dual_t4_runtime.TORCH_VERSION, "triton": dual_t4_runtime.TRITON_VERSION, "fla_core": dual_t4_runtime.FLA_VERSION},
        "max_steps_this_session": max_steps_this_session,
        "remote_checkpoint_every": REMOTE_EVERY,
        "schedule": _contract(),
        "resume": "automatic_verified_hf_or_exact_step15500_fork",
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_steps_this_session is not None and args.max_steps_this_session <= 0:
        raise SystemExit("--max-steps-this-session must be positive")
    if args.dry_run:
        print(json.dumps(_dry_run_payload(args.max_steps_this_session), indent=2, sort_keys=True))
        return 0

    runtime_base = _beam_runtime()
    checkpoint_id, completed_step, dataset = _prepare(runtime_base)
    if completed_step == FINAL_STEP:
        print(json.dumps({"status": "already_complete", "run_id": RUN_ID, "checkpoint_id": checkpoint_id, "completed_steps": completed_step, "final_step": FINAL_STEP}, indent=2, sort_keys=True))
        return 0

    remaining = FINAL_STEP - completed_step
    steps = min(remaining, args.max_steps_this_session or remaining)
    trainer_command = _build_trainer_command(runtime_base, dataset=dataset, resume_checkpoint_id=checkpoint_id, steps=steps)
    command = _dual_t4_command(trainer_command)
    log_path = RUN_DIR / "evidence" / f"kaggle-dual-t4-from-{checkpoint_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    runtime_base._run(command, cwd=ROOT, log_path=log_path)

    final_id, final_step = runtime_base._latest_checkpoint(CHECKPOINT_DIR)
    expected = completed_step + steps
    if final_id is None or final_step != expected:
        raise RuntimeError(f"durable checkpoint step {final_step} != expected {expected}")
    result = {
        "status": "complete" if final_step == FINAL_STEP else "segment_complete",
        "run_id": RUN_ID,
        "checkpoint_id": final_id,
        "completed_steps": final_step,
        "final_step": FINAL_STEP,
        "elapsed_seconds": time.perf_counter() - started,
        "execution": "kaggle_dual_t4_ddp_block64",
        "microbatch_size": MICROBATCH_SIZE,
        "lr_now": _expected_lr(final_step * TARGETS_PER_FULL_BLOCK),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
