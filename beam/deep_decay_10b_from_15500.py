#!/usr/bin/env python3
"""Resume the portable 100M/10B deep-decay continuation on Beam.

Usage from repository root:

    python beam/deep_decay_10b_from_15500.py --dry-run
    python beam/deep_decay_10b_from_15500.py --gpu RTX4090

The shared continuation namespace is provider-neutral. Beam first resolves the
newest verified local/Hugging-Face checkpoint, then canonicalizes only execution
slicing to single-GPU microbatch 4 when the checkpoint was last written by
Kaggle (microbatch 2) or Modal (microbatch 16). The ADR-0095 scientific
schedule, model, optimizer, scaler, data cursor, consumed tokens, CPU RNG state,
validation prefix, and corpus order remain unchanged.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beam import function  # noqa: E402
from beam import launch as base  # noqa: E402
from trainer.deep_decay_provider_migration import (  # noqa: E402
    execution_rewrite_needed,
    rewrite_execution_state,
    validate_target_execution_state,
)

SOURCE_RUN_ID = "100m-10b-data-001"
SOURCE_STEP = 15_500
SOURCE_CHECKPOINT_ID = f"step-{SOURCE_STEP:08d}"
RUN_ID = "100m-10b-deep-decay-from-step15500"
DATASET_PROFILE = "modal-10b-b64"
DATASET_RUN_ID = "modal-10b-b64-dataset-001"
MICROBATCH_SIZE = 4
TARGETS_PER_FULL_BLOCK = 64 * 2048
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

RUN_DIR = base.RUN_ROOT / RUN_ID
CHECKPOINT_DIR = RUN_DIR / "checkpoints"
CONTRACT_PATH = RUN_DIR / "deep_decay_10b_contract.json"

if FINAL_STEP * TARGETS_PER_FULL_BLOCK != TOTAL_TARGETS:
    raise RuntimeError("frozen 10B endpoint is not block-aligned as expected")
if SETTLE_END_TOKENS >= COOLDOWN_START_TOKENS:
    raise RuntimeError("settling phase overlaps terminal cooldown")
if COOLDOWN_START_TOKENS + COOLDOWN_TOKENS != TOTAL_TARGETS:
    raise RuntimeError("terminal cooldown does not end at exact 10B")


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

    progress = min(
        1.0,
        max(0.0, (tokens - COOLDOWN_START_TOKENS) / COOLDOWN_TOKENS),
    )
    return FINAL_LR + (COOLDOWN_START_LR - FINAL_LR) * (1.0 - progress)


if not math.isclose(
    _expected_lr(COOLDOWN_START_TOKENS),
    COOLDOWN_START_LR,
    rel_tol=1e-12,
    abs_tol=0.0,
):
    raise RuntimeError("calibrated power phase does not hit 1e-5 at cooldown start")
if not math.isclose(
    _expected_lr(TOTAL_TARGETS), FINAL_LR, rel_tol=1e-12, abs_tol=0.0
):
    raise RuntimeError("terminal cooldown does not hit 5e-6 at exact 10B")


def _contract() -> dict[str, object]:
    return {
        "version": 1,
        "kind": "step15500_deep_decay_settle_power_linear_cooldown",
        "source_run_id": SOURCE_RUN_ID,
        "source_checkpoint_id": SOURCE_CHECKPOINT_ID,
        "source_step": SOURCE_STEP,
        "source_expected_consumed_tokens": SOURCE_EXPECTED_TOKENS,
        "run_id": RUN_ID,
        "dataset_profile": DATASET_PROFILE,
        "dataset_run_id": DATASET_RUN_ID,
        "microbatch_size": MICROBATCH_SIZE,
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
        "scientific_change": "scheduler_only; preserve exact uncooled step15500 trainer/data state",
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


def _scientific_checkpoint_drift(config: Mapping[str, object]) -> dict[str, tuple[object, object]]:
    expected = {
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
    return {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }


def _dataset_configuration_hash(dataset: Path) -> str:
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    production = manifest.get("production") if isinstance(manifest, Mapping) else None
    value = production.get("configuration_hash") if isinstance(production, Mapping) else None
    if not isinstance(value, str) or not value:
        raise RuntimeError("staged 10B dataset lacks production.configuration_hash")
    return value


def _restore_pointer(
    runtime_base: Any,
    *,
    store: object,
    pointer: Mapping[str, object],
) -> dict[str, object]:
    from dataset.src.joint_checkpoint import restore_on_empty_vps
    from dataset.src.remote import TwoPhaseCheckpointPublisher

    checkpoint_id = pointer.get("checkpoint_id")
    if not isinstance(checkpoint_id, str):
        raise RuntimeError("HF deep-decay latest pointer has no checkpoint_id")
    restored = restore_on_empty_vps(
        publisher=TwoPhaseCheckpointPublisher(store, run_id=RUN_ID),
        store=None,
        run_id=RUN_ID,
        destination=RUN_DIR,
        checkpoint_pointer=pointer,
        prefetch_shards=0,
    )
    metadata = runtime_base._verified_checkpoint_metadata(restored, checkpoint_id)
    metadata["source"] = "hf_model_repo"
    return metadata


def _refresh_from_hf_if_newer(runtime_base: Any) -> tuple[str | None, int, str]:
    """Resolve newest verified continuation across Beam Volume and shared HF."""

    local_id, local_step = runtime_base._latest_checkpoint(CHECKPOINT_DIR)
    store = runtime_base._hf_model_repo_store()
    pointer = store.read_json(f"run/{RUN_ID}/latest.json")
    if pointer is None:
        return local_id, local_step, "beam_volume"
    if not isinstance(pointer, Mapping):
        raise RuntimeError("HF deep-decay latest pointer is not an object")
    remote_id = pointer.get("checkpoint_id")
    if not isinstance(remote_id, str):
        raise RuntimeError("HF deep-decay latest pointer has no checkpoint_id")
    remote_step = _checkpoint_step(remote_id)
    if not SOURCE_STEP <= remote_step <= FINAL_STEP:
        raise RuntimeError(f"HF deep-decay checkpoint step {remote_step} is outside the frozen horizon")
    if local_step >= remote_step:
        return local_id, local_step, "beam_volume"

    restored = _restore_pointer(runtime_base, store=store, pointer=pointer)
    restored_id = str(restored["checkpoint_id"])
    restored_step = int(restored["step"])
    if restored_id != remote_id or restored_step != remote_step:
        raise RuntimeError(
            "HF continuation changed during restore: "
            f"expected {remote_id}/{remote_step}, got {restored_id}/{restored_step}"
        )
    return restored_id, restored_step, "hf_model_repo"


def _fork_source_checkpoint(*, dataset: Path) -> None:
    import torch
    from dataset.src.joint_checkpoint import verify_local_manifest
    from dataset.src.remote import sha256_path
    from trainer.identity import canonical_hash
    from trainer.state import load_trainer_state_file, release_host_memory

    source = base.RUN_ROOT / SOURCE_RUN_ID / "checkpoints" / SOURCE_CHECKPOINT_ID
    if not source.is_dir():
        raise RuntimeError(
            f"exact source checkpoint is absent from Beam run Volume: {source}; "
            "this launcher never substitutes latest/nearest state"
        )
    verify_local_manifest(source)
    source_payload = json.loads((source / "checkpoint.json").read_text(encoding="utf-8"))
    if not isinstance(source_payload, Mapping):
        raise RuntimeError("source checkpoint.json is not an object")
    pipeline = source_payload.get("pipeline_state")
    if not isinstance(pipeline, Mapping) or pipeline.get("last_consumed_block_id") != SOURCE_STEP - 1:
        raise RuntimeError("step-15500 checkpoint does not carry the expected data cursor")

    state = load_trainer_state_file(source / "trainer_state.pkl", map_location="cpu")
    try:
        if state.get("global_step") != SOURCE_STEP:
            raise RuntimeError("step-15500 trainer state has the wrong global_step")
        if state.get("consumed_tokens") != SOURCE_EXPECTED_TOKENS:
            raise RuntimeError(
                "step-15500 trainer state has unexpected consumed-token count: "
                f"{state.get('consumed_tokens')!r} != {SOURCE_EXPECTED_TOKENS}"
            )
        config = state.get("config")
        scheduler = state.get("scheduler")
        model_config = state.get("model_config")
        if not isinstance(config, Mapping) or not isinstance(scheduler, Mapping):
            raise RuntimeError("source trainer state lacks config/scheduler mappings")
        if not isinstance(model_config, Mapping):
            raise RuntimeError("source trainer state lacks model_config")
        if config.get("microbatch_size") != MICROBATCH_SIZE:
            raise RuntimeError(
                f"source checkpoint froze microbatch {config.get('microbatch_size')!r}; "
                f"continuation requires {MICROBATCH_SIZE}"
            )
        if config.get("learning_rate") != PEAK_LR or config.get("schedule") != "wsd":
            raise RuntimeError("source checkpoint does not match expected 3e-4 WSD parent")

        patched_config = dict(config)
        patched_config.update(
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
        validate_target_execution_state(state, target_microbatch=MICROBATCH_SIZE)

        new_configuration_hash = canonical_hash(
            {
                "version": 1,
                "model": dict(model_config),
                "trainer": patched_config,
                "dataset_configuration_hash": _dataset_configuration_hash(dataset),
            }
        )

        staging = CHECKPOINT_DIR / f".{SOURCE_CHECKPOINT_ID}.fork"
        target = CHECKPOINT_DIR / SOURCE_CHECKPOINT_ID
        if target.exists():
            verify_local_manifest(target)
            return
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True, exist_ok=False)
        trainer_state_path = staging / "trainer_state.pkl"
        torch.save(state, trainer_state_path)

        checkpoint_payload = dict(source_payload)
        checkpoint_payload["configuration_hash"] = new_configuration_hash
        _write_json(staging / "checkpoint.json", checkpoint_payload)
        _write_json(
            staging / "local_manifest.json",
            {
                "version": 1,
                "files": [
                    {"name": "trainer_state.pkl", "sha256": sha256_path(trainer_state_path)},
                    {"name": "checkpoint.json", "sha256": sha256_path(staging / "checkpoint.json")},
                ],
            },
        )
        verify_local_manifest(staging)
        os.replace(staging, target)
    finally:
        del state
        release_host_memory()


def _install_execution_migration(*, checkpoint_id: str, dataset: Path) -> dict[str, object]:
    """Canonicalize an authorized continuation to Beam microbatch-4 / one GPU."""

    import torch
    from dataset.src.joint_checkpoint import verify_local_manifest
    from dataset.src.remote import sha256_path
    from trainer.identity import canonical_hash
    from trainer.state import load_trainer_state_file, release_host_memory

    step = _checkpoint_step(checkpoint_id)
    root = CHECKPOINT_DIR / checkpoint_id
    verify_local_manifest(root)
    payload = json.loads((root / "checkpoint.json").read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise RuntimeError("deep-decay checkpoint.json is not an object")
    pipeline = payload.get("pipeline_state")
    if not isinstance(pipeline, Mapping) or pipeline.get("last_consumed_block_id") != step - 1:
        raise RuntimeError("deep-decay checkpoint data cursor drifted")

    state = load_trainer_state_file(root / "trainer_state.pkl", map_location="cpu")
    try:
        if state.get("global_step") != step:
            raise RuntimeError("deep-decay checkpoint global_step disagrees with checkpoint ID")
        if state.get("consumed_tokens") != step * TARGETS_PER_FULL_BLOCK:
            raise RuntimeError("deep-decay checkpoint consumed-token count drifted")
        config = state.get("config")
        scheduler = state.get("scheduler")
        model_config = state.get("model_config")
        if not isinstance(config, Mapping) or not isinstance(scheduler, Mapping):
            raise RuntimeError("deep-decay checkpoint lacks config/scheduler mappings")
        if not isinstance(model_config, Mapping):
            raise RuntimeError("deep-decay checkpoint lacks model_config")
        drift = _scientific_checkpoint_drift(config)
        if drift:
            raise RuntimeError(
                "deep-decay checkpoint scientific config drifted: "
                + json.dumps(drift, sort_keys=True)
            )
        if scheduler.get("committed_tokens") != step * TARGETS_PER_FULL_BLOCK:
            raise RuntimeError("deep-decay scheduler committed_tokens drifted")
        scheduler_config = scheduler.get("config")
        if not isinstance(scheduler_config, Mapping):
            raise RuntimeError("deep-decay scheduler lacks config mapping")
        scheduler_drift = _scientific_checkpoint_drift(scheduler_config)
        if scheduler_drift:
            raise RuntimeError(
                "deep-decay scheduler scientific config drifted: "
                + json.dumps(scheduler_drift, sort_keys=True)
            )
        expected_lr = _expected_lr(step * TARGETS_PER_FULL_BLOCK)
        last_lr = scheduler.get("last_lr")
        if (
            isinstance(last_lr, bool)
            or not isinstance(last_lr, (int, float))
            or not math.isfinite(float(last_lr))
            or not math.isclose(float(last_lr), expected_lr, rel_tol=1e-10, abs_tol=1e-12)
        ):
            raise RuntimeError(f"deep-decay scheduler LR drifted: {last_lr!r} != {expected_lr!r}")

        if not execution_rewrite_needed(state, target_microbatch=MICROBATCH_SIZE):
            validate_target_execution_state(state, target_microbatch=MICROBATCH_SIZE)
            return {"status": "already_beam_sliced", "from_microbatch": MICROBATCH_SIZE}

        patched_state, migration = rewrite_execution_state(
            state,
            target_microbatch=MICROBATCH_SIZE,
        )
        patched_config = patched_state.get("config")
        if not isinstance(patched_config, Mapping):
            raise RuntimeError("execution migration lost trainer config")
        configuration_hash = canonical_hash(
            {
                "version": 1,
                "model": dict(model_config),
                "trainer": dict(patched_config),
                "dataset_configuration_hash": _dataset_configuration_hash(dataset),
            }
        )

        staging = CHECKPOINT_DIR / f".{checkpoint_id}.beam-provider-migration"
        backup = CHECKPOINT_DIR / f".{checkpoint_id}.pre-beam-provider"
        shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(backup, ignore_errors=True)
        shutil.copytree(root, staging)
        (staging / "checkpoint_manifest.json").unlink(missing_ok=True)
        trainer_state_path = staging / "trainer_state.pkl"
        torch.save(patched_state, trainer_state_path)
        patched_payload = dict(payload)
        patched_payload["configuration_hash"] = configuration_hash
        _write_json(staging / "checkpoint.json", patched_payload)
        _write_json(
            staging / "local_manifest.json",
            {
                "version": 1,
                "files": [
                    {"name": "trainer_state.pkl", "sha256": sha256_path(trainer_state_path)},
                    {"name": "checkpoint.json", "sha256": sha256_path(staging / "checkpoint.json")},
                ],
            },
        )
        verify_local_manifest(staging)
        os.replace(root, backup)
        try:
            os.replace(staging, root)
        except BaseException:
            os.replace(backup, root)
            raise
        else:
            shutil.rmtree(backup, ignore_errors=True)

        strict_state = load_trainer_state_file(root / "trainer_state.pkl", map_location="cpu")
        try:
            validate_target_execution_state(strict_state, target_microbatch=MICROBATCH_SIZE)
        finally:
            del strict_state
            release_host_memory()
        return {"status": "migrated_execution_slicing", **migration}
    finally:
        del state
        release_host_memory()


@function(
    name="small-llm-deep-decay-10b-15500-prepare",
    image=base.CPU_IMAGE,
    cpu=4,
    memory="16Gi",
    timeout=-1,
    retries=1,
    secrets=base.SECRETS,
    volumes=[base.RUN_VOLUME, base.CACHE_VOLUME],
    env=base.RUNTIME_ENV,
)
def prepare_remote(source_commit: str) -> dict[str, object]:
    del source_commit
    base._install_beam_imports()
    import runtime as runtime_base
    from dataset.incremental_stage import stage_incremental_window_when_ready
    from dataset.qualification import get_profile
    from dataset.src.hf_bucket_shards import HuggingFaceBucketShardStore
    from dataset.src.joint_checkpoint import verify_local_manifest
    from model_repo_checkpoint import install_model_repo_checkpoint_transport
    from rolling_dataset import hf_dataset_bucket_id

    install_model_repo_checkpoint_transport()
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    latest_id, latest_step, checkpoint_source = _refresh_from_hf_if_newer(runtime_base)
    if latest_step > FINAL_STEP:
        raise RuntimeError(f"continuation checkpoint step {latest_step} exceeds {FINAL_STEP}")
    if latest_step == 0:
        source = base.RUN_ROOT / SOURCE_RUN_ID / "checkpoints" / SOURCE_CHECKPOINT_ID
        if not source.is_dir():
            raise RuntimeError(
                f"exact source checkpoint {SOURCE_CHECKPOINT_ID} is absent from {source.parent}"
            )
        verify_local_manifest(source)
    if latest_step == FINAL_STEP:
        return {
            "status": "training_complete",
            "run_id": RUN_ID,
            "checkpoint_id": latest_id,
            "completed_steps": latest_step,
            "final_step": FINAL_STEP,
            "checkpoint_source": checkpoint_source,
        }

    required_block = SOURCE_STEP if latest_step == 0 else latest_step
    profile = get_profile(DATASET_PROFILE)
    if profile.run_id != DATASET_RUN_ID:
        raise RuntimeError("10B dataset profile/run ID drifted")
    bucket_id = hf_dataset_bucket_id()
    store = HuggingFaceBucketShardStore(
        bucket_id,
        token=runtime_base._hf_token(),
        private=True,
        create_bucket=False,
    )
    dataset = base.CACHE_ROOT / "deep-decay-10b" / RUN_ID / f"from-{required_block:08d}"
    staged = stage_incremental_window_when_ready(
        store=store,
        run_id=DATASET_RUN_ID,
        destination=dataset,
        start_block_id=required_block,
    )
    if staged.get("status") != "ready":
        raise RuntimeError(f"dataset stage did not become ready: {staged}")

    migration: dict[str, object]
    if latest_step == 0:
        _fork_source_checkpoint(dataset=dataset)
        latest_id, latest_step = runtime_base._latest_checkpoint(CHECKPOINT_DIR)
        if latest_id != SOURCE_CHECKPOINT_ID or latest_step != SOURCE_STEP:
            raise RuntimeError("fork did not install exact step-15500 checkpoint")
        checkpoint_source = "exact_source_fork"
        migration = {"status": "source_fork"}
        _write_json(CONTRACT_PATH, _contract())
    else:
        assert latest_id is not None
        migration = _install_execution_migration(
            checkpoint_id=latest_id,
            dataset=dataset,
        )
        if not CONTRACT_PATH.is_file():
            _write_json(CONTRACT_PATH, _contract())
        elif json.loads(CONTRACT_PATH.read_text(encoding="utf-8")) != _contract():
            raise RuntimeError("deep-decay contract drifted between segments")

    return {
        "status": "ready",
        "run_id": RUN_ID,
        "resume_checkpoint_id": latest_id,
        "completed_steps": latest_step,
        "remaining_steps": FINAL_STEP - latest_step,
        "final_step": FINAL_STEP,
        "required_block": latest_step,
        "dataset_dir": str(dataset),
        "dataset_bucket_id": bucket_id,
        "checkpoint_source": checkpoint_source,
        "execution_migration": migration,
        "lr_now": _expected_lr(latest_step * TARGETS_PER_FULL_BLOCK),
        "lr_at_settle_end": SETTLE_LR,
        "base_power": BASE_POWER,
        "lr_at_cooldown_start": COOLDOWN_START_LR,
        "lr_final": FINAL_LR,
    }


@function(
    name="small-llm-deep-decay-10b-15500-visibility",
    image=base.CPU_IMAGE,
    cpu=2,
    memory="8Gi",
    timeout=180,
    retries=1,
    secrets=base.SECRETS,
    volumes=[base.CACHE_VOLUME],
    env=base.RUNTIME_ENV,
)
def verify_stage_remote(dataset_dir: str, required_block: int) -> dict[str, object]:
    base._install_beam_imports()
    from dataset.incremental_stage import verify_incremental_stage
    from rolling_dataset import hf_dataset_bucket_id

    verification = verify_incremental_stage(
        destination=Path(dataset_dir),
        bucket_id=hf_dataset_bucket_id(),
        run_id=DATASET_RUN_ID,
        required_train_block=required_block,
    )
    return {"status": "visible", "verification": verification}


def _replace_option(command: list[str], option: str, value: str) -> None:
    try:
        index = command.index(option)
    except ValueError as error:
        raise RuntimeError(f"trainer command lacks required option {option}") from error
    if index + 1 >= len(command):
        raise RuntimeError(f"trainer command option {option} has no value")
    command[index + 1] = value


def _train_impl(
    source_commit: str,
    dataset_dir: str,
    resume_checkpoint_id: str,
    remaining_steps: int,
) -> dict[str, object]:
    repo = base._install_beam_imports()
    import runtime as runtime_base
    from model_repo_checkpoint import install_model_repo_checkpoint_transport
    from profiles import resolve_presets

    if remaining_steps <= 0:
        raise RuntimeError("GPU was allocated with no remaining work")
    dataset = Path(dataset_dir).resolve(strict=True)
    model_preset, token_preset = resolve_presets("100M", "10B")
    if token_preset.dataset_profile != DATASET_PROFILE:
        raise RuntimeError("100M/10B profile drifted")

    install_model_repo_checkpoint_transport()
    os.environ["SMALL_LLM_MODAL_ROLLING_DATASET"] = "1"
    os.environ["SMALL_LLM_DATASET_SHARD_BUCKET"] = (
        os.environ.get("SMALL_LLM_HF_DATASET_BUCKET_ID", "").strip()
        or f"{runtime_base._hf_model_repo_id()}-datasets"
    )
    os.environ["SMALL_LLM_DATASET_SHARD_RUN_ID"] = DATASET_RUN_ID
    os.environ["SMALL_LLM_DATASET_SHARD_PREFETCH"] = "1"

    checkpoint_repo_id = runtime_base._hf_checkpoint_bucket_id()
    remote_manifest = RUN_DIR / "hf_checkpoint_transport.json"
    runtime_base._write_hf_transport_manifest(
        remote_manifest,
        run_id=RUN_ID,
        dataset=dataset,
        dataset_profile=DATASET_PROFILE,
        source_commit=source_commit,
        microbatch_size=MICROBATCH_SIZE,
        resume_parent_source_commit=None,
        bucket_id=checkpoint_repo_id,
    )
    plan: dict[str, Any] = {
        "trainer": {
            "warmup_tokens": 0,
            "stable_tokens": 0,
            "decay_tokens": COOLDOWN_TOKENS,
            "validation_blocks": 16,
        }
    }
    environment = runtime_base._gpu_environment()
    gpu_tag = re.sub(r"[^a-z0-9]+", "-", str(environment["name"]).lower()).strip("-")
    command = runtime_base._trainer_command(
        model=model_preset,
        tokens=token_preset,
        dataset=dataset,
        plan=plan,
        checkpoint_dir=CHECKPOINT_DIR,
        steps=remaining_steps,
        microbatch=MICROBATCH_SIZE,
        precision="fp16",
        wandb_run_id=RUN_ID,
        gpu_tag=gpu_tag,
        online=True,
        resume=resume_checkpoint_id,
        remote_manifest=remote_manifest,
        remote_bucket_id=checkpoint_repo_id,
    )
    _replace_option(command, "--schedule", "wsqd")
    _replace_option(command, "--warmup-tokens", "0")
    _replace_option(command, "--stable-tokens", "0")
    _replace_option(command, "--decay-tokens", str(COOLDOWN_TOKENS))
    _replace_option(command, "--minimum-lr-ratio", str(MINIMUM_LR_RATIO))
    command += [
        "--schedule-anchor-tokens",
        str(SOURCE_EXPECTED_TOKENS),
        "--cooldown-start-tokens",
        str(COOLDOWN_START_TOKENS),
        "--settle-tokens",
        str(SETTLE_TOKENS),
        "--settle-lr-ratio",
        str(SETTLE_LR_RATIO),
        "--base-power",
        str(BASE_POWER),
    ]
    _replace_option(
        command,
        "--wandb-resume",
        "allow" if resume_checkpoint_id == SOURCE_CHECKPOINT_ID else "must",
    )
    _replace_option(
        command,
        "--wandb-run-name",
        "100M/10B deep-decay continuation from step 15500",
    )

    log_path = RUN_DIR / "evidence" / f"train-from-{resume_checkpoint_id}.log"
    started = time.perf_counter()
    runtime_base._run(command, cwd=repo, log_path=log_path)
    base.NOOP_VOLUME.commit()

    final_id, final_step = runtime_base._latest_checkpoint(CHECKPOINT_DIR)
    expected = int(resume_checkpoint_id.removeprefix("step-")) + remaining_steps
    if final_step != expected:
        raise RuntimeError(f"durable checkpoint step {final_step} != expected {expected}")
    return {
        "status": "complete" if final_step == FINAL_STEP else "segment_complete",
        "run_id": RUN_ID,
        "checkpoint_id": final_id,
        "completed_steps": final_step,
        "final_step": FINAL_STEP,
        "elapsed_seconds": time.perf_counter() - started,
        "source_checkpoint_id": SOURCE_CHECKPOINT_ID,
        "schedule": _contract(),
    }


@function(
    name="small-llm-deep-decay-10b-15500-rtx5090",
    gpu="RTX5090",
    image=base.BLACKWELL_IMAGE,
    **base._GPU_FUNCTION_KWARGS,
)
def train_rtx5090_remote(
    source_commit: str,
    dataset_dir: str,
    resume_checkpoint_id: str,
    remaining_steps: int,
) -> dict[str, object]:
    return _train_impl(source_commit, dataset_dir, resume_checkpoint_id, remaining_steps)


@function(
    name="small-llm-deep-decay-10b-15500-rtx4090",
    gpu="RTX4090",
    image=base.LEGACY_SERVERLESS_IMAGE,
    **base._GPU_FUNCTION_KWARGS,
)
def train_rtx4090_remote(
    source_commit: str,
    dataset_dir: str,
    resume_checkpoint_id: str,
    remaining_steps: int,
) -> dict[str, object]:
    return _train_impl(source_commit, dataset_dir, resume_checkpoint_id, remaining_steps)


@function(
    name="small-llm-deep-decay-10b-15500-a10g",
    gpu="A10G",
    image=base.LEGACY_SERVERLESS_IMAGE,
    **base._GPU_FUNCTION_KWARGS,
)
def train_a10g_remote(
    source_commit: str,
    dataset_dir: str,
    resume_checkpoint_id: str,
    remaining_steps: int,
) -> dict[str, object]:
    return _train_impl(source_commit, dataset_dir, resume_checkpoint_id, remaining_steps)


GPU_FUNCTIONS = {
    "RTX5090": train_rtx5090_remote,
    "RTX4090": train_rtx4090_remote,
    "A10G": train_a10g_remote,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="RTX4090", choices=sorted(GPU_FUNCTIONS))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    source_commit = base._local_source_commit()
    payload = {
        "action": "step15500_deep_decay_10b_continuation",
        "runtime": "beam/deep_decay_10b_from_15500.py",
        "gpu": args.gpu,
        "source_run_id": SOURCE_RUN_ID,
        "source_checkpoint_id": SOURCE_CHECKPOINT_ID,
        "run_id": RUN_ID,
        "source_step": SOURCE_STEP,
        "source_consumed_tokens": SOURCE_EXPECTED_TOKENS,
        "lr_at_source_step": PEAK_LR,
        "settle_steps": SETTLE_STEPS,
        "settle_tokens": SETTLE_TOKENS,
        "settle_end_step": SETTLE_END_STEP,
        "settle_end_tokens": SETTLE_END_TOKENS,
        "lr_at_settle_end": SETTLE_LR,
        "base_power": BASE_POWER,
        "cooldown_start_step": COOLDOWN_START_STEP,
        "cooldown_start_tokens": COOLDOWN_START_TOKENS,
        "lr_at_cooldown_start": COOLDOWN_START_LR,
        "cooldown_steps": COOLDOWN_STEPS,
        "cooldown_tokens": COOLDOWN_TOKENS,
        "lr_final": FINAL_LR,
        "additional_steps": ADDITIONAL_STEPS,
        "final_step": FINAL_STEP,
        "final_targets": TOTAL_TARGETS,
        "schedule": "300M cosine to 1e-4 + calibrated power to 1e-5 + 400M linear to 5e-6",
        "resume": "newest_verified_local_or_hf_then_execution_only_provider_migration",
        "source_commit": source_commit,
    }
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    if args.dry_run:
        return 0

    base._stage("deep_decay_10b_prepare_start", checkpoint=SOURCE_CHECKPOINT_ID)
    prepared = base._require_remote_mapping(
        prepare_remote.remote(source_commit),
        label="deep-decay 10B prepare",
    )
    base._stage("deep_decay_10b_prepare_complete", **prepared)
    if prepared.get("status") == "training_complete":
        print(json.dumps(prepared, indent=2, sort_keys=True), flush=True)
        return 0
    if prepared.get("status") != "ready":
        raise RuntimeError("CPU preparation did not authorize GPU dispatch")

    dataset_dir = prepared.get("dataset_dir")
    required_block = prepared.get("required_block")
    resume_checkpoint_id = prepared.get("resume_checkpoint_id")
    remaining_steps = prepared.get("remaining_steps")
    if not isinstance(dataset_dir, str) or not dataset_dir:
        raise RuntimeError("preparation returned no dataset directory")
    if isinstance(required_block, bool) or not isinstance(required_block, int):
        raise RuntimeError("preparation returned no required block")
    if not isinstance(resume_checkpoint_id, str):
        raise RuntimeError("preparation returned no resume checkpoint")
    if isinstance(remaining_steps, bool) or not isinstance(remaining_steps, int) or remaining_steps <= 0:
        raise RuntimeError("preparation returned invalid remaining steps")

    base._stage("deep_decay_10b_visibility_start", required_block=required_block)
    visible = base._require_remote_mapping(
        verify_stage_remote.remote(dataset_dir, required_block),
        label="deep-decay dataset visibility",
    )
    base._stage("deep_decay_10b_visibility_complete", **visible)

    base._stage(
        "deep_decay_10b_dispatch",
        gpu=args.gpu,
        resume_checkpoint_id=resume_checkpoint_id,
        remaining_steps=remaining_steps,
    )
    result = base._require_remote_mapping(
        GPU_FUNCTIONS[args.gpu].remote(
            source_commit,
            dataset_dir,
            resume_checkpoint_id,
            remaining_steps,
        ),
        label="deep-decay 10B training",
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
