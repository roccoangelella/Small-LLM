"""One-H100 Modal adapter for the frozen 100M/10B deep-decay continuation.

The provider migration changes only execution microbatch slicing. Existing
deep-decay checkpoints keep their complete scientific state and are rewritten
from their prior single-GPU/Kaggle microbatch to the H100-qualified microbatch
16. If the continuation namespace is empty, the exact original step-15,500
checkpoint is forked and receives the already-authorized ADR-0095 scheduler.
"""
from __future__ import annotations

from dataclasses import asdict
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any, Mapping

SOURCE_RUN_ID = "100m-10b-data-001"
SOURCE_STEP = 15_500
SOURCE_CHECKPOINT_ID = f"step-{SOURCE_STEP:08d}"
RUN_ID = "100m-10b-deep-decay-from-step15500"
DATASET_PROFILE = "modal-10b-b64"
DATASET_RUN_ID = "modal-10b-b64-dataset-001"

SOURCE_MICROBATCH_SIZE = 4
PRIOR_CONTINUATION_MICROBATCH_SIZES = frozenset({2, 4})
MICROBATCH_SIZE = 16
SEQUENCES_PER_BLOCK = 64
CONTEXT_LENGTH = 2_048
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
BASE_POWER = math.log(SETTLE_LR / COOLDOWN_START_LR) / math.log(
    COOLDOWN_START_TOKENS / SETTLE_END_TOKENS
)
REMOTE_EVERY = 250

if FINAL_STEP * TARGETS_PER_FULL_BLOCK != TOTAL_TARGETS:
    raise RuntimeError("frozen 10B endpoint is not block-aligned")
if SETTLE_END_TOKENS >= COOLDOWN_START_TOKENS:
    raise RuntimeError("settling phase overlaps terminal cooldown")
if COOLDOWN_START_TOKENS + COOLDOWN_TOKENS != TOTAL_TARGETS:
    raise RuntimeError("terminal cooldown does not end at exact 10B")


def expected_lr(tokens: int) -> float:
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


def contract() -> dict[str, object]:
    return {
        "version": 1,
        "kind": "step15500_deep_decay_settle_power_linear_cooldown",
        "execution": "modal_single_h100_block64",
        "source_run_id": SOURCE_RUN_ID,
        "source_checkpoint_id": SOURCE_CHECKPOINT_ID,
        "source_step": SOURCE_STEP,
        "source_expected_consumed_tokens": SOURCE_EXPECTED_TOKENS,
        "run_id": RUN_ID,
        "dataset_profile": DATASET_PROFILE,
        "dataset_run_id": DATASET_RUN_ID,
        "world_size": 1,
        "sequences_per_block": SEQUENCES_PER_BLOCK,
        "microbatch_size": MICROBATCH_SIZE,
        "microbatches_per_update": SEQUENCES_PER_BLOCK // MICROBATCH_SIZE,
        "schedule": "wsqd",
        "schedule_anchor_tokens": SOURCE_EXPECTED_TOKENS,
        "anchor_lr": PEAK_LR,
        "settle_steps": SETTLE_STEPS,
        "settle_tokens": SETTLE_TOKENS,
        "settle_end_step": SETTLE_END_STEP,
        "settle_end_tokens": SETTLE_END_TOKENS,
        "settle_lr_ratio": SETTLE_LR_RATIO,
        "base_power": BASE_POWER,
        "cooldown_start_step": COOLDOWN_START_STEP,
        "cooldown_start_tokens": COOLDOWN_START_TOKENS,
        "cooldown_steps": COOLDOWN_STEPS,
        "decay_tokens": COOLDOWN_TOKENS,
        "minimum_lr_ratio": MINIMUM_LR_RATIO,
        "final_step": FINAL_STEP,
        "final_targets": TOTAL_TARGETS,
        "remote_checkpoint_every": REMOTE_EVERY,
        "scientific_change": "none; Modal changes execution slicing only for an existing deep-decay checkpoint",
    }


def dry_run_payload(max_steps_this_session: int) -> dict[str, object]:
    return {
        "action": "deep-decay",
        "runtime": "modal/launch.py",
        "execution": "modal_single_h100_block64",
        "gpu": "H100!",
        "source_run_id": SOURCE_RUN_ID,
        "source_checkpoint_id": SOURCE_CHECKPOINT_ID,
        "continuation_run_id": RUN_ID,
        "dataset_profile": DATASET_PROFILE,
        "dataset_run_id": DATASET_RUN_ID,
        "world_size": 1,
        "sequences_per_block": SEQUENCES_PER_BLOCK,
        "microbatch_size": MICROBATCH_SIZE,
        "microbatches_per_update": SEQUENCES_PER_BLOCK // MICROBATCH_SIZE,
        "max_steps_this_session": max_steps_this_session or "remaining plan",
        "resume": "newest_verified_continuation_hf_then_exact_step_00015500_only",
        "schedule": contract(),
    }


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def checkpoint_step(checkpoint_id: str) -> int:
    prefix = "step-"
    if not checkpoint_id.startswith(prefix) or len(checkpoint_id) != 13:
        raise RuntimeError(f"invalid checkpoint ID: {checkpoint_id!r}")
    suffix = checkpoint_id[len(prefix):]
    if not suffix.isdigit():
        raise RuntimeError(f"invalid checkpoint ID: {checkpoint_id!r}")
    return int(suffix)


def _pointer_checkpoint_id(pointer: object, *, label: str) -> str | None:
    if pointer is None:
        return None
    if not isinstance(pointer, Mapping):
        raise RuntimeError(f"{label} latest pointer is not a JSON object")
    checkpoint_id = pointer.get("checkpoint_id")
    if not isinstance(checkpoint_id, str):
        raise RuntimeError(f"{label} latest pointer has no checkpoint_id")
    checkpoint_step(checkpoint_id)
    return checkpoint_id


def _scientific_config() -> dict[str, object]:
    return {
        "optimizer": "hybrid_muon_adamw",
        "learning_rate": PEAK_LR,
        "weight_decay": 0.1,
        "beta1": 0.9,
        "beta2": 0.95,
        "adam_epsilon": 1e-8,
        "muon_momentum": 0.95,
        "muon_lr_multiplier": 1.0,
        "muon_update_rms": 0.18,
        "muon_weight_decay": 0.1,
        "max_grad_norm": 1.0,
        "precision": "fp16",
        "schedule": "wsqd",
        "warmup_tokens": 0,
        "stable_tokens": 0,
        "decay_tokens": COOLDOWN_TOKENS,
        "minimum_lr_ratio": MINIMUM_LR_RATIO,
        "schedule_anchor_tokens": SOURCE_EXPECTED_TOKENS,
        "cooldown_start_tokens": COOLDOWN_START_TOKENS,
        "settle_tokens": SETTLE_TOKENS,
        "settle_lr_ratio": SETTLE_LR_RATIO,
        "base_power": BASE_POWER,
        "seed": 17,
        "max_overflow_retries": 3,
    }


def _source_scientific_config() -> dict[str, object]:
    expected = _scientific_config()
    expected.update(
        schedule="wsd",
        warmup_tokens=500_039_680,
        stable_tokens=7_499_939_840,
        decay_tokens=2_000_027_648,
        minimum_lr_ratio=0.1,
    )
    for key in (
        "schedule_anchor_tokens",
        "cooldown_start_tokens",
        "settle_tokens",
        "settle_lr_ratio",
        "base_power",
    ):
        expected.pop(key)
    return expected


def _model_config_drift(config: Mapping[str, object]) -> dict[str, tuple[object, object]]:
    from model.config import ModelConfig

    expected = asdict(ModelConfig.substantive(architecture="gdn2_hybrid", gdn_chunk_size=32))
    return {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }


def validate_state(
    state: Mapping[str, object],
    *,
    step: int,
    source_checkpoint: bool,
    allowed_microbatches: frozenset[int],
    expected_cuda_rng_states: int | None = None,
) -> int:
    """Validate every frozen state identity before any provider rewrite."""

    if not SOURCE_STEP <= step <= FINAL_STEP:
        raise RuntimeError(f"checkpoint step {step} is outside the frozen horizon")
    if state.get("version") != 1:
        raise RuntimeError("checkpoint trainer-state version drifted")
    if state.get("global_step") != step:
        raise RuntimeError("checkpoint global_step disagrees with checkpoint ID")
    committed = step * TARGETS_PER_FULL_BLOCK
    if state.get("consumed_tokens") != committed:
        raise RuntimeError("checkpoint consumed-token count drifted")
    config = state.get("config")
    scheduler = state.get("scheduler")
    model_config = state.get("model_config")
    if not all(isinstance(item, Mapping) for item in (config, scheduler, model_config)):
        raise RuntimeError("checkpoint lacks config/scheduler/model_config mappings")
    assert isinstance(config, Mapping) and isinstance(scheduler, Mapping)
    assert isinstance(model_config, Mapping)

    expected = _source_scientific_config() if source_checkpoint else _scientific_config()
    drift = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if drift:
        raise RuntimeError("checkpoint scientific config drifted: " + json.dumps(drift, sort_keys=True))
    saved_microbatch = config.get("microbatch_size")
    if isinstance(saved_microbatch, bool) or saved_microbatch not in allowed_microbatches:
        raise RuntimeError(
            f"checkpoint execution microbatch {saved_microbatch!r} is not an authorized migration source"
        )
    model_drift = _model_config_drift(model_config)
    if model_drift:
        raise RuntimeError("checkpoint model config drifted: " + json.dumps(model_drift, sort_keys=True))
    for key in ("model", "optimizer", "scaler"):
        if not isinstance(state.get(key), Mapping):
            raise RuntimeError(f"checkpoint lacks {key} state")
    if state.get("python_rng_state") is None or state.get("torch_rng_state") is None:
        raise RuntimeError("checkpoint lacks exact RNG state")
    cuda_rng_states = state.get("cuda_rng_states")
    if not isinstance(cuda_rng_states, list) or not cuda_rng_states:
        raise RuntimeError("checkpoint lacks CUDA RNG state list")
    allowed_rng_counts = {1, 2} if saved_microbatch in {2, MICROBATCH_SIZE} else {1}
    if len(cuda_rng_states) not in allowed_rng_counts:
        raise RuntimeError(
            "checkpoint CUDA RNG topology drifted: "
            f"microbatch={saved_microbatch!r}, states={len(cuda_rng_states)}"
        )
    if expected_cuda_rng_states is not None and len(cuda_rng_states) != expected_cuda_rng_states:
        raise RuntimeError(
            f"checkpoint CUDA RNG state count {len(cuda_rng_states)} != {expected_cuda_rng_states}"
        )
    if scheduler.get("committed_tokens") != committed:
        raise RuntimeError("checkpoint scheduler committed-token count drifted")
    scheduler_config = scheduler.get("config")
    if not isinstance(scheduler_config, Mapping) or dict(scheduler_config) != dict(config):
        raise RuntimeError("checkpoint scheduler config disagrees with trainer config")
    if not source_checkpoint:
        last_lr = scheduler.get("last_lr")
        if isinstance(last_lr, bool) or not isinstance(last_lr, (int, float)):
            raise RuntimeError("checkpoint scheduler lacks a finite last_lr")
        expected = expected_lr(committed)
        if not math.isfinite(float(last_lr)) or not math.isclose(
            float(last_lr), expected, rel_tol=1e-10, abs_tol=1e-12
        ):
            raise RuntimeError(f"checkpoint scheduler LR drifted: {last_lr!r} != {expected!r}")
    return int(saved_microbatch)


def _patched_state(state: Mapping[str, object], *, source_checkpoint: bool) -> dict[str, object]:
    """Return a shallow state rewrite that touches only authorized config fields."""

    patched = dict(state)
    config = dict(state["config"])  # type: ignore[arg-type]
    scheduler = dict(state["scheduler"])  # type: ignore[arg-type]
    if source_checkpoint:
        config.update(_scientific_config())
        for key in ("checkpoint_every_steps", "evaluation_every_steps"):
            if key in state["config"]:  # type: ignore[operator]
                config[key] = state["config"][key]  # type: ignore[index]
        scheduler["committed_tokens"] = SOURCE_EXPECTED_TOKENS
        scheduler["last_lr"] = PEAK_LR
    config["microbatch_size"] = MICROBATCH_SIZE
    scheduler["config"] = dict(config)
    patched["config"] = config
    patched["scheduler"] = scheduler
    cuda_rng_states = state["cuda_rng_states"]
    assert isinstance(cuda_rng_states, list) and cuda_rng_states
    patched["cuda_rng_states"] = [cuda_rng_states[0]]
    return patched


def _verify_checkpoint_payload(root: Path, *, checkpoint_id: str) -> dict[str, object]:
    payload = json.loads((root / "checkpoint.json").read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise RuntimeError("checkpoint.json is not an object")
    step = checkpoint_step(checkpoint_id)
    pipeline = payload.get("pipeline_state")
    if not isinstance(pipeline, Mapping) or pipeline.get("last_consumed_block_id") != step - 1:
        raise RuntimeError("checkpoint data cursor drifted")
    return dict(payload)


def verify_checkpoint(root: Path, *, checkpoint_id: str, source_checkpoint: bool) -> int:
    from dataset.src.joint_checkpoint import verify_local_manifest
    from trainer.state import load_trainer_state_file, release_host_memory

    verify_local_manifest(root)
    _verify_checkpoint_payload(root, checkpoint_id=checkpoint_id)
    state = load_trainer_state_file(root / "trainer_state.pkl", map_location="cpu")
    try:
        return validate_state(
            state,
            step=checkpoint_step(checkpoint_id),
            source_checkpoint=source_checkpoint,
            allowed_microbatches=(
                frozenset({SOURCE_MICROBATCH_SIZE})
                if source_checkpoint
                else PRIOR_CONTINUATION_MICROBATCH_SIZES | {MICROBATCH_SIZE}
            ),
        )
    finally:
        del state
        release_host_memory()


def migrate_checkpoint(
    root: Path,
    *,
    checkpoint_id: str,
    dataset: Path,
    source_checkpoint: bool,
) -> dict[str, object]:
    """Atomically install a Modal-sliced checkpoint while retaining a hidden backup."""

    import torch
    from dataset.src.joint_checkpoint import verify_local_manifest
    from dataset.src.remote import sha256_path
    from trainer.identity import canonical_hash
    from trainer.state import load_trainer_state_file, release_host_memory

    verify_local_manifest(root)
    payload = _verify_checkpoint_payload(root, checkpoint_id=checkpoint_id)
    state = load_trainer_state_file(root / "trainer_state.pkl", map_location="cpu")
    try:
        saved_microbatch = validate_state(
            state,
            step=checkpoint_step(checkpoint_id),
            source_checkpoint=source_checkpoint,
            allowed_microbatches=(
                frozenset({SOURCE_MICROBATCH_SIZE})
                if source_checkpoint
                else PRIOR_CONTINUATION_MICROBATCH_SIZES | {MICROBATCH_SIZE}
            ),
        )
        cuda_rng_states = state.get("cuda_rng_states")
        assert isinstance(cuda_rng_states, list)
        if (
            saved_microbatch == MICROBATCH_SIZE
            and len(cuda_rng_states) == 1
            and not source_checkpoint
        ):
            return {"status": "already_modal_sliced", "from_microbatch": saved_microbatch}

        patched = _patched_state(state, source_checkpoint=source_checkpoint)
        manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
        production = manifest.get("production") if isinstance(manifest, Mapping) else None
        dataset_hash = production.get("configuration_hash") if isinstance(production, Mapping) else None
        if not isinstance(dataset_hash, str) or not dataset_hash:
            raise RuntimeError("rolling dataset manifest lacks configuration_hash")
        model_config = patched.get("model_config")
        if not isinstance(model_config, Mapping):
            raise RuntimeError("checkpoint model_config disappeared during migration")
        configuration_hash = canonical_hash(
            {
                "version": 1,
                "model": dict(model_config),
                "trainer": patched["config"],
                "dataset_configuration_hash": dataset_hash,
            }
        )

        parent = root.parent
        staging = parent / f".{checkpoint_id}.modal-h100-migration"
        backup = parent / f".{checkpoint_id}.pre-modal-h100"
        shutil.rmtree(staging, ignore_errors=True)
        swap_backup = backup
        if backup.exists():
            swap_backup = parent / f".{checkpoint_id}.pre-modal-h100-rng-projection"
            if swap_backup.exists():
                raise RuntimeError(
                    "both provider-migration backups exist while target still needs migration"
                )
        shutil.copytree(root, staging)
        (staging / "checkpoint_manifest.json").unlink(missing_ok=True)
        trainer_state = staging / "trainer_state.pkl"
        torch.save(patched, trainer_state)
        payload["configuration_hash"] = configuration_hash
        _write_json(staging / "checkpoint.json", payload)
        _write_json(
            staging / "local_manifest.json",
            {
                "version": 1,
                "files": [
                    {"name": "trainer_state.pkl", "sha256": sha256_path(trainer_state)},
                    {"name": "checkpoint.json", "sha256": sha256_path(staging / "checkpoint.json")},
                ],
            },
        )
        verify_local_manifest(staging)
        os.replace(root, swap_backup)
        try:
            os.replace(staging, root)
        except BaseException:
            os.replace(swap_backup, root)
            raise
        strict_state = load_trainer_state_file(root / "trainer_state.pkl", map_location="cpu")
        try:
            validate_state(
                strict_state,
                step=checkpoint_step(checkpoint_id),
                source_checkpoint=False,
                allowed_microbatches=frozenset({MICROBATCH_SIZE}),
                expected_cuda_rng_states=1,
            )
        finally:
            del strict_state
            release_host_memory()
        return {
            "status": "migrated_execution_slicing",
            "from_microbatch": saved_microbatch,
            "to_microbatch": MICROBATCH_SIZE,
            "backup": str(backup),
            "cuda_rng_states": {"from": len(cuda_rng_states), "to": 1, "selected_rank": 0},
        }
    finally:
        del state
        release_host_memory()


def _restore_pointer(
    runtime: Any,
    *,
    store: object,
    run_id: str,
    run_dir: Path,
    pointer: Mapping[str, object],
    source: str,
) -> dict[str, object]:
    from dataset.src.joint_checkpoint import restore_on_empty_vps
    from dataset.src.remote import TwoPhaseCheckpointPublisher

    checkpoint_id = _pointer_checkpoint_id(pointer, label=run_id)
    assert checkpoint_id is not None
    target = run_dir / "checkpoints" / checkpoint_id
    if target.is_dir():
        metadata = runtime._verified_checkpoint_metadata(target, checkpoint_id)
        metadata["source"] = f"local_and_{source}"
        return metadata
    restored = restore_on_empty_vps(
        publisher=TwoPhaseCheckpointPublisher(store, run_id=run_id),
        store=None,
        run_id=run_id,
        destination=run_dir,
        checkpoint_pointer=pointer,
        prefetch_shards=0,
    )
    metadata = runtime._verified_checkpoint_metadata(restored, checkpoint_id)
    metadata["source"] = source
    return metadata


def _exact_source_pointer(store: object) -> Mapping[str, object] | None:
    prefix = f"run/{SOURCE_RUN_ID}/checkpoints/{SOURCE_CHECKPOINT_ID}/last"
    manifest = store.read_json(f"{prefix}/checkpoint_manifest.json")  # type: ignore[attr-defined]
    if manifest is None:
        return None
    if not isinstance(manifest, Mapping):
        raise RuntimeError("exact step-15,500 checkpoint manifest is not a JSON object")
    return {
        "checkpoint_id": SOURCE_CHECKPOINT_ID,
        "last_prefix": prefix,
        "checkpoint_manifest": dict(manifest),
        "metric": None,
    }


def _local_best_checkpoint(checkpoint_dir: Path) -> tuple[str, float]:
    """Select the manifest-bearing local checkpoint with minimum validation loss."""

    candidates: list[tuple[float, int, str]] = []
    for root in checkpoint_dir.glob("step-*"):
        if not root.is_dir() or root.is_symlink():
            continue
        try:
            step = checkpoint_step(root.name)
        except ValueError:
            continue
        if not SOURCE_STEP <= step <= FINAL_STEP:
            continue
        payload_path = root / "checkpoint.json"
        manifest_path = root / "local_manifest.json"
        state_path = root / "trainer_state.pkl"
        if not payload_path.is_file() or not manifest_path.is_file() or not state_path.is_file():
            continue
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        metrics = payload.get("validation_metrics") if isinstance(payload, Mapping) else None
        loss = metrics.get("loss") if isinstance(metrics, Mapping) else None
        if isinstance(loss, bool) or not isinstance(loss, (int, float)):
            continue
        value = float(loss)
        if value < 0 or not math.isfinite(value):
            raise RuntimeError(f"checkpoint {root.name} has invalid validation loss")
        candidates.append((value, -step, root.name))
    if not candidates:
        raise RuntimeError("no local checkpoint has verified validation metrics for best selection")
    loss, _, checkpoint_id = min(candidates)
    return checkpoint_id, loss


def _ensure_dedicated_model_best(
    *,
    runtime: Any,
    checkpoint_dir: Path,
    bucket_latest_id: str,
) -> dict[str, object]:
    """Initialize or strictly improve the dedicated model repo from proven local best."""

    from model_repo_checkpoint import _dedicated_best_repo_id
    from trainer.best_model import get_dedicated_best_metric, publish_dedicated_best_model

    checkpoint_id, validation_loss = _local_best_checkpoint(checkpoint_dir)
    metric = -validation_loss
    root = checkpoint_dir / checkpoint_id
    verify_checkpoint(root, checkpoint_id=checkpoint_id, source_checkpoint=False)
    repo_id = _dedicated_best_repo_id(RUN_ID)
    existing_metric = get_dedicated_best_metric(
        repo_id=repo_id,
        run_id=RUN_ID,
        token=runtime._hf_token(),
    )
    if existing_metric is not None and metric <= existing_metric:
        return {
            "status": "already_current",
            "repo_id": repo_id,
            "checkpoint_id": checkpoint_id,
            "validation_loss": validation_loss,
            "existing_metric": existing_metric,
        }
    if existing_metric is not None and checkpoint_id != bucket_latest_id:
        raise RuntimeError(
            "refusing to replace an existing best model unless its improving checkpoint "
            "is the verified Bucket latest recovery copy"
        )
    result = publish_dedicated_best_model(
        repo_id=repo_id,
        run_id=RUN_ID,
        checkpoint_dir=root,
        checkpoint_id=checkpoint_id,
        metric=metric,
        validation_loss=validation_loss,
        token=runtime._hf_token(),
        recreate=True,
    )
    return dict(result)


def prepare(
    *,
    source_commit: str,
    repo_root: Path,
    run_root: Path,
    cache_root: Path,
    run_volume: object,
    cache_volume: object,
) -> dict[str, object]:
    """Resolve/verify the newest continuation, stage data, and migrate slicing on CPU."""

    import runtime
    from dataset.incremental_stage import stage_incremental_window_when_ready, verify_incremental_stage
    from dataset.qualification import get_profile
    from dataset.src.hf_bucket_shards import HuggingFaceBucketShardStore
    from model_repo_checkpoint import install_model_repo_checkpoint_transport
    from rolling_dataset import hf_dataset_bucket_id

    del repo_root
    install_model_repo_checkpoint_transport()
    run_dir = run_root / RUN_ID
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    local_id, local_step = runtime._latest_checkpoint(checkpoint_dir)
    bucket_store = runtime._hf_bucket_store()
    bucket_pointer = bucket_store.read_json(f"run/{RUN_ID}/latest.json")
    bucket_id = _pointer_checkpoint_id(bucket_pointer, label=f"{RUN_ID} bucket")
    bucket_step = checkpoint_step(bucket_id) if bucket_id else 0
    legacy_store = runtime._hf_model_repo_store()
    legacy_pointer = legacy_store.read_json(f"run/{RUN_ID}/latest.json")
    legacy_id = _pointer_checkpoint_id(legacy_pointer, label=f"{RUN_ID} legacy model repo")
    legacy_step = checkpoint_step(legacy_id) if legacy_id else 0
    for remote_step in (bucket_step, legacy_step):
        if remote_step and not SOURCE_STEP <= remote_step <= FINAL_STEP:
            raise RuntimeError("Hugging Face continuation pointer is outside the frozen horizon")

    selected_remote_source: str | None = None
    newest_remote_step = max(bucket_step, legacy_step)
    if newest_remote_step > local_step and bucket_step >= legacy_step:
        assert isinstance(bucket_pointer, Mapping)
        restored = _restore_pointer(
            runtime,
            store=bucket_store,
            run_id=RUN_ID,
            run_dir=run_dir,
            pointer=bucket_pointer,
            source="hf_bucket",
        )
        local_id, local_step = str(restored["checkpoint_id"]), int(restored["step"])
        selected_remote_source = "hf_bucket"
    elif newest_remote_step > local_step:
        assert isinstance(legacy_pointer, Mapping)
        restored = _restore_pointer(
            runtime,
            store=legacy_store,
            run_id=RUN_ID,
            run_dir=run_dir,
            pointer=legacy_pointer,
            source="legacy_hf_model_repo",
        )
        local_id, local_step = str(restored["checkpoint_id"]), int(restored["step"])
        selected_remote_source = "legacy_hf_model_repo"

    source_checkpoint = False
    if local_id is None:
        if bucket_pointer is not None or legacy_pointer is not None:
            raise RuntimeError("continuation pointer exists but no verified continuation was installed")
        source_pointer = _exact_source_pointer(legacy_store)
        if source_pointer is None:
            raise RuntimeError(
                f"no verified continuation exists and exact source {SOURCE_RUN_ID}/{SOURCE_CHECKPOINT_ID} is unavailable"
            )
        source_cache = run_root / ".deep-decay-source" / SOURCE_RUN_ID
        restored = _restore_pointer(
            runtime,
            store=legacy_store,
            run_id=SOURCE_RUN_ID,
            run_dir=source_cache,
            pointer=source_pointer,
            source="legacy_hf_model_repo_source",
        )
        if restored.get("checkpoint_id") != SOURCE_CHECKPOINT_ID or restored.get("step") != SOURCE_STEP:
            raise RuntimeError("exact source restore did not produce step-00015500")
        source_root = source_cache / "checkpoints" / SOURCE_CHECKPOINT_ID
        target = checkpoint_dir / SOURCE_CHECKPOINT_ID
        if target.exists():
            raise RuntimeError("step-00015500 continuation target unexpectedly exists")
        shutil.copytree(source_root, target)
        local_id, local_step = SOURCE_CHECKPOINT_ID, SOURCE_STEP
        source_checkpoint = True

    assert local_id is not None
    root = checkpoint_dir / local_id
    verify_checkpoint(root, checkpoint_id=local_id, source_checkpoint=source_checkpoint)
    if local_step == FINAL_STEP:
        return {
            "status": "training_complete",
            "run_id": RUN_ID,
            "checkpoint_id": local_id,
            "completed_steps": local_step,
            "final_step": FINAL_STEP,
        }

    profile = get_profile(DATASET_PROFILE)
    if profile.run_id != DATASET_RUN_ID or profile.sequences_per_block != SEQUENCES_PER_BLOCK:
        raise RuntimeError("rolling 10B dataset identity/geometry drifted")
    dataset_bucket_id = hf_dataset_bucket_id()
    dataset_store = HuggingFaceBucketShardStore(
        dataset_bucket_id,
        token=runtime._hf_token(),
        private=True,
        create_bucket=False,
    )
    dataset = cache_root / "deep-decay-10b" / RUN_ID / f"from-{local_step:08d}"
    staged = stage_incremental_window_when_ready(
        store=dataset_store,
        run_id=DATASET_RUN_ID,
        destination=dataset,
        start_block_id=local_step,
    )
    if staged.get("status") != "ready":
        raise RuntimeError(f"dataset stage did not become ready: {staged}")
    verification = verify_incremental_stage(
        destination=dataset,
        bucket_id=dataset_bucket_id,
        run_id=DATASET_RUN_ID,
        required_train_block=local_step,
    )
    migration = migrate_checkpoint(
        root,
        checkpoint_id=local_id,
        dataset=dataset,
        source_checkpoint=source_checkpoint,
    )
    verify_checkpoint(root, checkpoint_id=local_id, source_checkpoint=False)

    transport_path = run_dir / "hf_checkpoint_transport.json"
    previous_source_commit: str | None = None
    previous_transport = root / "drive_manifest.json"
    if previous_transport.is_file():
        previous_payload = json.loads(previous_transport.read_text(encoding="utf-8"))
        observed_source = previous_payload.get("source_commit") if isinstance(previous_payload, Mapping) else None
        if isinstance(observed_source, str) and observed_source and observed_source != source_commit:
            previous_source_commit = observed_source
    transport = runtime._write_hf_transport_manifest(
        transport_path,
        run_id=RUN_ID,
        dataset=dataset,
        dataset_profile=DATASET_PROFILE,
        source_commit=source_commit,
        microbatch_size=MICROBATCH_SIZE,
        resume_parent_source_commit=previous_source_commit,
        bucket_id=runtime._hf_checkpoint_bucket_id(),
    )
    if bucket_step < local_step:
        from dataset.src.remote import TwoPhaseCheckpointPublisher

        publisher = TwoPhaseCheckpointPublisher(bucket_store, run_id=RUN_ID)
        publisher.publish(
            root,
            checkpoint_id=local_id,
            drive_manifest=transport,
            metric=None,
            best_metric=None,
        )
        cleanup = bucket_store.prune_run_checkpoints(run_id=RUN_ID, checkpoint_id=local_id)
    else:
        cleanup = {"status": "already_current", "checkpoint_id": local_id}
    durable_pointer = bucket_store.read_json(f"run/{RUN_ID}/latest.json")
    durable_id = _pointer_checkpoint_id(durable_pointer, label=f"{RUN_ID} bucket after migration")
    if durable_id != local_id:
        raise RuntimeError(
            f"Hugging Face checkpoint Bucket latest {durable_id!r} != verified local {local_id!r}"
        )
    best_model = _ensure_dedicated_model_best(
        runtime=runtime,
        checkpoint_dir=checkpoint_dir,
        bucket_latest_id=durable_id,
    )
    modal_contract = contract()
    contract_path = run_dir / "modal_deep_decay_contract.json"
    if contract_path.is_file():
        existing = json.loads(contract_path.read_text(encoding="utf-8"))
        if existing != modal_contract:
            raise RuntimeError("Modal deep-decay execution contract drifted")
    else:
        _write_json(contract_path, modal_contract)
    getattr(run_volume, "commit")()
    getattr(cache_volume, "commit")()
    return {
        "status": "ready",
        "run_id": RUN_ID,
        "resume_checkpoint_id": local_id,
        "completed_steps": local_step,
        "remaining_steps": FINAL_STEP - local_step,
        "final_step": FINAL_STEP,
        "required_block": local_step,
        "dataset_dir": str(dataset),
        "dataset_bucket_id": dataset_bucket_id,
        "checkpoint_bucket_id": runtime._hf_checkpoint_bucket_id(),
        "checkpoint_source": selected_remote_source
        or ("exact_step15500_fallback" if source_checkpoint else "local_modal_volume"),
        "continuation_hf_checkpoint_id": durable_id,
        "legacy_model_repo_checkpoint_id": legacy_id,
        "bucket_migration_cleanup": cleanup,
        "best_model": best_model,
        "migration": migration,
        "verification": verification,
        "expected_lr": expected_lr(local_step * TARGETS_PER_FULL_BLOCK),
    }


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


def train(
    *,
    source_commit: str,
    dataset_dir: str,
    resume_checkpoint_id: str,
    steps: int,
    repo_root: Path,
    run_root: Path,
    cache_root: Path,
    run_volume: object,
    cache_volume: object,
) -> dict[str, object]:
    """Run one exact-resume Modal H100 segment after the CPU preparation gate."""

    import re
    import runtime
    from dataset.incremental_stage import verify_incremental_stage
    from model_repo_checkpoint import install_model_repo_checkpoint_transport
    from profiles import resolve_presets
    from rolling_dataset import hf_dataset_bucket_id

    if steps <= 0:
        raise RuntimeError("H100 was allocated with no remaining work")
    install_model_repo_checkpoint_transport()
    dataset = Path(dataset_dir).resolve(strict=True)
    cache_resolved = cache_root.resolve(strict=True)
    try:
        dataset.relative_to(cache_resolved)
    except ValueError as error:
        raise RuntimeError("deep-decay dataset is outside the Modal cache Volume") from error
    run_dir = run_root / RUN_ID
    checkpoint_dir = run_dir / "checkpoints"
    latest_id, latest_step = runtime._latest_checkpoint(checkpoint_dir)
    if latest_id != resume_checkpoint_id:
        raise RuntimeError("checkpoint advanced or changed after CPU staging; rerun the launcher")
    if latest_step != checkpoint_step(resume_checkpoint_id):
        raise RuntimeError("resume checkpoint ID/step mismatch after CPU staging")
    verify_checkpoint(
        checkpoint_dir / resume_checkpoint_id,
        checkpoint_id=resume_checkpoint_id,
        source_checkpoint=False,
    )
    verify_incremental_stage(
        destination=dataset,
        bucket_id=hf_dataset_bucket_id(),
        run_id=DATASET_RUN_ID,
        required_train_block=latest_step,
    )

    model_preset, token_preset = resolve_presets("100M", "10B")
    if token_preset.dataset_profile != DATASET_PROFILE:
        raise RuntimeError("100M/10B Modal profile drifted")
    os.environ["SMALL_LLM_MODAL_ROLLING_DATASET"] = "1"
    os.environ["SMALL_LLM_DATASET_SHARD_BUCKET"] = hf_dataset_bucket_id()
    os.environ["SMALL_LLM_DATASET_SHARD_RUN_ID"] = DATASET_RUN_ID
    os.environ["SMALL_LLM_DATASET_SHARD_PREFETCH"] = "1"

    checkpoint_bucket_id = runtime._hf_checkpoint_bucket_id()
    remote_manifest = run_dir / "hf_checkpoint_transport.json"
    runtime._write_hf_transport_manifest(
        remote_manifest,
        run_id=RUN_ID,
        dataset=dataset,
        dataset_profile=DATASET_PROFILE,
        source_commit=source_commit,
        microbatch_size=MICROBATCH_SIZE,
        resume_parent_source_commit=None,
        bucket_id=checkpoint_bucket_id,
    )
    plan: dict[str, Any] = {
        "trainer": {
            "warmup_tokens": 0,
            "stable_tokens": 0,
            "decay_tokens": COOLDOWN_TOKENS,
            "validation_blocks": 16,
        }
    }
    environment = runtime._gpu_environment()
    if "H100" not in str(environment.get("name", "")):
        raise RuntimeError(f"deep-decay Modal lane requires one H100, got {environment!r}")
    gpu_tag = re.sub(r"[^a-z0-9]+", "-", str(environment["name"]).lower()).strip("-")
    command = runtime._trainer_command(
        model=model_preset,
        tokens=token_preset,
        dataset=dataset,
        plan=plan,
        checkpoint_dir=checkpoint_dir,
        steps=steps,
        microbatch=MICROBATCH_SIZE,
        precision="fp16",
        wandb_run_id=RUN_ID,
        gpu_tag=gpu_tag,
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
    _replace_option(
        command,
        "--wandb-resume",
        "allow" if resume_checkpoint_id == SOURCE_CHECKPOINT_ID else "must",
    )
    _replace_option(command, "--wandb-run-name", "100M/10B deep-decay continuation from step 15500")
    command += [
        "--schedule-anchor-tokens", str(SOURCE_EXPECTED_TOKENS),
        "--cooldown-start-tokens", str(COOLDOWN_START_TOKENS),
        "--settle-tokens", str(SETTLE_TOKENS),
        "--settle-lr-ratio", str(SETTLE_LR_RATIO),
        "--base-power", str(BASE_POWER),
    ]
    _replace_tag(command, "modal", "modal-deep-decay")

    expected_start_lr = expected_lr(latest_step * TARGETS_PER_FULL_BLOCK)
    log_path = run_dir / "evidence" / f"modal-h100-from-{resume_checkpoint_id}.log"
    started = time.perf_counter()
    runtime._run(command, cwd=repo_root, log_path=log_path)
    getattr(run_volume, "commit")()
    getattr(cache_volume, "commit")()
    final_id, final_step = runtime._latest_checkpoint(checkpoint_dir)
    expected_step = latest_step + steps
    if final_id is None or final_step != expected_step:
        raise RuntimeError(f"durable checkpoint step {final_step} != expected {expected_step}")
    rows = runtime._training_rows(log_path)
    if not rows:
        raise RuntimeError("training completed without any parseable finite-update telemetry")
    for row in rows:
        if not all(math.isfinite(float(row[key])) for key in ("loss", "gradient_norm", "tokens_per_second")):
            raise RuntimeError("training log contains non-finite update telemetry")
    first_lr = rows[0].get("learning_rate")
    if isinstance(first_lr, (int, float)) and not isinstance(first_lr, bool):
        expected_first_lr = expected_lr((latest_step + 1) * TARGETS_PER_FULL_BLOCK)
        if not math.isclose(float(first_lr), expected_first_lr, rel_tol=1e-9, abs_tol=1e-12):
            raise RuntimeError(f"first update LR {first_lr!r} != expected {expected_first_lr!r}")
    return {
        "status": "complete" if final_step == FINAL_STEP else "segment_complete",
        "run_id": RUN_ID,
        "checkpoint_id": final_id,
        "restored_checkpoint_id": resume_checkpoint_id,
        "completed_steps": final_step,
        "final_step": FINAL_STEP,
        "elapsed_seconds": time.perf_counter() - started,
        "finite_updates": len(rows),
        "expected_start_lr": expected_start_lr,
        "first_update_lr": first_lr,
        "microbatch_size": MICROBATCH_SIZE,
        "sequences_per_block": SEQUENCES_PER_BLOCK,
        "gpu": environment,
    }


__all__ = [
    "FINAL_STEP",
    "MICROBATCH_SIZE",
    "RUN_ID",
    "SOURCE_CHECKPOINT_ID",
    "checkpoint_step",
    "contract",
    "dry_run_payload",
    "expected_lr",
    "migrate_checkpoint",
    "prepare",
    "train",
    "validate_state",
    "verify_checkpoint",
]
