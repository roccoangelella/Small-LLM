#!/usr/bin/env python3
"""Kaggle 2xT4 deep-decay entrypoint with automatic provider migration.

The scientific deep-decay implementation lives in
``deep_decay_10b_from_15500_impl.py``. Before delegating to it, this shim
resolves the newest verified continuation checkpoint from the shared Hugging
Face namespace and rewrites only provider execution topology when needed.

Authorized continuation slices are Kaggle microbatch 2, Beam microbatch 4, and
Modal microbatch 16. Kaggle canonicalizes all of them to microbatch 2 with two
CUDA RNG states while preserving model, optimizer, scaler, scheduler/LR, data
cursor, optimizer-block geometry, consumed tokens, CPU RNG state, and the
ADR-0095 scientific schedule.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

KAGGLE = Path(__file__).resolve().parent
ROOT = KAGGLE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(KAGGLE) not in sys.path:
    sys.path.insert(0, str(KAGGLE))

import deep_decay_10b_from_15500_impl as _impl
from trainer.deep_decay_provider_migration import (
    execution_rewrite_needed,
    rewrite_execution_state,
    validate_target_execution_state,
)

# Re-export the implementation surface so existing tooling/tests importing this
# canonical module keep working.
for _name in dir(_impl):
    if not _name.startswith("__"):
        globals().setdefault(_name, getattr(_impl, _name))

HF_HUB_VERSION = "1.5.0"
HF_BUCKET_METHODS = (
    "create_bucket",
    "list_bucket_tree",
    "download_bucket_files",
    "batch_bucket_files",
)


def _hf_bucket_api_available() -> bool:
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return False
    return all(callable(getattr(HfApi, name, None)) for name in HF_BUCKET_METHODS)


def _ensure_host_hf_bucket_runtime(argv: Sequence[str]) -> None:
    """Re-exec with a private HF 1.5 runtime when Kaggle ships an older client."""

    if _hf_bucket_api_available():
        return

    runtime = _impl.WORK_ROOT / ".runtime" / f"huggingface-hub-{HF_HUB_VERSION}"
    marker = runtime / ".complete"
    if not marker.is_file():
        staging = runtime.with_name(runtime.name + ".tmp")
        shutil.rmtree(staging, ignore_errors=True)
        staging.parent.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--quiet",
                "--disable-pip-version-check",
                "--target",
                str(staging),
                f"huggingface_hub=={HF_HUB_VERSION}",
            ]
        )
        (staging / ".complete").write_text(HF_HUB_VERSION + "\n", encoding="utf-8")
        shutil.rmtree(runtime, ignore_errors=True)
        os.replace(staging, runtime)

    env = dict(os.environ)
    previous = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(runtime) + (os.pathsep + previous if previous else "")
    print(
        f"[kaggle-deep-decay] Kaggle host lacks HF Storage Buckets API; "
        f"restarting with private huggingface_hub=={HF_HUB_VERSION}",
        flush=True,
    )
    os.execve(
        sys.executable,
        [sys.executable, str(Path(__file__).resolve()), *list(argv)],
        env,
    )


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


def _remote_continuation(runtime_base: Any) -> tuple[str | None, int]:
    remote = _impl._remote_checkpoint_state(runtime_base, run_id=RUN_ID)
    if remote is None:
        return None, 0
    checkpoint_id = remote.get("checkpoint_id")
    step = remote.get("step")
    if not isinstance(checkpoint_id, str) or isinstance(step, bool) or not isinstance(step, int):
        raise RuntimeError("HF deep-decay continuation state is malformed")
    if not SOURCE_STEP <= step <= FINAL_STEP:
        raise RuntimeError(f"HF deep-decay checkpoint step {step} is outside the frozen horizon")
    return checkpoint_id, step


def _restore_newest_verified_continuation(runtime_base: Any) -> tuple[str | None, int]:
    """Prefer the newest verified checkpoint across Kaggle scratch and shared HF."""

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    local_id, local_step = runtime_base._latest_checkpoint(CHECKPOINT_DIR)
    remote_id, remote_step = _remote_continuation(runtime_base)
    if remote_id is None or local_step >= remote_step:
        return local_id, local_step

    refresh = RUN_DIR.parent / f".{RUN_ID}.hf-refresh"
    shutil.rmtree(refresh, ignore_errors=True)
    restored = _impl._restore_pointer(
        runtime_base,
        run_id=RUN_ID,
        destination_run_dir=refresh,
    )
    if restored is None:
        raise RuntimeError("HF continuation pointer disappeared during refresh")
    restored_id = str(restored["checkpoint_id"])
    restored_step = int(restored["step"])
    if restored_id != remote_id or restored_step != remote_step:
        raise RuntimeError(
            "HF continuation changed during refresh: "
            f"expected {remote_id}/{remote_step}, got {restored_id}/{restored_step}"
        )

    from dataset.src.joint_checkpoint import verify_local_manifest

    source = refresh / "checkpoints" / restored_id
    verify_local_manifest(source)
    target = CHECKPOINT_DIR / restored_id
    if not target.exists():
        staging = CHECKPOINT_DIR / f".{restored_id}.hf-refresh"
        shutil.rmtree(staging, ignore_errors=True)
        shutil.copytree(source, staging)
        verify_local_manifest(staging)
        os.replace(staging, target)
    else:
        verify_local_manifest(target)
    shutil.rmtree(refresh, ignore_errors=True)
    print(
        f"[kaggle-deep-decay] selected newer HF continuation {restored_id} "
        f"over local step {local_step}",
        flush=True,
    )
    return restored_id, restored_step


def _install_execution_migration(
    *,
    checkpoint_id: str,
    dataset: Path,
) -> bool:
    """Canonicalize an authorized provider checkpoint to Kaggle 2xT4 execution."""

    import torch
    from dataset.src.joint_checkpoint import verify_local_manifest
    from dataset.src.remote import sha256_path
    from trainer.identity import canonical_hash
    from trainer.state import load_trainer_state_file, release_host_memory

    step = _impl._checkpoint_step(checkpoint_id)
    root = CHECKPOINT_DIR / checkpoint_id
    verify_local_manifest(root)

    try:
        checkpoint_payload = json.loads(
            (root / "checkpoint.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError, TypeError) as error:
        raise RuntimeError("deep-decay checkpoint.json is not readable JSON") from error
    if not isinstance(checkpoint_payload, Mapping):
        raise RuntimeError("deep-decay checkpoint.json is not an object")
    pipeline = checkpoint_payload.get("pipeline_state")
    if (
        not isinstance(pipeline, Mapping)
        or pipeline.get("last_consumed_block_id") != step - 1
    ):
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

        scientific_drift = _scientific_checkpoint_drift(config)
        if scientific_drift:
            raise RuntimeError(
                "deep-decay checkpoint scientific config drifted: "
                + json.dumps(scientific_drift, sort_keys=True)
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

        if not execution_rewrite_needed(state, target_microbatch=MICROBATCH_SIZE):
            validate_target_execution_state(state, target_microbatch=MICROBATCH_SIZE)
            return False
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

        staging = CHECKPOINT_DIR / f".{checkpoint_id}.kaggle-provider-migration"
        backup = CHECKPOINT_DIR / f".{checkpoint_id}.pre-kaggle-provider"
        shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(backup, ignore_errors=True)
        shutil.copytree(root, staging)
        (staging / "checkpoint_manifest.json").unlink(missing_ok=True)

        trainer_state_path = staging / "trainer_state.pkl"
        torch.save(patched_state, trainer_state_path)

        patched_checkpoint = dict(checkpoint_payload)
        patched_checkpoint["configuration_hash"] = configuration_hash
        _impl._write_json(staging / "checkpoint.json", patched_checkpoint)
        _impl._write_json(
            staging / "local_manifest.json",
            {
                "version": 1,
                "files": [
                    {
                        "name": "trainer_state.pkl",
                        "sha256": sha256_path(trainer_state_path),
                    },
                    {
                        "name": "checkpoint.json",
                        "sha256": sha256_path(staging / "checkpoint.json"),
                    },
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
            validate_target_execution_state(
                strict_state,
                target_microbatch=MICROBATCH_SIZE,
            )
        finally:
            del strict_state
            release_host_memory()

        print(
            f"[kaggle-deep-decay] migrated {checkpoint_id} execution "
            f"{migration['from_microbatch']}->{migration['to_microbatch']}; "
            f"cuda_rng_states={migration['cuda_rng_states']}; "
            f"policy={migration['cuda_rng_policy']}; scientific state unchanged",
            flush=True,
        )
        return True
    finally:
        del state
        release_host_memory()


def _migrate_existing_deep_decay_checkpoint(runtime_base: Any) -> bool:
    """Refresh from HF if newer, then canonicalize execution before strict prepare."""

    checkpoint_id, step = _restore_newest_verified_continuation(runtime_base)
    if checkpoint_id is None:
        return False
    if not SOURCE_STEP <= step <= FINAL_STEP:
        raise RuntimeError(f"deep-decay checkpoint step {step} is outside the frozen horizon")

    root = CHECKPOINT_DIR / checkpoint_id
    from dataset.src.joint_checkpoint import verify_local_manifest
    from trainer.state import load_trainer_state_file, release_host_memory

    verify_local_manifest(root)
    state = load_trainer_state_file(root / "trainer_state.pkl", map_location="cpu")
    try:
        needs_migration = execution_rewrite_needed(
            state,
            target_microbatch=MICROBATCH_SIZE,
        )
        if not needs_migration:
            validate_target_execution_state(state, target_microbatch=MICROBATCH_SIZE)
            return False
    finally:
        del state
        release_host_memory()

    # The manifest identity is invariant across rolling windows, so for a
    # completed final checkpoint use the final consumable block to obtain it.
    stage_block = min(step, FINAL_STEP - 1)
    dataset = _impl._stage_dataset(runtime_base, start_block_id=stage_block)
    migrated = _install_execution_migration(
        checkpoint_id=checkpoint_id,
        dataset=dataset,
    )
    if migrated:
        verified_step = _impl._verify_deep_decay_checkpoint(runtime_base, checkpoint_id)
        if verified_step != step:
            raise RuntimeError("migrated deep-decay checkpoint step changed unexpectedly")
    return migrated


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--dry-run" not in args:
        _ensure_host_hf_bucket_runtime(args)
        runtime_base = _impl._beam_runtime()
        migrated = _migrate_existing_deep_decay_checkpoint(runtime_base)
        if migrated:
            checkpoint_id, _ = runtime_base._latest_checkpoint(CHECKPOINT_DIR)
            if checkpoint_id is None:
                raise RuntimeError("provider migration completed without a local checkpoint")
            os.environ["SMALL_LLM_KAGGLE_PROVIDER_MIGRATION_CHECKPOINT_ID"] = checkpoint_id
    return int(_impl.main(args))


if __name__ == "__main__":
    raise SystemExit(main())
