#!/usr/bin/env python3
"""Kaggle 2xT4 deep-decay entrypoint with one-time execution migration.

The scientific deep-decay implementation lives in
``deep_decay_10b_from_15500_impl.py``. Before delegating to it, this shim accepts
an already-published deep-decay checkpoint created on the single-GPU lane with
execution microbatch four, rewrites only execution slicing to Kaggle microbatch
two, recomputes the checkpoint configuration hash, and then lets the strict
implementation re-verify the migrated checkpoint.

Model, optimizer, scaler, RNG, data cursor, optimizer-block geometry, consumed
tokens, and the ADR-0095 LR schedule are not changed by this migration.
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
if str(KAGGLE) not in sys.path:
    sys.path.insert(0, str(KAGGLE))

import deep_decay_10b_from_15500_impl as _impl

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


def _install_execution_migration(
    *,
    checkpoint_id: str,
    dataset: Path,
) -> bool:
    """Rewrite a verified deep-decay microbatch-4 checkpoint to microbatch 2."""

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

        saved_microbatch = config.get("microbatch_size")
        if saved_microbatch == MICROBATCH_SIZE:
            return False
        if saved_microbatch != SOURCE_MICROBATCH_SIZE:
            raise RuntimeError(
                "deep-decay checkpoint execution microbatch is neither the "
                f"single-GPU source value {SOURCE_MICROBATCH_SIZE} nor the "
                f"Kaggle value {MICROBATCH_SIZE}: {saved_microbatch!r}"
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
        if scheduler_config.get("microbatch_size") != SOURCE_MICROBATCH_SIZE:
            raise RuntimeError("deep-decay scheduler execution microbatch drifted")

        patched_config = dict(config)
        patched_config["microbatch_size"] = MICROBATCH_SIZE
        patched_scheduler = dict(scheduler)
        patched_scheduler_config = dict(scheduler_config)
        patched_scheduler_config["microbatch_size"] = MICROBATCH_SIZE
        patched_scheduler["config"] = patched_scheduler_config
        state["config"] = patched_config
        state["scheduler"] = patched_scheduler

        configuration_hash = canonical_hash(
            {
                "version": 1,
                "model": dict(model_config),
                "trainer": patched_config,
                "dataset_configuration_hash": _dataset_configuration_hash(dataset),
            }
        )

        staging = CHECKPOINT_DIR / f".{checkpoint_id}.kaggle-micro2"
        backup = CHECKPOINT_DIR / f".{checkpoint_id}.pre-kaggle-micro4"
        shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(backup, ignore_errors=True)
        staging.mkdir(parents=True, exist_ok=False)

        trainer_state_path = staging / "trainer_state.pkl"
        torch.save(state, trainer_state_path)

        patched_checkpoint = dict(checkpoint_payload)
        patched_checkpoint["configuration_hash"] = configuration_hash
        _impl._write_json(staging / "checkpoint.json", patched_checkpoint)
        _impl._write_json(
            staging / "local_manifest.json",
            {
                "files": [
                    {
                        "name": "trainer_state.pkl",
                        "sha256": sha256_path(trainer_state_path),
                    },
                    {
                        "name": "checkpoint.json",
                        "sha256": sha256_path(staging / "checkpoint.json"),
                    },
                ]
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

        print(
            f"[kaggle-deep-decay] migrated {checkpoint_id} execution slicing "
            f"{SOURCE_MICROBATCH_SIZE}->{MICROBATCH_SIZE}; scientific state unchanged",
            flush=True,
        )
        return True
    finally:
        del state
        release_host_memory()


def _migrate_existing_deep_decay_checkpoint(runtime_base: Any) -> bool:
    """Migrate a local/HF deep-decay checkpoint before strict Kaggle prepare."""

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_id, step = runtime_base._latest_checkpoint(CHECKPOINT_DIR)
    if checkpoint_id is None:
        restored = _impl._restore_pointer(
            runtime_base,
            run_id=RUN_ID,
            destination_run_dir=RUN_DIR,
        )
        if restored is None:
            return False
        checkpoint_id = str(restored["checkpoint_id"])
        step = int(restored["step"])

    root = CHECKPOINT_DIR / checkpoint_id
    from dataset.src.joint_checkpoint import verify_local_manifest
    from trainer.state import load_trainer_state_file, release_host_memory

    verify_local_manifest(root)
    state = load_trainer_state_file(root / "trainer_state.pkl", map_location="cpu")
    try:
        config = state.get("config")
        if not isinstance(config, Mapping):
            raise RuntimeError("deep-decay checkpoint lacks config mapping")
        saved_microbatch = config.get("microbatch_size")
        if saved_microbatch == MICROBATCH_SIZE:
            return False
        if saved_microbatch != SOURCE_MICROBATCH_SIZE:
            raise RuntimeError(
                "deep-decay checkpoint execution microbatch cannot be migrated: "
                f"{saved_microbatch!r}"
            )
    finally:
        del state
        release_host_memory()

    if not SOURCE_STEP <= step <= FINAL_STEP:
        raise RuntimeError(f"deep-decay checkpoint step {step} is outside the frozen horizon")

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
        _migrate_existing_deep_decay_checkpoint(runtime_base)
    return int(_impl.main(args))


if __name__ == "__main__":
    raise SystemExit(main())
