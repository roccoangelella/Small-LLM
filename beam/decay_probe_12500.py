#!/usr/bin/env python3
"""Temporary Beam counterfactual: fork 100M/10B step 12,500 onto the 2B cooldown.

Usage from a clean repository root:

    python beam/decay_probe_12500.py --dry-run
    python beam/decay_probe_12500.py --gpu RTX4090

This is deliberately a diagnostic fork, not an exact resume of the production
100M/10B trajectory. It requires the exact local Beam Volume checkpoint
``step-00012500`` and never substitutes a newer checkpoint. The fork keeps the
model, optimizer, scaler, RNG, and data cursor, but replaces only the LR
scheduler with the historical block-64 2B WSD schedule. At step 12,500 that
schedule is already 9.60% into decay, so LR starts near 2.94e-4 and reaches
3e-5 at step 15,259. The probe therefore consumes 2,759 more optimizer blocks
(~361.63M targets) from the 10B corpus.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beam import function  # noqa: E402
from beam import launch as base  # noqa: E402

SOURCE_RUN_ID = "100m-10b-data-001"
SOURCE_STEP = 12_500
SOURCE_CHECKPOINT_ID = f"step-{SOURCE_STEP:08d}"
PROBE_RUN_ID = "100m-10b-decay-probe-step12500"
DATASET_PROFILE = "modal-10b-b64"
DATASET_RUN_ID = "modal-10b-b64-dataset-001"
MICROBATCH_SIZE = 4
TARGETS_PER_FULL_BLOCK = 64 * 2048
PEAK_LR = 3e-4
MINIMUM_LR_RATIO = 0.1

# Exact historical block-64 2B WSD contract. The counterfactual uses this
# absolute-token schedule rather than inventing a new arbitrary low LR.
MATCHED_WARMUP_TOKENS = 100_007_936
MATCHED_STABLE_TOKENS = 1_499_987_968
MATCHED_DECAY_TOKENS = 399_998_976
MATCHED_DECAY_END_TOKENS = (
    MATCHED_WARMUP_TOKENS + MATCHED_STABLE_TOKENS + MATCHED_DECAY_TOKENS
)
SOURCE_EXPECTED_TOKENS = SOURCE_STEP * TARGETS_PER_FULL_BLOCK
PROBE_ADDITIONAL_STEPS = math.ceil(
    (MATCHED_DECAY_END_TOKENS - SOURCE_EXPECTED_TOKENS) / TARGETS_PER_FULL_BLOCK
)
PROBE_FINAL_STEP = SOURCE_STEP + PROBE_ADDITIONAL_STEPS
PROBE_RUN_DIR = base.RUN_ROOT / PROBE_RUN_ID
PROBE_CHECKPOINT_DIR = PROBE_RUN_DIR / "checkpoints"
PROBE_CONTRACT_PATH = PROBE_RUN_DIR / "decay_probe_contract.json"


def _expected_lr(tokens: int) -> float:
    stable_end = MATCHED_WARMUP_TOKENS + MATCHED_STABLE_TOKENS
    progress = min(
        1.0,
        max(0.0, (tokens - stable_end) / MATCHED_DECAY_TOKENS),
    )
    minimum = PEAK_LR * MINIMUM_LR_RATIO
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum + (PEAK_LR - minimum) * cosine


def _probe_contract() -> dict[str, object]:
    return {
        "version": 1,
        "kind": "temporary_matched_2b_wsd_decay_probe",
        "source_run_id": SOURCE_RUN_ID,
        "source_checkpoint_id": SOURCE_CHECKPOINT_ID,
        "source_step": SOURCE_STEP,
        "source_expected_consumed_tokens": SOURCE_EXPECTED_TOKENS,
        "probe_run_id": PROBE_RUN_ID,
        "dataset_profile": DATASET_PROFILE,
        "dataset_run_id": DATASET_RUN_ID,
        "microbatch_size": MICROBATCH_SIZE,
        "peak_lr": PEAK_LR,
        "lr_at_source_step": _expected_lr(SOURCE_EXPECTED_TOKENS),
        "minimum_lr_ratio": MINIMUM_LR_RATIO,
        "minimum_lr": PEAK_LR * MINIMUM_LR_RATIO,
        "warmup_tokens": MATCHED_WARMUP_TOKENS,
        "stable_tokens": MATCHED_STABLE_TOKENS,
        "decay_tokens": MATCHED_DECAY_TOKENS,
        "decay_end_tokens": MATCHED_DECAY_END_TOKENS,
        "additional_steps": PROBE_ADDITIONAL_STEPS,
        "final_step": PROBE_FINAL_STEP,
        "additional_full_block_targets": PROBE_ADDITIONAL_STEPS * TARGETS_PER_FULL_BLOCK,
        "scientific_change": "scheduler_only; preserve model/optimizer/scaler/RNG/data cursor",
    }


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _fork_source_checkpoint(*, dataset: Path) -> None:
    """Create a clean probe checkpoint with only the trainer schedule changed."""

    import torch
    from dataset.src.joint_checkpoint import verify_local_manifest
    from dataset.src.remote import sha256_path
    from trainer.identity import canonical_hash
    from trainer.state import load_trainer_state_file, release_host_memory

    source = base.RUN_ROOT / SOURCE_RUN_ID / "checkpoints" / SOURCE_CHECKPOINT_ID
    if not source.is_dir():
        raise RuntimeError(
            f"exact source checkpoint is absent from the Beam run Volume: {source}; "
            "this probe refuses to substitute latest/nearest state"
        )
    verify_local_manifest(source)
    source_payload = json.loads((source / "checkpoint.json").read_text(encoding="utf-8"))
    if not isinstance(source_payload, Mapping):
        raise RuntimeError("source checkpoint.json is not an object")
    pipeline = source_payload.get("pipeline_state")
    if not isinstance(pipeline, Mapping) or pipeline.get("last_consumed_block_id") != SOURCE_STEP - 1:
        raise RuntimeError("step-12500 checkpoint does not carry the expected data cursor")

    source_state_path = source / "trainer_state.pkl"
    state = load_trainer_state_file(source_state_path, map_location="cpu")
    try:
        if state.get("global_step") != SOURCE_STEP:
            raise RuntimeError("step-12500 trainer state has the wrong global_step")
        if state.get("consumed_tokens") != SOURCE_EXPECTED_TOKENS:
            raise RuntimeError(
                "step-12500 trainer state has unexpected consumed-token count: "
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
                f"probe requires {MICROBATCH_SIZE}"
            )
        if config.get("learning_rate") != PEAK_LR or config.get("schedule") != "wsd":
            raise RuntimeError("source checkpoint does not match the expected 3e-4 WSD recipe")

        patched_config = dict(config)
        patched_config.update(
            schedule="wsd",
            warmup_tokens=MATCHED_WARMUP_TOKENS,
            stable_tokens=MATCHED_STABLE_TOKENS,
            decay_tokens=MATCHED_DECAY_TOKENS,
            minimum_lr_ratio=MINIMUM_LR_RATIO,
        )
        patched_scheduler = dict(scheduler)
        patched_scheduler["config"] = dict(patched_config)
        patched_scheduler["committed_tokens"] = SOURCE_EXPECTED_TOKENS
        patched_scheduler["last_lr"] = _expected_lr(SOURCE_EXPECTED_TOKENS)
        state["config"] = patched_config
        state["scheduler"] = patched_scheduler

        manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
        production = manifest.get("production") if isinstance(manifest, Mapping) else None
        dataset_configuration_hash = (
            production.get("configuration_hash") if isinstance(production, Mapping) else None
        )
        new_configuration_hash = canonical_hash(
            {
                "version": 1,
                "model": dict(model_config),
                "trainer": patched_config,
                "dataset_configuration_hash": dataset_configuration_hash,
            }
        )

        staging = PROBE_CHECKPOINT_DIR / f".{SOURCE_CHECKPOINT_ID}.fork"
        target = PROBE_CHECKPOINT_DIR / SOURCE_CHECKPOINT_ID
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
                "files": [
                    {"name": "trainer_state.pkl", "sha256": sha256_path(trainer_state_path)},
                    {
                        "name": "checkpoint.json",
                        "sha256": sha256_path(staging / "checkpoint.json"),
                    },
                ]
            },
        )
        verify_local_manifest(staging)
        os.replace(staging, target)
    finally:
        del state
        release_host_memory()


@function(
    name="small-llm-decay-probe-12500-prepare",
    image=base.CPU_IMAGE,
    cpu=4,
    memory="16Gi",
    timeout=-1,
    retries=1,
    secrets=base.SECRETS,
    volumes=[base.RUN_VOLUME, base.CACHE_VOLUME],
    env=base.RUNTIME_ENV,
)
def prepare_probe_remote(source_commit: str) -> dict[str, object]:
    """Fork the exact checkpoint and stage its dataset window before GPU allocation."""

    del source_commit  # recorded by the GPU-side publication manifest, not scientific state
    base._install_beam_imports()
    import runtime as runtime_base
    from dataset.incremental_stage import stage_incremental_window_when_ready
    from dataset.qualification import get_profile
    from dataset.src.hf_bucket_shards import HuggingFaceBucketShardStore
    from dataset.src.joint_checkpoint import verify_local_manifest
    from rolling_dataset import hf_dataset_bucket_id

    PROBE_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    latest_id, latest_step = runtime_base._latest_checkpoint(PROBE_CHECKPOINT_DIR)
    if latest_step > PROBE_FINAL_STEP:
        raise RuntimeError(
            f"probe checkpoint step {latest_step} exceeds frozen final step {PROBE_FINAL_STEP}"
        )
    if latest_step == 0:
        source = base.RUN_ROOT / SOURCE_RUN_ID / "checkpoints" / SOURCE_CHECKPOINT_ID
        if not source.is_dir():
            raise RuntimeError(
                f"exact source checkpoint {SOURCE_CHECKPOINT_ID} is not present in {source.parent}; "
                "do not stop/remove the Beam run Volume until this checkpoint is verified"
            )
        verify_local_manifest(source)
    if latest_step == PROBE_FINAL_STEP:
        return {
            "status": "training_complete",
            "run_id": PROBE_RUN_ID,
            "checkpoint_id": latest_id,
            "completed_steps": latest_step,
            "final_step": PROBE_FINAL_STEP,
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
    dataset = base.CACHE_ROOT / "decay-probes" / PROBE_RUN_ID / f"from-{required_block:08d}"
    staged = stage_incremental_window_when_ready(
        store=store,
        run_id=DATASET_RUN_ID,
        destination=dataset,
        start_block_id=required_block,
    )
    if staged.get("status") != "ready":
        raise RuntimeError(f"probe dataset stage did not become ready: {staged}")

    if latest_step == 0:
        _fork_source_checkpoint(dataset=dataset)
        latest_id, latest_step = runtime_base._latest_checkpoint(PROBE_CHECKPOINT_DIR)
        if latest_id != SOURCE_CHECKPOINT_ID or latest_step != SOURCE_STEP:
            raise RuntimeError("probe fork did not install the exact step-12500 checkpoint")
        _write_json(PROBE_CONTRACT_PATH, _probe_contract())
    elif not PROBE_CONTRACT_PATH.is_file():
        raise RuntimeError("resumed decay probe is missing decay_probe_contract.json")
    elif json.loads(PROBE_CONTRACT_PATH.read_text(encoding="utf-8")) != _probe_contract():
        raise RuntimeError("decay probe contract drifted between segments")

    remaining = PROBE_FINAL_STEP - latest_step
    return {
        "status": "ready",
        "run_id": PROBE_RUN_ID,
        "resume_checkpoint_id": latest_id,
        "completed_steps": latest_step,
        "remaining_steps": remaining,
        "final_step": PROBE_FINAL_STEP,
        "required_block": latest_step,
        "dataset_dir": str(dataset),
        "dataset_bucket_id": bucket_id,
        "lr_now": _expected_lr(latest_step * TARGETS_PER_FULL_BLOCK),
        "lr_final": PEAK_LR * MINIMUM_LR_RATIO,
    }


@function(
    name="small-llm-decay-probe-12500-visibility",
    image=base.CPU_IMAGE,
    cpu=2,
    memory="8Gi",
    timeout=180,
    retries=1,
    secrets=base.SECRETS,
    volumes=[base.CACHE_VOLUME],
    env=base.RUNTIME_ENV,
)
def verify_probe_stage_remote(dataset_dir: str, required_block: int) -> dict[str, object]:
    """Verify the CPU stage from a fresh container before renting the GPU."""

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


def _train_probe_impl(
    source_commit: str,
    dataset_dir: str,
    resume_checkpoint_id: str,
    remaining_steps: int,
) -> dict[str, object]:
    repo = base._install_beam_imports()
    import runtime as runtime_base
    from profiles import resolve_presets

    if remaining_steps <= 0:
        raise RuntimeError("decay probe GPU was allocated with no remaining work")
    dataset = Path(dataset_dir).resolve(strict=True)
    model_preset, token_preset = resolve_presets("100M", "10B")
    if token_preset.dataset_profile != DATASET_PROFILE:
        raise RuntimeError("100M/10B profile drifted")

    os.environ["SMALL_LLM_MODAL_ROLLING_DATASET"] = "1"
    os.environ["SMALL_LLM_DATASET_SHARD_BUCKET"] = (
        os.environ.get("SMALL_LLM_HF_DATASET_BUCKET_ID", "").strip()
        or f"{runtime_base._hf_model_repo_id()}-datasets"
    )
    os.environ["SMALL_LLM_DATASET_SHARD_RUN_ID"] = DATASET_RUN_ID
    os.environ["SMALL_LLM_DATASET_SHARD_PREFETCH"] = "1"

    bucket_id = runtime_base._hf_checkpoint_bucket_id()
    remote_manifest = PROBE_RUN_DIR / "hf_checkpoint_transport.json"
    runtime_base._write_hf_transport_manifest(
        remote_manifest,
        run_id=PROBE_RUN_ID,
        dataset=dataset,
        dataset_profile=DATASET_PROFILE,
        source_commit=source_commit,
        microbatch_size=MICROBATCH_SIZE,
        resume_parent_source_commit=None,
        bucket_id=bucket_id,
    )
    probe_plan: dict[str, Any] = {
        "trainer": {
            "warmup_tokens": MATCHED_WARMUP_TOKENS,
            "stable_tokens": MATCHED_STABLE_TOKENS,
            "decay_tokens": MATCHED_DECAY_TOKENS,
            "validation_blocks": 16,
        }
    }
    environment = runtime_base._gpu_environment()
    gpu_tag = str(environment["name"]).lower().replace(" ", "-")
    command = runtime_base._trainer_command(
        model=model_preset,
        tokens=token_preset,
        dataset=dataset,
        plan=probe_plan,
        checkpoint_dir=PROBE_CHECKPOINT_DIR,
        steps=remaining_steps,
        microbatch=MICROBATCH_SIZE,
        precision="fp16",
        wandb_run_id=PROBE_RUN_ID,
        gpu_tag=gpu_tag,
        online=True,
        resume=resume_checkpoint_id,
        remote_manifest=remote_manifest,
        remote_bucket_id=bucket_id,
    )
    # This is a new W&B branch even though it resumes model/optimizer state from
    # a local checkpoint. `must` would incorrectly require a pre-existing W&B run.
    _replace_option(command, "--wandb-resume", "allow")
    _replace_option(
        command,
        "--wandb-run-name",
        "100M/10B step-12500 matched 2B cooldown probe",
    )

    log_path = PROBE_RUN_DIR / "evidence" / f"train-from-{resume_checkpoint_id}.log"
    started = __import__("time").perf_counter()
    runtime_base._run(command, cwd=repo, log_path=log_path)
    base.NOOP_VOLUME.commit()

    final_id, final_step = runtime_base._latest_checkpoint(PROBE_CHECKPOINT_DIR)
    expected = int(resume_checkpoint_id.removeprefix("step-")) + remaining_steps
    if final_step != expected:
        raise RuntimeError(f"probe durable checkpoint step {final_step} != expected {expected}")
    return {
        "status": "complete" if final_step == PROBE_FINAL_STEP else "segment_complete",
        "run_id": PROBE_RUN_ID,
        "checkpoint_id": final_id,
        "completed_steps": final_step,
        "final_step": PROBE_FINAL_STEP,
        "elapsed_seconds": __import__("time").perf_counter() - started,
        "source_checkpoint_id": SOURCE_CHECKPOINT_ID,
        "matched_schedule": {
            "warmup_tokens": MATCHED_WARMUP_TOKENS,
            "stable_tokens": MATCHED_STABLE_TOKENS,
            "decay_tokens": MATCHED_DECAY_TOKENS,
            "minimum_lr_ratio": MINIMUM_LR_RATIO,
        },
    }


@function(
    name="small-llm-decay-probe-12500-rtx5090",
    gpu="RTX5090",
    image=base.BLACKWELL_IMAGE,
    **base._GPU_FUNCTION_KWARGS,
)
def train_probe_rtx5090_remote(
    source_commit: str,
    dataset_dir: str,
    resume_checkpoint_id: str,
    remaining_steps: int,
) -> dict[str, object]:
    return _train_probe_impl(source_commit, dataset_dir, resume_checkpoint_id, remaining_steps)


@function(
    name="small-llm-decay-probe-12500-rtx4090",
    gpu="RTX4090",
    image=base.LEGACY_SERVERLESS_IMAGE,
    **base._GPU_FUNCTION_KWARGS,
)
def train_probe_rtx4090_remote(
    source_commit: str,
    dataset_dir: str,
    resume_checkpoint_id: str,
    remaining_steps: int,
) -> dict[str, object]:
    return _train_probe_impl(source_commit, dataset_dir, resume_checkpoint_id, remaining_steps)


@function(
    name="small-llm-decay-probe-12500-a10g",
    gpu="A10G",
    image=base.LEGACY_SERVERLESS_IMAGE,
    **base._GPU_FUNCTION_KWARGS,
)
def train_probe_a10g_remote(
    source_commit: str,
    dataset_dir: str,
    resume_checkpoint_id: str,
    remaining_steps: int,
) -> dict[str, object]:
    return _train_probe_impl(source_commit, dataset_dir, resume_checkpoint_id, remaining_steps)


GPU_FUNCTIONS = {
    "RTX5090": train_probe_rtx5090_remote,
    "RTX4090": train_probe_rtx4090_remote,
    "A10G": train_probe_a10g_remote,
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
        "action": "matched_2b_decay_probe",
        "runtime": "beam/decay_probe_12500.py",
        "gpu": args.gpu,
        "source_run_id": SOURCE_RUN_ID,
        "source_checkpoint_id": SOURCE_CHECKPOINT_ID,
        "probe_run_id": PROBE_RUN_ID,
        "source_step": SOURCE_STEP,
        "source_consumed_tokens": SOURCE_EXPECTED_TOKENS,
        "lr_at_source_step": _expected_lr(SOURCE_EXPECTED_TOKENS),
        "lr_final": PEAK_LR * MINIMUM_LR_RATIO,
        "additional_steps": PROBE_ADDITIONAL_STEPS,
        "additional_full_block_targets": PROBE_ADDITIONAL_STEPS * TARGETS_PER_FULL_BLOCK,
        "final_step": PROBE_FINAL_STEP,
        "schedule": "exact historical modal-2b-b64 WSD cooldown",
        "source_commit": source_commit,
    }
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    if args.dry_run:
        return 0

    base._stage("decay_probe_prepare_start", checkpoint=SOURCE_CHECKPOINT_ID)
    prepared = base._require_remote_mapping(
        prepare_probe_remote.remote(source_commit),
        label="decay probe prepare",
    )
    base._stage("decay_probe_prepare_complete", **prepared)
    if prepared.get("status") == "training_complete":
        print(json.dumps(prepared, indent=2, sort_keys=True), flush=True)
        return 0
    if prepared.get("status") != "ready":
        raise RuntimeError("decay probe CPU preparation did not authorize GPU dispatch")

    dataset_dir = prepared.get("dataset_dir")
    required_block = prepared.get("required_block")
    resume_checkpoint_id = prepared.get("resume_checkpoint_id")
    remaining_steps = prepared.get("remaining_steps")
    if not isinstance(dataset_dir, str) or not dataset_dir:
        raise RuntimeError("decay probe preparation returned no dataset directory")
    if isinstance(required_block, bool) or not isinstance(required_block, int):
        raise RuntimeError("decay probe preparation returned no required block")
    if not isinstance(resume_checkpoint_id, str):
        raise RuntimeError("decay probe preparation returned no resume checkpoint")
    if isinstance(remaining_steps, bool) or not isinstance(remaining_steps, int) or remaining_steps <= 0:
        raise RuntimeError("decay probe preparation returned invalid remaining steps")

    base._stage("decay_probe_visibility_start", required_block=required_block)
    visible = base._require_remote_mapping(
        verify_probe_stage_remote.remote(dataset_dir, required_block),
        label="decay probe dataset visibility",
    )
    base._stage("decay_probe_visibility_complete", **visible)

    base._stage(
        "decay_probe_dispatch",
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
        label="decay probe training",
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
