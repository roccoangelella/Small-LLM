#!/usr/bin/env python3
"""Resume 100M/10B from exact step 15,500 with aggressive WSqD-style decay.

Usage from repository root:

    python beam/aggressive_wsqd_10b_from_15500.py --dry-run
    python beam/aggressive_wsqd_10b_from_15500.py --gpu RTX4090

The schedule has three phases after the exact uncooled step-15,500 fork:
1. cosine settle from 3e-4 to 1.5e-4 over ~300M targets;
2. inverse-square-root decay from that settled anchor to 9.6B targets;
3. linear terminal cooldown over ~400M targets to 1.5e-5 at exact 10B.

Only the LR scheduler changes. Model, optimizer, scaler, RNG, data cursor,
architecture, precision, optimizer recipe, and corpus order are preserved.
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

SOURCE_RUN_ID = "100m-10b-data-001"
SOURCE_STEP = 15_500
SOURCE_CHECKPOINT_ID = f"step-{SOURCE_STEP:08d}"
RUN_ID = "100m-10b-aggressive-wsqd-from-step15500"
DATASET_PROFILE = "modal-10b-b64"
DATASET_RUN_ID = "modal-10b-b64-dataset-001"
MICROBATCH_SIZE = 4
TARGETS_PER_FULL_BLOCK = 64 * 2048
PEAK_LR = 3e-4
SETTLE_LR_RATIO = 0.5
MINIMUM_LR_RATIO = 0.05
TOTAL_TARGETS = 10_000_007_168
FINAL_STEP = 76_294
SOURCE_EXPECTED_TOKENS = SOURCE_STEP * TARGETS_PER_FULL_BLOCK

REQUESTED_SETTLE_TOKENS = 300_000_000
SETTLE_STEPS = math.ceil(REQUESTED_SETTLE_TOKENS / TARGETS_PER_FULL_BLOCK)
SETTLE_TOKENS = SETTLE_STEPS * TARGETS_PER_FULL_BLOCK
SETTLE_END_STEP = SOURCE_STEP + SETTLE_STEPS
SETTLE_END_TOKENS = SOURCE_EXPECTED_TOKENS + SETTLE_TOKENS
SETTLE_LR = PEAK_LR * SETTLE_LR_RATIO

COOLDOWN_STEPS = 3_052
COOLDOWN_TOKENS = COOLDOWN_STEPS * TARGETS_PER_FULL_BLOCK
COOLDOWN_START_STEP = FINAL_STEP - COOLDOWN_STEPS
COOLDOWN_START_TOKENS = COOLDOWN_START_STEP * TARGETS_PER_FULL_BLOCK
ADDITIONAL_STEPS = FINAL_STEP - SOURCE_STEP

RUN_DIR = base.RUN_ROOT / RUN_ID
CHECKPOINT_DIR = RUN_DIR / "checkpoints"
CONTRACT_PATH = RUN_DIR / "aggressive_wsqd_10b_contract.json"

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

    base = SETTLE_LR * math.sqrt(SETTLE_END_TOKENS / tokens)
    if tokens <= COOLDOWN_START_TOKENS:
        return base

    cooldown_base = SETTLE_LR * math.sqrt(
        SETTLE_END_TOKENS / COOLDOWN_START_TOKENS
    )
    terminal = PEAK_LR * MINIMUM_LR_RATIO
    progress = min(
        1.0,
        max(0.0, (tokens - COOLDOWN_START_TOKENS) / COOLDOWN_TOKENS),
    )
    return terminal + (cooldown_base - terminal) * (1.0 - progress)


def _contract() -> dict[str, object]:
    return {
        "version": 1,
        "kind": "step15500_aggressive_wsqd_settle_invsqrt_linear_cooldown",
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
        "cooldown_start_step": COOLDOWN_START_STEP,
        "cooldown_start_tokens": COOLDOWN_START_TOKENS,
        "lr_at_cooldown_start": _expected_lr(COOLDOWN_START_TOKENS),
        "cooldown_steps": COOLDOWN_STEPS,
        "decay_tokens": COOLDOWN_TOKENS,
        "minimum_lr_ratio": MINIMUM_LR_RATIO,
        "final_lr": PEAK_LR * MINIMUM_LR_RATIO,
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
        )
        patched_scheduler = dict(scheduler)
        patched_scheduler["config"] = dict(patched_config)
        patched_scheduler["committed_tokens"] = SOURCE_EXPECTED_TOKENS
        patched_scheduler["last_lr"] = PEAK_LR
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
                "files": [
                    {"name": "trainer_state.pkl", "sha256": sha256_path(trainer_state_path)},
                    {"name": "checkpoint.json", "sha256": sha256_path(staging / "checkpoint.json")},
                ]
            },
        )
        verify_local_manifest(staging)
        os.replace(staging, target)
    finally:
        del state
        release_host_memory()


@function(
    name="small-llm-aggressive-wsqd-10b-15500-prepare",
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
    from rolling_dataset import hf_dataset_bucket_id

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    latest_id, latest_step = runtime_base._latest_checkpoint(CHECKPOINT_DIR)
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
    dataset = base.CACHE_ROOT / "aggressive-wsqd-10b" / RUN_ID / f"from-{required_block:08d}"
    staged = stage_incremental_window_when_ready(
        store=store,
        run_id=DATASET_RUN_ID,
        destination=dataset,
        start_block_id=required_block,
    )
    if staged.get("status") != "ready":
        raise RuntimeError(f"dataset stage did not become ready: {staged}")

    if latest_step == 0:
        _fork_source_checkpoint(dataset=dataset)
        latest_id, latest_step = runtime_base._latest_checkpoint(CHECKPOINT_DIR)
        if latest_id != SOURCE_CHECKPOINT_ID or latest_step != SOURCE_STEP:
            raise RuntimeError("fork did not install exact step-15500 checkpoint")
        _write_json(CONTRACT_PATH, _contract())
    elif not CONTRACT_PATH.is_file():
        raise RuntimeError("resumed continuation is missing aggressive_wsqd_10b_contract.json")
    elif json.loads(CONTRACT_PATH.read_text(encoding="utf-8")) != _contract():
        raise RuntimeError("aggressive WSqD contract drifted between segments")

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
        "lr_now": _expected_lr(latest_step * TARGETS_PER_FULL_BLOCK),
        "lr_at_settle_end": SETTLE_LR,
        "lr_at_cooldown_start": _expected_lr(COOLDOWN_START_TOKENS),
        "lr_final": PEAK_LR * MINIMUM_LR_RATIO,
    }


@function(
    name="small-llm-aggressive-wsqd-10b-15500-visibility",
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

    dispatched_resume_checkpoint_id = resume_checkpoint_id
    try:
        dispatched_step = int(dispatched_resume_checkpoint_id.removeprefix("step-"))
    except ValueError as error:
        raise RuntimeError("GPU dispatch carried an invalid resume checkpoint") from error
    latest_id, latest_step = runtime_base._latest_checkpoint(CHECKPOINT_DIR)
    if latest_id is None or latest_step < SOURCE_STEP:
        raise RuntimeError("GPU worker found no valid aggressive continuation checkpoint")
    if latest_step > FINAL_STEP:
        raise RuntimeError(f"continuation checkpoint step {latest_step} exceeds {FINAL_STEP}")
    if latest_step < dispatched_step:
        raise RuntimeError(
            "GPU worker checkpoint regressed below its CPU-authorized resume point: "
            f"{latest_step} < {dispatched_step}"
        )
    resume_checkpoint_id = latest_id
    remaining_steps = FINAL_STEP - latest_step
    if remaining_steps <= 0:
        raise RuntimeError("GPU was allocated with no remaining work")
    print(
        json.dumps(
            {
                "beam_stage": "aggressive_wsqd_10b_gpu_resume_resolved",
                "dispatched_resume_checkpoint_id": dispatched_resume_checkpoint_id,
                "resume_checkpoint_id": resume_checkpoint_id,
                "completed_steps": latest_step,
                "remaining_steps": remaining_steps,
            },
            sort_keys=True,
        ),
        flush=True,
    )
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

    checkpoint_bucket_id = runtime_base._hf_checkpoint_bucket_id()
    remote_manifest = RUN_DIR / "hf_checkpoint_transport.json"
    runtime_base._write_hf_transport_manifest(
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
        remote_bucket_id=checkpoint_bucket_id,
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
    ]
    _replace_option(
        command,
        "--wandb-resume",
        "allow" if resume_checkpoint_id == SOURCE_CHECKPOINT_ID else "must",
    )
    _replace_option(
        command,
        "--wandb-run-name",
        "100M/10B aggressive WSqD continuation from step 15500",
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
    name="small-llm-aggressive-wsqd-10b-15500-rtx5090",
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
    name="small-llm-aggressive-wsqd-10b-15500-rtx4090",
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
    name="small-llm-aggressive-wsqd-10b-15500-a10g",
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
        "action": "step15500_aggressive_wsqd_10b_continuation",
        "runtime": "beam/aggressive_wsqd_10b_from_15500.py",
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
        "cooldown_start_step": COOLDOWN_START_STEP,
        "cooldown_start_tokens": COOLDOWN_START_TOKENS,
        "lr_at_cooldown_start": _expected_lr(COOLDOWN_START_TOKENS),
        "cooldown_steps": COOLDOWN_STEPS,
        "cooldown_tokens": COOLDOWN_TOKENS,
        "lr_final": PEAK_LR * MINIMUM_LR_RATIO,
        "additional_steps": ADDITIONAL_STEPS,
        "final_step": FINAL_STEP,
        "final_targets": TOTAL_TARGETS,
        "schedule": "300M cosine settle + inverse-sqrt base + 400M linear terminal cooldown",
        "source_commit": source_commit,
    }
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    if args.dry_run:
        return 0

    base._stage("aggressive_wsqd_10b_prepare_start", checkpoint=SOURCE_CHECKPOINT_ID)
    prepared = base._require_remote_mapping(
        prepare_remote.remote(source_commit),
        label="aggressive WSqD 10B prepare",
    )
    base._stage("aggressive_wsqd_10b_prepare_complete", **prepared)
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

    base._stage("aggressive_wsqd_10b_visibility_start", required_block=required_block)
    visible = base._require_remote_mapping(
        verify_stage_remote.remote(dataset_dir, required_block),
        label="aggressive WSqD dataset visibility",
    )
    base._stage("aggressive_wsqd_10b_visibility_complete", **visible)

    base._stage(
        "aggressive_wsqd_10b_dispatch",
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
        label="aggressive WSqD 10B training",
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
