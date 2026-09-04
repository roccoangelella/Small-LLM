#!/usr/bin/env python3
"""Canonical 100M/10B pretraining probe launcher for Kaggle dual-T4.

This is the single home for short, W&B-visible 100M/10B pretraining probes.
The active experiment set tests whether the apparent 10B validation plateau was
caused by the terminal learning-rate decay rather than by model/data saturation:

* ``hold-1e-5``: 3,000 updates at constant LR 1e-5.
* ``hold-2e-5``: 3,000 updates at constant LR 2e-5.

Both branches use the same source checkpoint and the same following corpus
blocks. The preferred anchor is the historical step-00071750 checkpoint used by
``100m-10b-probe-a-reset-low-from-step71750``. The launcher first attempts to
restore that exact artifact from the dedicated 100M/10B best-model repository.
If it is genuinely absent, it may fall back to the repository's current
``best_model.json`` checkpoint, but it never falls back to rolling ``latest``.
The actual source checkpoint is encoded in every W&B run ID.

The completed constant-1e-4 reset run remains available only as a legacy
reproduction target and is not part of the default active probe set. The old
3e-4 reset branch is retired: the completed 1e-4 reset already showed that a
large late-training reheat is harmful.

Probe checkpoints are local-only. Hugging Face is read for the source checkpoint
and rolling 10B dataset shards, but probe checkpoints/best models are never
published.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

SRC_DIR = Path(__file__).resolve().parent
KAGGLE_DIR = SRC_DIR.parent
ROOT = KAGGLE_DIR.parent
for candidate in (ROOT, SRC_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import deep_decay_10b_from_15500 as deep_decay

_impl = deep_decay._impl

HF_HUB_VERSION = getattr(deep_decay, "HF_HUB_VERSION", "1.5.0")
BASE_HF_REPO_ID = (
    os.environ.get("SMALL_LLM_100M10B_PROBES_BASE_REPO_ID", "").strip()
    or os.environ.get("SMALL_LLM_PROBE_A_BASE_REPO_ID", "").strip()
    or "roccoangelella/small-llm-100m-qualification"
)
PREFERRED_SOURCE_CHECKPOINT_ID = "step-00071750"
DEFAULT_PROBE_STEPS = 3_000
DEFAULT_EVAL_EVERY_STEPS = 250
DEFAULT_VALIDATION_BLOCKS = 16
PROBE_NAME = "100m-10b-probes"
WORK_ROOT = _impl.WORK_ROOT
PROBE_ROOT = WORK_ROOT / "runs" / PROBE_NAME
_POST_SAVE_METADATA = frozenset(
    {"local_manifest.json", "drive_manifest.json", "checkpoint_manifest.json"}
)

WANDB_IDENTITY_ENV = (
    "WANDB_RUN_ID",
    "WANDB_ID",
    "WANDB_NAME",
    "WANDB_RESUME",
    "WANDB_RUN_GROUP",
    "WANDB_JOB_TYPE",
    "WANDB_DIR",
)


@dataclass(frozen=True, slots=True)
class ProbeBranch:
    selector: str
    label: str
    learning_rate: float
    run_slug: str
    legacy_run_id: bool = False


ACTIVE_BRANCHES = (
    ProbeBranch("hold-1e-5", "Hold 1e-5", 1e-5, "hold-1e-5"),
    ProbeBranch("hold-2e-5", "Hold 2e-5", 2e-5, "hold-2e-5"),
)
LEGACY_RESET_BRANCH = ProbeBranch(
    "legacy-reset-1e-4",
    "Legacy reset 1e-4",
    1e-4,
    "reset-low",
    legacy_run_id=True,
)


def _force_probe_hf_identity() -> None:
    """Prevent stale Kaggle environment variables from selecting another model."""

    checkpoint_bucket = (
        os.environ.get("SMALL_LLM_100M10B_PROBES_CHECKPOINT_BUCKET_ID", "").strip()
        or os.environ.get("SMALL_LLM_PROBE_A_CHECKPOINT_BUCKET_ID", "").strip()
        or f"{BASE_HF_REPO_ID}-checkpoints"
    )
    dataset_bucket = (
        os.environ.get("SMALL_LLM_100M10B_PROBES_DATASET_BUCKET_ID", "").strip()
        or os.environ.get("SMALL_LLM_PROBE_A_DATASET_BUCKET_ID", "").strip()
        or f"{BASE_HF_REPO_ID}-datasets"
    )
    desired = {
        "SMALL_LLM_HF_REPO_ID": BASE_HF_REPO_ID,
        "SMALL_LLM_HF_CHECKPOINT_BUCKET_ID": checkpoint_bucket,
        "SMALL_LLM_HF_DATASET_BUCKET_ID": dataset_bucket,
    }
    previous = {key: os.environ.get(key) for key in desired}
    for key, value in desired.items():
        os.environ[key] = value

    overridden = {
        key: {"previous": previous[key], "current": value}
        for key, value in desired.items()
        if previous[key] and previous[key] != value
    }
    if overridden:
        print(json.dumps({"100m_10b_probe_hf_identity_override": overridden}, sort_keys=True))


def _ensure_probe_hf_runtime(argv: Sequence[str]) -> None:
    """Re-exec this file with the private HF Hub runtime if Storage Buckets are absent."""

    if deep_decay._hf_bucket_api_available():
        return

    runtime = WORK_ROOT / ".runtime" / f"huggingface-hub-{HF_HUB_VERSION}"
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
        f"[100m-10b-probes] restarting with private huggingface_hub=={HF_HUB_VERSION}",
        flush=True,
    )
    os.execve(
        sys.executable,
        [sys.executable, str(Path(__file__).resolve()), *list(argv)],
        env,
    )


def _source_repo_id() -> str:
    return (
        os.environ.get("SMALL_LLM_100M10B_PROBES_SOURCE_REPO_ID", "").strip()
        or os.environ.get("SMALL_LLM_PROBE_A_SOURCE_REPO_ID", "").strip()
        or f"{BASE_HF_REPO_ID}-best-{_impl.RUN_ID}"
    )


def _checkpoint_prefix(checkpoint_id: str) -> str:
    return f"models/{_impl.RUN_ID}/{checkpoint_id}"


def _load_best_marker(*, repo_id: str, token: str | None) -> dict[str, object]:
    from huggingface_hub import hf_hub_download

    marker_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="model",
            filename="best_model.json",
            token=token,
            force_download=True,
        )
    )
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if not isinstance(marker, Mapping):
        raise RuntimeError("100M/10B best_model.json is not a JSON object")
    if marker.get("role") != "small-llm-dedicated-best-model":
        raise RuntimeError(f"{repo_id!r} is not a Small-LLM dedicated best-model repo")
    if marker.get("run_id") != _impl.RUN_ID:
        raise RuntimeError(
            f"best-model repo belongs to {marker.get('run_id')!r}, not {_impl.RUN_ID!r}"
        )
    checkpoint_id = marker.get("checkpoint_id")
    if not isinstance(checkpoint_id, str) or not checkpoint_id.startswith("step-"):
        raise RuntimeError("best_model.json lacks a valid checkpoint_id")
    expected_path = _checkpoint_prefix(checkpoint_id)
    artifact_path = marker.get("artifact_path")
    if artifact_path is not None and artifact_path != expected_path:
        raise RuntimeError("best_model.json artifact_path disagrees with checkpoint_id")
    return dict(marker)


def _ensure_local_manifest(root: Path) -> bool:
    """Reconstruct local_manifest.json for older dedicated-best snapshots if needed."""

    manifest_path = root / "local_manifest.json"
    if manifest_path.is_symlink():
        raise RuntimeError("probe source local_manifest.json is still a symlink")
    if manifest_path.is_file():
        return False

    names: list[str] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(
                f"probe source still contains a symlink after materialization: {path.relative_to(root)}"
            )
        if not path.is_file():
            continue
        name = path.relative_to(root).as_posix()
        if name in _POST_SAVE_METADATA:
            continue
        names.append(name)

    required = {"trainer_state.pkl", "checkpoint.json"}
    missing = sorted(required - set(names))
    if missing:
        raise RuntimeError(f"cannot reconstruct source local_manifest.json; missing {missing}")

    from dataset.src.remote import sha256_path

    payload = {
        "version": 1,
        "files": [
            {
                "name": name,
                "sha256": sha256_path(root / name),
                "byte_size": (root / name).stat().st_size,
            }
            for name in names
        ],
    }
    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, manifest_path)
    return True


def _materialize_snapshot_tree(source: Path, staging: Path) -> None:
    shutil.rmtree(staging, ignore_errors=True)
    shutil.copytree(source, staging, symlinks=False)
    for path in staging.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(
                f"materialized probe checkpoint still contains symlink {path.relative_to(staging)}"
            )


def _restore_candidate_checkpoint(
    *,
    repo_id: str,
    checkpoint_id: str,
    token: str | None,
) -> tuple[Path | None, bool]:
    """Restore one candidate; return (local path, rebuilt_manifest). None means absent."""

    from dataset.src.joint_checkpoint import verify_local_manifest
    from huggingface_hub import snapshot_download

    target = _impl.CHECKPOINT_DIR / checkpoint_id
    if target.exists():
        verify_local_manifest(target)
        return target, False

    prefix = _checkpoint_prefix(checkpoint_id)
    snapshot_root = Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            token=token,
            allow_patterns=[f"{prefix}/*"],
            force_download=True,
        )
    )
    source = snapshot_root / prefix
    if not source.is_dir():
        return None, False

    _impl.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    staging = _impl.CHECKPOINT_DIR / f".{checkpoint_id}.100m-10b-probe-source"
    _materialize_snapshot_tree(source, staging)
    rebuilt = _ensure_local_manifest(staging)
    verify_local_manifest(staging)
    shutil.rmtree(target, ignore_errors=True)
    os.replace(staging, target)
    verify_local_manifest(target)
    return target, rebuilt


def _restore_source_checkpoint(runtime_base: Any) -> dict[str, object]:
    """Prefer exact step 71,750; otherwise use the dedicated repo's current best."""

    token = runtime_base._hf_token()
    repo_id = _source_repo_id()
    marker = _load_best_marker(repo_id=repo_id, token=token)
    current_best = str(marker["checkpoint_id"])

    candidates = [PREFERRED_SOURCE_CHECKPOINT_ID]
    if current_best != PREFERRED_SOURCE_CHECKPOINT_ID:
        candidates.append(current_best)

    errors: dict[str, str] = {}
    for checkpoint_id in candidates:
        try:
            local_path, rebuilt = _restore_candidate_checkpoint(
                repo_id=repo_id,
                checkpoint_id=checkpoint_id,
                token=token,
            )
        except Exception as error:
            errors[checkpoint_id] = f"{type(error).__name__}: {error}"
            if checkpoint_id == current_best:
                raise
            continue
        if local_path is None:
            errors[checkpoint_id] = "artifact path absent from dedicated best-model repository"
            continue

        source_kind = (
            "preferred_step_71750_dedicated_best_repo"
            if checkpoint_id == PREFERRED_SOURCE_CHECKPOINT_ID
            else "current_best_fallback"
        )
        step = int(_impl._checkpoint_step(checkpoint_id))
        return {
            "checkpoint_id": checkpoint_id,
            "step": step,
            "source_kind": source_kind,
            "repo_id": repo_id,
            "path_in_repo": _checkpoint_prefix(checkpoint_id),
            "best_marker_checkpoint_id": current_best,
            "validation_loss": (
                marker.get("validation_loss") if checkpoint_id == current_best else None
            ),
            "local_manifest_rebuilt": rebuilt,
            "preferred_restore_errors": errors,
        }

    raise RuntimeError(
        "No usable 100M/10B probe source checkpoint. Tried exact step-00071750 and "
        f"current best {current_best}; errors={errors}"
    )


def _set_rolling_dataset_env(runtime_base: Any) -> None:
    dataset_bucket_id = (
        os.environ.get("SMALL_LLM_HF_DATASET_BUCKET_ID", "").strip()
        or f"{runtime_base._hf_model_repo_id()}-datasets"
    )
    os.environ["SMALL_LLM_MODAL_ROLLING_DATASET"] = "1"
    os.environ["SMALL_LLM_DATASET_SHARD_BUCKET"] = dataset_bucket_id
    os.environ["SMALL_LLM_DATASET_SHARD_RUN_ID"] = _impl.DATASET_RUN_ID
    os.environ["SMALL_LLM_DATASET_SHARD_PREFETCH"] = "1"
    os.environ.setdefault("WANDB_DISABLE_CODE", "true")


def _prepare_source_checkpoint(runtime_base: Any) -> tuple[str, int, Path, dict[str, object]]:
    restored = _restore_source_checkpoint(runtime_base)
    checkpoint_id = str(restored["checkpoint_id"])
    step = int(restored["step"])

    dataset = _impl._stage_dataset(
        runtime_base,
        start_block_id=min(step, _impl.FINAL_STEP - 1),
    )
    migrated = deep_decay._install_execution_migration(
        checkpoint_id=checkpoint_id,
        dataset=dataset,
    )
    verified_step = _impl._verify_deep_decay_checkpoint(runtime_base, checkpoint_id)
    if verified_step != step:
        raise RuntimeError("verified probe source step disagrees with restored checkpoint")

    restored = dict(restored)
    restored.update(
        {
            "migrated_to_kaggle_execution": bool(migrated),
            "dataset": str(dataset),
        }
    )
    print(json.dumps({"100m_10b_probe_source": restored}, sort_keys=True), flush=True)
    return checkpoint_id, step, dataset, restored


def _replace_option(command: list[str], option: str, value: str) -> None:
    try:
        index = command.index(option)
    except ValueError as error:
        raise RuntimeError(f"trainer command lacks {option}") from error
    if index + 1 >= len(command):
        raise RuntimeError(f"trainer command option {option} has no value")
    command[index + 1] = value


def _append_option(command: list[str], option: str, value: str) -> None:
    if option in command:
        _replace_option(command, option, value)
    else:
        command += [option, value]


def _remove_option_with_value(command: list[str], option: str) -> None:
    while option in command:
        index = command.index(option)
        del command[index : index + 2]


def _remove_flag(command: list[str], option: str) -> None:
    while option in command:
        command.remove(option)


def _remove_option_values_until_next(command: list[str], option: str) -> None:
    while option in command:
        index = command.index(option)
        stop = index + 1
        while stop < len(command) and not command[stop].startswith("--"):
            stop += 1
        del command[index:stop]


def _all_option_values(command: Sequence[str], option: str) -> list[str]:
    return [
        command[index + 1]
        for index, token in enumerate(command[:-1])
        if token == option
    ]


def _branch_run_id(branch: ProbeBranch, source_step: int) -> str:
    if branch.legacy_run_id:
        return f"100m-10b-probe-a-reset-low-from-step{source_step}"
    return f"100m-10b-probe-{branch.run_slug}-from-step{source_step}"


def _branch_run_name(branch: ProbeBranch, source_step: int) -> str:
    return (
        f"100M/10B probe {branch.label} from step {source_step} "
        f"(constant LR {branch.learning_rate:g}, no HF publish)"
    )


def _branch_wandb_dir(branch_run_dir: Path) -> Path:
    return branch_run_dir / "wandb"


def _wandb_env_prefix(branch: ProbeBranch, source_step: int, branch_run_dir: Path) -> list[str]:
    run_id = _branch_run_id(branch, source_step)
    env_binary = shutil.which("env") or "env"
    return [
        env_binary,
        *(item for name in WANDB_IDENTITY_ENV for item in ("-u", name)),
        f"WANDB_RUN_ID={run_id}",
        f"WANDB_ID={run_id}",
        f"WANDB_NAME={_branch_run_name(branch, source_step)}",
        "WANDB_RESUME=allow",
        "WANDB_MODE=online",
        f"WANDB_RUN_GROUP={PROBE_NAME}",
        "WANDB_JOB_TYPE=pretraining-lr-probe",
        f"WANDB_DIR={_branch_wandb_dir(branch_run_dir)}",
    ]


def _with_branch_wandb_environment(
    command: Sequence[str],
    *,
    branch: ProbeBranch,
    source_step: int,
    branch_run_dir: Path,
) -> list[str]:
    return [
        *_wandb_env_prefix(branch, source_step, branch_run_dir),
        *list(command),
    ]


def _dataset_configuration_hash(dataset: Path) -> str:
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    production = manifest.get("production") if isinstance(manifest, Mapping) else None
    value = production.get("configuration_hash") if isinstance(production, Mapping) else None
    if not isinstance(value, str) or not value:
        raise RuntimeError("staged 10B dataset lacks production.configuration_hash")
    return value


def _probe_config(
    original_config: Mapping[str, object],
    *,
    branch: ProbeBranch,
    eval_every_steps: int,
) -> dict[str, object]:
    from trainer.config import TrainerConfig

    config = dict(original_config)
    config.update(
        {
            "learning_rate": float(branch.learning_rate),
            "schedule": "constant",
            "warmup_tokens": 0,
            "stable_tokens": 0,
            "decay_tokens": 0,
            "minimum_lr_ratio": 1.0,
            "schedule_anchor_tokens": 0,
            "cooldown_start_tokens": 0,
            "settle_tokens": 0,
            "settle_lr_ratio": 1.0,
            "base_power": 0.5,
            "checkpoint_every_steps": 0,
            "evaluation_every_steps": int(eval_every_steps),
        }
    )
    return TrainerConfig(**config).as_dict()


def _fork_branch_checkpoint(
    *,
    runtime_base: Any,
    source_checkpoint_id: str,
    source_step: int,
    branch: ProbeBranch,
    branch_checkpoint_dir: Path,
    dataset: Path,
    eval_every_steps: int,
) -> None:
    import torch
    from dataset.src.joint_checkpoint import verify_local_manifest
    from dataset.src.remote import sha256_path
    from trainer.identity import canonical_hash
    from trainer.state import load_trainer_state_file, release_host_memory

    source_root = _impl.CHECKPOINT_DIR / source_checkpoint_id
    if _impl._verify_deep_decay_checkpoint(runtime_base, source_checkpoint_id) != source_step:
        raise RuntimeError("source checkpoint step changed during probe fork")
    verify_local_manifest(source_root)

    branch_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    target_root = branch_checkpoint_dir / source_checkpoint_id
    staging = branch_checkpoint_dir / f".{source_checkpoint_id}.{branch.selector}.fork"
    shutil.rmtree(staging, ignore_errors=True)

    state: dict[str, object] | None = load_trainer_state_file(
        source_root / "trainer_state.pkl",
        map_location="cpu",
    )
    try:
        if state.get("global_step") != source_step:
            raise RuntimeError("source trainer_state global_step disagrees with checkpoint ID")
        if state.get("consumed_tokens") != source_step * _impl.TARGETS_PER_FULL_BLOCK:
            raise RuntimeError("source trainer_state consumed_tokens disagrees with block cursor")
        config = state.get("config")
        scheduler = state.get("scheduler")
        model_config = state.get("model_config")
        if not isinstance(config, Mapping):
            raise RuntimeError("source trainer_state lacks config mapping")
        if not isinstance(scheduler, Mapping):
            raise RuntimeError("source trainer_state lacks scheduler mapping")
        if not isinstance(model_config, Mapping):
            raise RuntimeError("source trainer_state lacks model_config mapping")
        if config.get("microbatch_size") != _impl.MICROBATCH_SIZE:
            raise RuntimeError("source checkpoint is not canonical Kaggle microbatch execution")
        if config.get("schedule") != "wsqd":
            raise RuntimeError("probe source must be a deep-decay WSqD checkpoint")
        if scheduler.get("committed_tokens") != state.get("consumed_tokens"):
            raise RuntimeError("source scheduler and consumed token counters disagree")

        patched_config = _probe_config(
            config,
            branch=branch,
            eval_every_steps=eval_every_steps,
        )
        patched_state = dict(state)
        patched_state["config"] = patched_config
        patched_state["scheduler"] = {
            "version": 1,
            "config": dict(patched_config),
            "committed_tokens": int(state["consumed_tokens"]),
            "last_lr": float(branch.learning_rate),
        }

        checkpoint_payload = json.loads(
            (source_root / "checkpoint.json").read_text(encoding="utf-8")
        )
        if not isinstance(checkpoint_payload, Mapping):
            raise RuntimeError("source checkpoint.json is not an object")
        configuration_hash = canonical_hash(
            {
                "version": 1,
                "model": dict(model_config),
                "trainer": dict(patched_config),
                "dataset_configuration_hash": _dataset_configuration_hash(dataset),
            }
        )

        shutil.copytree(source_root, staging)
        (staging / "checkpoint_manifest.json").unlink(missing_ok=True)
        (staging / "drive_manifest.json").unlink(missing_ok=True)
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
    finally:
        if state is not None:
            del state
        release_host_memory()

    shutil.rmtree(target_root, ignore_errors=True)
    os.replace(staging, target_root)
    verify_local_manifest(target_root)


def _assert_no_hf_publication(command: Sequence[str]) -> None:
    forbidden = {
        "--remote-drive-manifest",
        "--remote-checkpoint-bucket",
        "--remote-checkpoint-repo",
        "--best-model-repo",
    }
    present = sorted(flag for flag in forbidden if flag in command)
    if present:
        raise RuntimeError(f"HF publication flags are forbidden for 100M/10B probes: {present}")
    values = _all_option_values(command, "--remote-publish-every-steps")
    if values != ["0"]:
        raise RuntimeError(
            "100M/10B probes require exactly one --remote-publish-every-steps 0"
        )


def _assert_branch_wandb_identity(
    command: Sequence[str],
    *,
    branch: ProbeBranch,
    source_step: int,
) -> None:
    expected = _branch_run_id(branch, source_step)
    if _all_option_values(command, "--wandb-run-id") != [expected]:
        raise RuntimeError(f"probe branch must use W&B run ID {expected!r}")
    if _all_option_values(command, "--wandb-resume") != ["allow"]:
        raise RuntimeError("probe branch must use --wandb-resume allow")


def _build_branch_trainer_command(
    runtime_base: Any,
    *,
    branch: ProbeBranch,
    dataset: Path,
    branch_checkpoint_dir: Path,
    branch_run_dir: Path,
    source_checkpoint_id: str,
    source_step: int,
    steps: int,
    eval_every_steps: int,
    validation_blocks: int,
) -> list[str]:
    from profiles import resolve_presets  # type: ignore

    model_preset, token_preset = resolve_presets("100M", "10B")
    if token_preset.dataset_profile != _impl.DATASET_PROFILE:
        raise RuntimeError("100M/10B profile drifted away from the canonical dataset profile")

    command = runtime_base._trainer_command(
        model=model_preset,
        tokens=token_preset,
        dataset=dataset,
        plan={
            "trainer": {
                "warmup_tokens": 0,
                "stable_tokens": 0,
                "decay_tokens": 1,
                "validation_blocks": validation_blocks,
            }
        },
        checkpoint_dir=branch_checkpoint_dir,
        steps=steps,
        microbatch=_impl.MICROBATCH_SIZE,
        precision="fp16",
        wandb_run_id=_branch_run_id(branch, source_step),
        gpu_tag="dual-t4",
        online=False,
        resume=source_checkpoint_id,
    )

    _replace_option(command, "--learning-rate", f"{branch.learning_rate:g}")
    _replace_option(command, "--schedule", "constant")
    _replace_option(command, "--warmup-tokens", "0")
    _replace_option(command, "--stable-tokens", "0")
    _replace_option(command, "--decay-tokens", "0")
    _replace_option(command, "--minimum-lr-ratio", "1.0")
    _replace_option(command, "--checkpoint-every-steps", "0")
    _replace_option(command, "--evaluation-every-steps", str(eval_every_steps))
    _replace_option(command, "--validation-blocks", str(validation_blocks))
    _replace_option(command, "--remote-publish-every-steps", "0")
    _replace_option(command, "--wandb-mode", "online")

    for option in (
        "--remote-drive-manifest",
        "--remote-checkpoint-bucket",
        "--remote-checkpoint-repo",
        "--remote-token-env",
        "--best-model-repo",
    ):
        _remove_option_with_value(command, option)
    for flag in (
        "--remote-create-bucket",
        "--remote-create-repo",
        "--remote-rolling-latest-only",
        "--best-model-recreate",
    ):
        _remove_flag(command, flag)
    _remove_option_values_until_next(command, "--wandb-tags")

    run_id = _branch_run_id(branch, source_step)
    _append_option(command, "--wandb-project", "Small-LLM")
    _append_option(command, "--wandb-run-id", run_id)
    _append_option(command, "--wandb-run-name", _branch_run_name(branch, source_step))
    _append_option(command, "--wandb-resume", "allow")
    _append_option(command, "--wandb-dir", str(_branch_wandb_dir(branch_run_dir)))
    command += [
        "--wandb-tags",
        "100m",
        "10b-tokens",
        "kaggle",
        "dual-t4-ddp",
        "pretraining-probe",
        "lr-hold",
        branch.selector,
        "no-hf-publication",
        "exact-resume",
        "constant-lr",
    ]
    entity = os.environ.get("WANDB_ENTITY")
    if entity:
        _append_option(command, "--wandb-entity", entity)

    _assert_no_hf_publication(command)
    _assert_branch_wandb_identity(command, branch=branch, source_step=source_step)
    return command


def _run_branch(
    runtime_base: Any,
    *,
    branch: ProbeBranch,
    source_checkpoint_id: str,
    source_step: int,
    dataset: Path,
    steps: int,
    eval_every_steps: int,
    validation_blocks: int,
) -> dict[str, object]:
    run_id = _branch_run_id(branch, source_step)
    branch_run_dir = PROBE_ROOT / run_id
    branch_checkpoint_dir = branch_run_dir / "checkpoints"
    _fork_branch_checkpoint(
        runtime_base=runtime_base,
        source_checkpoint_id=source_checkpoint_id,
        source_step=source_step,
        branch=branch,
        branch_checkpoint_dir=branch_checkpoint_dir,
        dataset=dataset,
        eval_every_steps=eval_every_steps,
    )
    trainer_command = _build_branch_trainer_command(
        runtime_base,
        branch=branch,
        dataset=dataset,
        branch_checkpoint_dir=branch_checkpoint_dir,
        branch_run_dir=branch_run_dir,
        source_checkpoint_id=source_checkpoint_id,
        source_step=source_step,
        steps=steps,
        eval_every_steps=eval_every_steps,
        validation_blocks=validation_blocks,
    )
    command = _impl._dual_t4_command(trainer_command)
    command = _with_branch_wandb_environment(
        command,
        branch=branch,
        source_step=source_step,
        branch_run_dir=branch_run_dir,
    )
    _assert_no_hf_publication(command)

    log_path = branch_run_dir / "evidence" / f"{run_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    start_payload = {
        "run_id": run_id,
        "probe": branch.selector,
        "source_checkpoint_id": source_checkpoint_id,
        "source_step": source_step,
        "learning_rate": branch.learning_rate,
        "schedule": "constant",
        "steps": steps,
        "evaluation_every_steps": eval_every_steps,
        "validation_blocks": validation_blocks,
        "hf_publication": "disabled",
        "wandb": "online",
    }
    print(json.dumps({"100m_10b_probe_start": start_payload}, sort_keys=True), flush=True)
    runtime_base._run(command, cwd=ROOT, log_path=log_path)

    final_id, final_step = runtime_base._latest_checkpoint(branch_checkpoint_dir)
    expected_step = source_step + steps
    if final_id is None or final_step != expected_step:
        raise RuntimeError(
            f"probe {branch.selector} ended at {final_id}/{final_step}; expected {expected_step}"
        )
    result = {
        **start_payload,
        "latest_local_checkpoint_id": final_id,
        "latest_local_step": final_step,
        "elapsed_seconds": time.perf_counter() - started,
        "log_path": str(log_path),
    }
    print(json.dumps({"100m_10b_probe_complete": result}, sort_keys=True), flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probe",
        choices=("active", "hold-1e-5", "hold-2e-5", "legacy-reset-1e-4"),
        default="active",
        help="Probe to run. 'active' runs the two current LR-hold branches sequentially.",
    )
    parser.add_argument(
        "--max-steps-this-session",
        type=int,
        default=DEFAULT_PROBE_STEPS,
        help=f"Updates per selected branch. Default: {DEFAULT_PROBE_STEPS}.",
    )
    parser.add_argument(
        "--eval-every-steps",
        type=int,
        default=DEFAULT_EVAL_EVERY_STEPS,
        help=f"Validation cadence. Default: {DEFAULT_EVAL_EVERY_STEPS}.",
    )
    parser.add_argument(
        "--validation-blocks",
        type=int,
        default=DEFAULT_VALIDATION_BLOCKS,
        help=f"Frozen validation blocks per evaluation. Default: {DEFAULT_VALIDATION_BLOCKS}.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _branches(args: argparse.Namespace) -> list[ProbeBranch]:
    if args.probe == "active":
        return list(ACTIVE_BRANCHES)
    if args.probe == LEGACY_RESET_BRANCH.selector:
        return [LEGACY_RESET_BRANCH]
    return [branch for branch in ACTIVE_BRANCHES if branch.selector == args.probe]


def _validate_args(args: argparse.Namespace) -> None:
    if args.max_steps_this_session <= 0:
        raise SystemExit("--max-steps-this-session must be positive")
    if args.eval_every_steps <= 0:
        raise SystemExit("--eval-every-steps must be positive")
    if args.validation_blocks <= 0:
        raise SystemExit("--validation-blocks must be positive")


def _dry_run_payload(args: argparse.Namespace) -> dict[str, object]:
    return {
        "action": "100m_10b_pretraining_probes",
        "execution": "kaggle_dual_t4_ddp_block64",
        "control_run": _impl.RUN_ID,
        "source_policy": {
            "preferred_checkpoint_id": PREFERRED_SOURCE_CHECKPOINT_ID,
            "preferred_repo": _source_repo_id(),
            "fallback": "current best_model.json checkpoint in same dedicated best-model repo",
            "rolling_latest_fallback": False,
        },
        "historical_reference": {
            "wandb_run_id": "100m-10b-probe-a-reset-low-from-step71750",
            "learning_rate": 1e-4,
            "result": "completed; large late-training reset was harmful",
        },
        "hf_reads": ["dedicated best-model checkpoint", "rolling 10B dataset shards"],
        "hf_publication": "disabled",
        "remote_publish_every_steps": 0,
        "checkpoint_every_steps": 0,
        "max_steps_this_session": args.max_steps_this_session,
        "eval_every_steps": args.eval_every_steps,
        "validation_blocks": args.validation_blocks,
        "branches": [
            {
                "probe": branch.selector,
                "learning_rate": branch.learning_rate,
                "schedule": "constant",
                "wandb_run_id_template": (
                    "100m-10b-probe-a-reset-low-from-step<SOURCE_STEP>"
                    if branch.legacy_run_id
                    else f"100m-10b-probe-{branch.run_slug}-from-step<SOURCE_STEP>"
                ),
            }
            for branch in _branches(args)
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(args_list)
    _validate_args(args)
    _force_probe_hf_identity()

    if args.dry_run:
        print(json.dumps(_dry_run_payload(args), indent=2, sort_keys=True))
        return 0

    _ensure_probe_hf_runtime(args_list)
    # Never call the deep-decay shim's self-reexec helper from this launcher.
    deep_decay._ensure_host_hf_bucket_runtime = lambda argv: None

    runtime_base = _impl._beam_runtime()
    _set_rolling_dataset_env(runtime_base)
    source_checkpoint_id, source_step, dataset, source_metadata = _prepare_source_checkpoint(
        runtime_base
    )

    requested_steps = int(args.max_steps_this_session)
    if source_step + requested_steps > _impl.FINAL_STEP:
        raise RuntimeError(
            f"source step {source_step} has only {_impl.FINAL_STEP - source_step} canonical "
            f"10B updates remaining, fewer than requested {requested_steps}. Choose an earlier "
            "available source; the launcher will not silently shorten the experiment."
        )

    results = [
        _run_branch(
            runtime_base,
            branch=branch,
            source_checkpoint_id=source_checkpoint_id,
            source_step=source_step,
            dataset=dataset,
            steps=requested_steps,
            eval_every_steps=args.eval_every_steps,
            validation_blocks=args.validation_blocks,
        )
        for branch in _branches(args)
    ]
    print(
        json.dumps(
            {
                "100m_10b_probes_complete": {
                    "source": source_metadata,
                    "steps_per_branch": requested_steps,
                    "results": results,
                }
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
