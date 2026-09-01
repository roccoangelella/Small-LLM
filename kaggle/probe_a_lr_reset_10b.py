#!/usr/bin/env python3
"""Disposable Probe A LR-reset branches for the 100M/10B continuation on Kaggle 2xT4.

This launcher compares two optimizer/schedule reset hypotheses against the
already-running control branch:

* reset-low: resume the newest verified deep-decay checkpoint, keep optimizer
  moments/data cursor, reset to constant 1e-4.
* reset-mid: same source checkpoint, reset to constant 3e-4.

The probe is intentionally W&B-only for remote observability. It reads Hugging
Face only to hydrate the source checkpoint and rolling 10B dataset shards. It
must not publish checkpoints or best models to Hugging Face.

Usage from repository root on Kaggle:

    python kaggle/probe_a_lr_reset_10b.py --dry-run
    python kaggle/probe_a_lr_reset_10b.py
    python kaggle/probe_a_lr_reset_10b.py --branch reset-low --max-steps-this-session 2000
    python kaggle/probe_a_lr_reset_10b.py --reset-mid-lr 2e-4 --max-steps-this-session 5000
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
KAGGLE = ROOT / "kaggle"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(KAGGLE) not in sys.path:
    sys.path.insert(0, str(KAGGLE))

import dual_t4_runtime
import deep_decay_10b_from_15500 as deep_decay

_impl = deep_decay._impl  # canonical scientific continuation implementation

PROBE_NAME = "probe-a-lr-reset"
DEFAULT_PROBE_STEPS = 3_000
DEFAULT_EVAL_EVERY_STEPS = 250
DEFAULT_VALIDATION_BLOCKS = 16
RESET_LOW_LR = 1e-4
RESET_MID_LR = 3e-4

WORK_ROOT = _impl.WORK_ROOT
PROBE_ROOT = WORK_ROOT / "runs" / PROBE_NAME


@dataclass(frozen=True, slots=True)
class ProbeBranch:
    slug: str
    label: str
    learning_rate: float


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


def _checkpoint_step(checkpoint_id: str) -> int:
    return _impl._checkpoint_step(checkpoint_id)


def _dataset_configuration_hash(dataset: Path) -> str:
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    production = manifest.get("production") if isinstance(manifest, Mapping) else None
    value = production.get("configuration_hash") if isinstance(production, Mapping) else None
    if not isinstance(value, str) or not value:
        raise RuntimeError("staged 10B dataset lacks production.configuration_hash")
    return value


def _prepare_source_checkpoint(runtime_base: Any) -> tuple[str, int, Path]:
    """Restore the newest verified control checkpoint locally without publishing."""

    migrated = deep_decay._migrate_existing_deep_decay_checkpoint(runtime_base)
    checkpoint_id, step = runtime_base._latest_checkpoint(_impl.CHECKPOINT_DIR)
    if checkpoint_id is None:
        raise RuntimeError(
            "Probe A requires an existing 100M/10B deep-decay continuation checkpoint. "
            "The control branch is the source; this launcher will not fork the original "
            "step-15500 source into a new HF-published run."
        )
    verified_step = _impl._verify_deep_decay_checkpoint(runtime_base, checkpoint_id)
    if verified_step != step:
        raise RuntimeError("verified control checkpoint step disagrees with runtime cursor")

    stage_block = min(step, _impl.FINAL_STEP - 1)
    dataset = _impl._stage_dataset(runtime_base, start_block_id=stage_block)
    print(
        json.dumps(
            {
                "probe_a_source": {
                    "checkpoint_id": checkpoint_id,
                    "step": step,
                    "migrated_to_kaggle_execution": bool(migrated),
                    "dataset": str(dataset),
                }
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return checkpoint_id, step, dataset


def _probe_config(
    original_config: Mapping[str, object],
    *,
    branch: ProbeBranch,
    eval_every_steps: int,
) -> dict[str, object]:
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
    return config


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
    """Copy the control checkpoint into a disposable branch and patch only LR schedule."""

    import torch
    from dataset.src.joint_checkpoint import verify_local_manifest
    from dataset.src.remote import sha256_path
    from trainer.identity import canonical_hash
    from trainer.state import load_trainer_state_file, release_host_memory

    source_root = _impl.CHECKPOINT_DIR / source_checkpoint_id
    source_verified_step = _impl._verify_deep_decay_checkpoint(runtime_base, source_checkpoint_id)
    if source_verified_step != source_step:
        raise RuntimeError("source checkpoint step changed during probe fork")
    verify_local_manifest(source_root)

    branch_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    target_root = branch_checkpoint_dir / source_checkpoint_id
    staging = branch_checkpoint_dir / f".{source_checkpoint_id}.{branch.slug}.fork"
    shutil.rmtree(staging, ignore_errors=True)

    state: dict[str, object] | None = None
    state = load_trainer_state_file(source_root / "trainer_state.pkl", map_location="cpu")
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
            raise RuntimeError("source checkpoint is not canonical Kaggle microbatch-2 execution")
        if config.get("schedule") != "wsqd":
            raise RuntimeError("Probe A source must be the deep-decay WSqD control checkpoint")
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


def _branch_run_id(branch: ProbeBranch, source_step: int) -> str:
    return f"100m-10b-probe-a-{branch.slug}-from-step{source_step}"


def _branch_run_name(branch: ProbeBranch, source_step: int) -> str:
    return (
        f"100M/10B Probe A {branch.label} from step {source_step} "
        f"(constant LR {branch.learning_rate:g}, no HF publish)"
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


def _build_branch_trainer_command(
    runtime_base: Any,
    *,
    branch: ProbeBranch,
    dataset: Path,
    branch_checkpoint_dir: Path,
    source_checkpoint_id: str,
    source_step: int,
    steps: int,
    eval_every_steps: int,
    validation_blocks: int,
) -> list[str]:
    from profiles import resolve_presets  # type: ignore

    model_preset, token_preset = resolve_presets("100M", "10B")
    if token_preset.dataset_profile != _impl.DATASET_PROFILE:
        raise RuntimeError("100M/10B profile drifted away from modal-10b-b64")

    # We deliberately request runtime online=False to avoid the HF publication
    # manifest/bucket path, then opt W&B back in by explicit CLI flags below.
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

    _remove_option_with_value(command, "--remote-drive-manifest")
    _remove_option_with_value(command, "--remote-checkpoint-bucket")
    _remove_option_with_value(command, "--remote-token-env")
    _remove_option_with_value(command, "--best-model-repo")

    command += [
        "--wandb-project",
        "Small-LLM",
        "--wandb-run-id",
        _branch_run_id(branch, source_step),
        "--wandb-run-name",
        _branch_run_name(branch, source_step),
        "--wandb-tags",
        "100m",
        "10b-tokens",
        "kaggle",
        "dual-t4-ddp",
        "probe-a",
        "lr-reset",
        branch.slug,
        "no-hf-publication",
        "exact-resume",
        "constant-lr",
        "--wandb-resume",
        "allow",
    ]
    entity = os.environ.get("WANDB_ENTITY")
    if entity:
        command += ["--wandb-entity", entity]
    _assert_no_hf_publication(command)
    return command


def _assert_no_hf_publication(command: Sequence[str]) -> None:
    forbidden = {
        "--remote-drive-manifest",
        "--remote-checkpoint-bucket",
        "--remote-checkpoint-repo",
        "--best-model-repo",
    }
    present = sorted(flag for flag in forbidden if flag in command)
    if present:
        raise RuntimeError(f"HF publication flags are forbidden for Probe A: {present}")

    if "--remote-publish-every-steps" not in command:
        raise RuntimeError("Probe A trainer command must explicitly disable remote publication")
    index = command.index("--remote-publish-every-steps")
    try:
        value = int(command[index + 1])
    except (IndexError, ValueError) as error:
        raise RuntimeError("invalid Probe A remote publication flag") from error
    if value != 0:
        raise RuntimeError("Probe A must run with --remote-publish-every-steps 0")


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
        source_checkpoint_id=source_checkpoint_id,
        source_step=source_step,
        steps=steps,
        eval_every_steps=eval_every_steps,
        validation_blocks=validation_blocks,
    )
    command = _impl._dual_t4_command(trainer_command)
    _assert_no_hf_publication(command)

    log_path = branch_run_dir / "evidence" / f"{run_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    print(
        json.dumps(
            {
                "probe_a_branch_start": {
                    "run_id": run_id,
                    "branch": branch.slug,
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
            },
            sort_keys=True,
        ),
        flush=True,
    )
    runtime_base._run(command, cwd=ROOT, log_path=log_path)
    final_id, final_step = runtime_base._latest_checkpoint(branch_checkpoint_dir)
    expected_step = source_step + steps
    if final_id is None or final_step != expected_step:
        raise RuntimeError(
            f"Probe A branch {branch.slug} ended at {final_id}/{final_step}, "
            f"expected step {expected_step}"
        )
    result = {
        "run_id": run_id,
        "branch": branch.slug,
        "source_checkpoint_id": source_checkpoint_id,
        "source_step": source_step,
        "latest_local_checkpoint_id": final_id,
        "latest_local_step": final_step,
        "expected_step": expected_step,
        "learning_rate": branch.learning_rate,
        "schedule": "constant",
        "elapsed_seconds": time.perf_counter() - started,
        "hf_publication": "disabled",
        "log_path": str(log_path),
    }
    print(json.dumps({"probe_a_branch_complete": result}, sort_keys=True), flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--branch",
        choices=("all", "reset-low", "reset-mid"),
        default="all",
        help="Which Probe A branch to run. Default: both sequentially.",
    )
    parser.add_argument(
        "--max-steps-this-session",
        type=int,
        default=DEFAULT_PROBE_STEPS,
        help=f"Probe steps per selected branch. Default: {DEFAULT_PROBE_STEPS}.",
    )
    parser.add_argument(
        "--eval-every-steps",
        type=int,
        default=DEFAULT_EVAL_EVERY_STEPS,
        help=f"Validation cadence per branch. Default: {DEFAULT_EVAL_EVERY_STEPS}.",
    )
    parser.add_argument(
        "--validation-blocks",
        type=int,
        default=DEFAULT_VALIDATION_BLOCKS,
        help=f"Held-out validation blocks at each eval. Default: {DEFAULT_VALIDATION_BLOCKS}.",
    )
    parser.add_argument("--reset-low-lr", type=float, default=RESET_LOW_LR)
    parser.add_argument("--reset-mid-lr", type=float, default=RESET_MID_LR)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _branches(args: argparse.Namespace) -> list[ProbeBranch]:
    branches = [
        ProbeBranch("reset-low", "Reset-low", float(args.reset_low_lr)),
        ProbeBranch("reset-mid", "Reset-mid", float(args.reset_mid_lr)),
    ]
    if args.branch == "all":
        return branches
    return [branch for branch in branches if branch.slug == args.branch]


def _dry_run_payload(args: argparse.Namespace) -> dict[str, object]:
    return {
        "action": "probe_a_lr_reset_100m_10b",
        "execution": "kaggle_dual_t4_ddp_block64",
        "control_branch": _impl.RUN_ID,
        "source": "newest_verified_control_checkpoint_at_runtime",
        "hf_reads": ["control checkpoint restore", "rolling 10B dataset shards"],
        "hf_publication": "disabled",
        "remote_publish_every_steps": 0,
        "checkpoint_every_steps": 0,
        "final_checkpoint": "local_only_trainer_default",
        "max_steps_this_session": args.max_steps_this_session,
        "eval_every_steps": args.eval_every_steps,
        "validation_blocks": args.validation_blocks,
        "branches": [
            {
                "branch": branch.slug,
                "learning_rate": branch.learning_rate,
                "schedule": "constant",
                "wandb_run_id_template": f"100m-10b-probe-a-{branch.slug}-from-step<SOURCE_STEP>",
            }
            for branch in _branches(args)
        ],
    }


def _validate_args(args: argparse.Namespace) -> None:
    if args.max_steps_this_session <= 0:
        raise SystemExit("--max-steps-this-session must be positive")
    if args.eval_every_steps <= 0:
        raise SystemExit("--eval-every-steps must be positive")
    if args.validation_blocks <= 0:
        raise SystemExit("--validation-blocks must be positive")
    for name in ("reset_low_lr", "reset_mid_lr"):
        value = getattr(args, name)
        if value <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    if args.dry_run:
        print(json.dumps(_dry_run_payload(args), indent=2, sort_keys=True))
        return 0

    deep_decay._ensure_host_hf_bucket_runtime(list(sys.argv[1:] if argv is None else argv))
    runtime_base = _impl._beam_runtime()
    _set_rolling_dataset_env(runtime_base)
    source_checkpoint_id, source_step, dataset = _prepare_source_checkpoint(runtime_base)

    remaining = max(0, _impl.FINAL_STEP - source_step)
    if remaining <= 0:
        raise RuntimeError("control checkpoint is already at the frozen 10B final step")
    steps = min(args.max_steps_this_session, remaining)

    results = [
        _run_branch(
            runtime_base,
            branch=branch,
            source_checkpoint_id=source_checkpoint_id,
            source_step=source_step,
            dataset=dataset,
            steps=steps,
            eval_every_steps=args.eval_every_steps,
            validation_blocks=args.validation_blocks,
        )
        for branch in _branches(args)
    ]
    print(
        json.dumps(
            {
                "probe_a_complete": {
                    "source_checkpoint_id": source_checkpoint_id,
                    "source_step": source_step,
                    "steps_per_branch": steps,
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
