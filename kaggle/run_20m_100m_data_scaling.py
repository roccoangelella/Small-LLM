#!/usr/bin/env python3
"""One-entry Kaggle launcher for the 20M-model/100M-token scaling run.

The 100M finite dataset is built and mirrored separately, then attached to the
Kaggle notebook as immutable schema-v2 shards. This launcher never downloads or
rebuilds the source corpus. It:

1. checks the controlling repository and creates a clean detached launch worktree;
2. identifies exactly one attached dataset matching the fixed 100M profile;
3. performs a literal full shard scan and derives the exact one-pass WSD plan;
4. compares microbatch 1 with the fixed candidate microbatch 4 on the same first
   eight blocks from the same seed and fails closed unless microbatch 4 is safe
   and measurably faster;
5. starts the complete from-scratch run with the frozen 20M model and one
   optimizer update per 16-sequence block.

Run from a clean clone:

    %cd /kaggle/working/Small-LLM
    !git pull --ff-only
    !python kaggle/run_20m_100m_data_scaling.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import run_20m_one_click as common

REPO = Path(__file__).resolve().parents[1]
# Updated in a follow-up commit after all implementation files are present.
DEFAULT_COMMIT = "__PIN_20M_100M_LAUNCH_COMMIT__"
DATASET_RUN_ID = "20m-100m-dataset-001"
PROFILE = "20m-100m-data-scaling-v1"
ROOT = common.WORK / "small-llm-20m-100m-data-scaling"
WORKTREE = ROOT / "launch-worktree"
EVIDENCE = ROOT / ("evidence-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
CHECKPOINTS = common.WORK / "checkpoints-20m-100m-data-scaling"
SUMMARY = common.WORK / "small_llm_20m_100m_data_scaling_summary.json"
WANDB_RUN_ID = "20m-100m-data-001"
MICROBATCH_BASELINE = 1
MICROBATCH_CANDIDATE = 4
MICROBATCH_PROBE_STEPS = 8
MICROBATCH_WARMUP_DISCARD = 2
MICROBATCH_MIN_SPEEDUP = 1.05
MICROBATCH_MAX_LOSS_DELTA = 0.05
MAX_RESERVED_MEMORY_FRACTION = 0.90
LOCAL_CHECKPOINT_EVERY = 250
EVALUATION_EVERY = 500
REMOTE_PUBLISH_EVERY = 500


class LaunchFailure(common.GateFailure):
    pass


def read_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as error:
        raise LaunchFailure(f"Cannot read {label}: {path}") from error
    if not isinstance(payload, Mapping):
        raise LaunchFailure(f"{label} must contain a JSON object: {path}")
    return dict(payload)


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the fixed 20M-model/100M-token data-scaling run."
    )
    parser.add_argument(
        "--launch-commit",
        default=os.environ.get("SMALL_LLM_100M_LAUNCH_COMMIT", DEFAULT_COMMIT),
    )
    parser.add_argument("--dataset-dir", type=Path)
    return parser.parse_args(argv)


def repo_head(commit: str) -> str:
    try:
        top = Path(
            subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"], cwd=REPO, text=True
            ).strip()
        ).resolve()
    except (OSError, subprocess.CalledProcessError) as error:
        raise LaunchFailure("Run from the cloned Small-LLM repository") from error
    if top != REPO.resolve():
        raise LaunchFailure(f"Repository root mismatch: {top}")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPO,
        text=True,
    ).strip()
    if dirty:
        raise LaunchFailure("The controlling clone has tracked modifications")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise LaunchFailure(
            "The 100M launch commit is not pinned. Pull the commit that freezes "
            "this launcher or set SMALL_LLM_100M_LAUNCH_COMMIT explicitly."
        )
    if subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=REPO,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode:
        raise LaunchFailure(f"Frozen launch commit {commit} is missing")
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()


def prepare_worktree(commit: str) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    if WORKTREE.exists():
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(WORKTREE)],
            cwd=REPO,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if WORKTREE.exists():
            shutil.rmtree(WORKTREE)
    subprocess.run(["git", "worktree", "prune"], cwd=REPO, check=False)
    common.run(
        ["git", "worktree", "add", "--detach", str(WORKTREE), commit],
        name="git-launch-worktree",
        evidence=EVIDENCE,
        cwd=REPO,
    )
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=WORKTREE, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=WORKTREE, text=True
    ).strip()
    if actual != commit or dirty:
        raise LaunchFailure(
            f"Frozen worktree mismatch: actual={actual}, dirty={bool(dirty)}"
        )


def dataset_profile_matches(root: Path) -> tuple[bool, dict[str, Any]]:
    manifest_path = root / "manifest.json"
    drive_path = root / "drive_manifest.json"
    row: dict[str, Any] = {
        "root": str(root),
        "manifest": manifest_path.is_file(),
        "drive_manifest": drive_path.is_file(),
        "train": (root / "train").is_dir(),
        "validation": (root / "validation").is_dir(),
    }
    if not all(row[key] for key in ("manifest", "drive_manifest", "train", "validation")):
        return False, row
    try:
        manifest = read_object(manifest_path, label="dataset manifest")
    except LaunchFailure as error:
        row["error"] = str(error)
        return False, row
    production = manifest.get("production")
    row.update(
        schema_version=manifest.get("schema_version"),
        context_length=manifest.get("context_length"),
        sequences_per_block=manifest.get("sequences_per_block"),
        target_shard_bytes=manifest.get("target_shard_bytes"),
        run_id=production.get("run_id") if isinstance(production, Mapping) else None,
    )
    expected_top = {
        "schema_version": 2,
        "sequence_format": "context_plus_one",
        "context_length": 2_048,
        "stored_tokens_per_sequence": 2_049,
        "sequences_per_block": 16,
        "target_shard_bytes": 8 * 1024 * 1024,
    }
    expected_production = {
        "run_id": DATASET_RUN_ID,
        "target_source_tokens": 100_000_000,
        "minimum_source_tokens": 90_000_000,
        "maximum_source_tokens": 110_000_000,
        "checkpoint_source_tokens": 2_000_000,
        "target_reached": True,
        "remote_required": True,
    }
    matched = all(manifest.get(key) == value for key, value in expected_top.items())
    matched = matched and isinstance(production, Mapping)
    if isinstance(production, Mapping):
        matched = matched and all(
            production.get(key) == value for key, value in expected_production.items()
        )
    if matched:
        row["manifest_sha256"] = common.sha256(manifest_path)
        row["drive_manifest_sha256"] = common.sha256(drive_path)
    return bool(matched), row


def find_dataset(explicit: Path | None) -> tuple[Path, list[dict[str, Any]]]:
    if explicit is not None:
        roots = [explicit.resolve()]
    else:
        roots = sorted({path.parent for path in common.INPUT.rglob("manifest.json")})
    inspected: list[dict[str, Any]] = []
    matches: list[Path] = []
    for root in roots:
        matched, row = dataset_profile_matches(root)
        inspected.append(row)
        if matched:
            matches.append(root)
    if len(matches) != 1:
        raise LaunchFailure(
            "Expected exactly one attached 100M qualification dataset; "
            f"found {len(matches)}. Inspected:\n" + json.dumps(inspected, indent=2)
        )
    return matches[0], inspected


def validate_plan(path: Path) -> dict[str, Any]:
    plan = read_object(path, label="100M qualification plan")
    trainer = plan.get("trainer")
    identity = plan.get("identity")
    train = plan.get("train")
    validation = plan.get("validation")
    if plan.get("version") != 1 or plan.get("qualification_profile") != PROFILE:
        raise LaunchFailure("Generated plan has the wrong profile identity")
    if plan.get("context_length") != 2_048 or plan.get("sequences_per_block") != 16:
        raise LaunchFailure("Generated plan changed the frozen block geometry")
    if plan.get("target_shard_bytes") != 8 * 1024 * 1024:
        raise LaunchFailure("Generated plan changed the frozen shard size")
    accepted = plan.get("accepted_source_tokens")
    if not isinstance(accepted, int) or not 100_000_000 <= accepted <= 110_000_000:
        raise LaunchFailure("Generated plan has an invalid accepted-source-token count")
    if not all(isinstance(value, Mapping) for value in (trainer, identity, train, validation)):
        raise LaunchFailure("Generated plan is missing required sections")
    assert isinstance(trainer, Mapping)
    assert isinstance(identity, Mapping)
    assert isinstance(train, Mapping)
    assert isinstance(validation, Mapping)
    steps = trainer.get("steps")
    warmup = trainer.get("warmup_updates")
    stable = trainer.get("stable_updates")
    decay = trainer.get("decay_updates")
    if not all(isinstance(value, int) and value > 0 for value in (steps, warmup, stable, decay)):
        raise LaunchFailure("Generated plan has invalid WSD update counts")
    assert isinstance(steps, int)
    assert isinstance(warmup, int)
    assert isinstance(stable, int)
    assert isinstance(decay, int)
    if warmup + stable + decay != steps:
        raise LaunchFailure("Generated WSD phases do not sum to the planned updates")
    if trainer.get("passes") != 1 or trainer.get("schedule") != "wsd":
        raise LaunchFailure("Generated plan changed the one-pass WSD policy")
    if trainer.get("full_block_target_tokens") != 32_768:
        raise LaunchFailure("Generated plan changed the effective optimizer batch")
    if trainer.get("minimum_lr_ratio") != 0.1:
        raise LaunchFailure("Generated plan changed the minimum LR ratio")
    block_ids = train.get("block_ids")
    if not isinstance(block_ids, list) or block_ids != list(range(steps)):
        raise LaunchFailure("Training block IDs are not exactly contiguous")
    validation_blocks = trainer.get("validation_blocks")
    if not isinstance(validation_blocks, int) or validation_blocks <= 0:
        raise LaunchFailure("Generated plan has no validation blocks")
    if identity.get("drive_run_id") != DATASET_RUN_ID:
        raise LaunchFailure("Generated plan has the wrong Drive run ID")
    for key in ("manifest_sha256", "drive_manifest_sha256"):
        value = identity.get(key)
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise LaunchFailure(f"Generated plan has an invalid {key}")
    return plan


def trainer_command(
    uv: str,
    dataset: Path,
    plan: Mapping[str, Any],
    *,
    checkpoint_dir: Path,
    steps: int,
    microbatch_size: int,
    wandb: bool,
    entity: str | None = None,
) -> list[str]:
    trainer = plan["trainer"]
    assert isinstance(trainer, Mapping)
    command = [
        uv,
        "run",
        "--python",
        "3.13",
        "--extra",
        "model",
        "--with",
        "wandb==0.26.1",
        "--with-requirements",
        "dataset/requirements-remote.txt",
        "python",
        "-m",
        "trainer",
        "--dataset-dir",
        str(dataset),
        "--dataset-manifest",
        str(dataset / "manifest.json"),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--steps",
        str(steps),
        "--sequences-per-block",
        "16",
        "--model-size",
        "smoke",
        "--architecture",
        "gdn2_hybrid",
        "--gdn-chunk-size",
        "32",
        "--initialization",
        "normal",
        "--optimizer",
        "hybrid_muon_adamw",
        "--device",
        "cuda",
        "--precision",
        "fp16",
        "--microbatch-size",
        str(microbatch_size),
        "--learning-rate",
        "3e-4",
        "--weight-decay",
        "0.1",
        "--muon-momentum",
        "0.95",
        "--muon-lr-multiplier",
        "1.0",
        "--muon-update-rms",
        "0.18",
        "--muon-weight-decay",
        "0.1",
        "--max-grad-norm",
        "1.0",
        "--schedule",
        "wsd",
        "--warmup-tokens",
        str(trainer["warmup_tokens"]),
        "--stable-tokens",
        str(trainer["stable_tokens"]),
        "--decay-tokens",
        str(trainer["decay_tokens"]),
        "--minimum-lr-ratio",
        "0.1",
        "--seed",
        "17",
    ]
    if not wandb:
        return command + [
            "--checkpoint-every-steps",
            "0",
            "--evaluation-every-steps",
            "0",
            "--validation-blocks",
            "0",
            "--remote-publish-every-steps",
            "0",
            "--wandb-mode",
            "disabled",
        ]
    command += [
        "--checkpoint-every-steps",
        str(LOCAL_CHECKPOINT_EVERY),
        "--evaluation-every-steps",
        str(EVALUATION_EVERY),
        "--validation-blocks",
        str(trainer["validation_blocks"]),
        "--remote-publish-every-steps",
        str(REMOTE_PUBLISH_EVERY),
        "--remote-drive-manifest",
        str(dataset / "drive_manifest.json"),
        "--remote-token-env",
        "HF_TOKEN",
        "--wandb-mode",
        "online",
        "--wandb-project",
        "Small-LLM",
        "--wandb-run-id",
        WANDB_RUN_ID,
        "--wandb-run-name",
        "20M model on 100M tokens",
        "--wandb-tags",
        "20m",
        "100m-tokens",
        "t4",
        "data-scaling",
        "microbatch-4",
        "one-pass",
    ]
    if entity:
        command += ["--wandb-entity", entity]
    return command


def parse_training_metrics(path: Path, *, expected_steps: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
        except (TypeError, ValueError):
            continue
        if not isinstance(item, Mapping):
            continue
        if all(key in item for key in ("step", "block_id", "loss", "tokens_per_second")):
            rows.append(dict(item))
    if len(rows) != expected_steps:
        raise LaunchFailure(
            f"Expected {expected_steps} training metrics in {path}, found {len(rows)}"
        )
    expected = list(range(expected_steps))
    if [row.get("block_id") for row in rows] != expected:
        raise LaunchFailure(f"Microbatch probe block IDs are not {expected}")
    if [row.get("step") for row in rows] != list(range(1, expected_steps + 1)):
        raise LaunchFailure("Microbatch probe optimizer steps are not contiguous")
    for row in rows:
        for key in ("loss", "gradient_norm", "tokens_per_second", "grad_scaler_scale"):
            value = row.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise LaunchFailure(f"Microbatch probe emitted non-finite {key}: {row}")
        if row.get("overflow_retries") != 0 or row.get("overflow_events_total") != 0:
            raise LaunchFailure(f"Microbatch probe encountered an FP16 overflow: {row}")
    return rows


def gpu_total_memory_bytes() -> int:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).splitlines()[0]
    mib = int(output.strip().replace(",", ""))
    return mib * 1024 * 1024


def summarize_probe(rows: list[dict[str, Any]]) -> dict[str, Any]:
    measured = rows[MICROBATCH_WARMUP_DISCARD:]
    throughputs = [float(row["tokens_per_second"]) for row in measured]
    return {
        "steps": len(rows),
        "measured_steps": len(measured),
        "median_tokens_per_second": statistics.median(throughputs),
        "mean_tokens_per_second": statistics.fmean(throughputs),
        "minimum_tokens_per_second": min(throughputs),
        "maximum_peak_allocated_bytes": max(int(row["peak_memory_bytes"]) for row in rows),
        "maximum_peak_reserved_bytes": max(
            int(row["peak_reserved_memory_bytes"]) for row in rows
        ),
        "final_loss": float(rows[-1]["loss"]),
        "maximum_gradient_norm": max(float(row["gradient_norm"]) for row in rows),
        "clipped_updates": sum(bool(row.get("gradient_clipped")) for row in rows),
        "grad_scaler_minimum": min(float(row["grad_scaler_scale"]) for row in rows),
        "grad_scaler_maximum": max(float(row["grad_scaler_scale"]) for row in rows),
    }


def qualify_microbatch(
    uv: str,
    dataset: Path,
    plan: Mapping[str, Any],
    env: Mapping[str, str],
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    histories: dict[int, list[dict[str, Any]]] = {}
    for microbatch in (MICROBATCH_BASELINE, MICROBATCH_CANDIDATE):
        checkpoint_dir = EVIDENCE / f"microbatch-{microbatch}-checkpoints"
        stage = common.run(
            trainer_command(
                uv,
                dataset,
                plan,
                checkpoint_dir=checkpoint_dir,
                steps=MICROBATCH_PROBE_STEPS,
                microbatch_size=microbatch,
                wandb=False,
            ),
            name=f"microbatch-{microbatch}-probe",
            evidence=EVIDENCE,
            cwd=WORKTREE,
            env=env,
        )
        rows = parse_training_metrics(
            Path(stage["log"]), expected_steps=MICROBATCH_PROBE_STEPS
        )
        histories[microbatch] = rows
        results[str(microbatch)] = summarize_probe(rows)
        shutil.rmtree(checkpoint_dir, ignore_errors=True)

    baseline = results[str(MICROBATCH_BASELINE)]
    candidate = results[str(MICROBATCH_CANDIDATE)]
    speedup = (
        float(candidate["median_tokens_per_second"])
        / float(baseline["median_tokens_per_second"])
    )
    loss_deltas = [
        abs(float(left["loss"]) - float(right["loss"]))
        for left, right in zip(
            histories[MICROBATCH_BASELINE], histories[MICROBATCH_CANDIDATE]
        )
    ]
    total_memory = gpu_total_memory_bytes()
    reserved_fraction = float(candidate["maximum_peak_reserved_bytes"]) / total_memory
    verdict = {
        "baseline_microbatch": MICROBATCH_BASELINE,
        "selected_microbatch": MICROBATCH_CANDIDATE,
        "minimum_required_speedup": MICROBATCH_MIN_SPEEDUP,
        "observed_median_speedup": speedup,
        "maximum_step_loss_delta": max(loss_deltas),
        "maximum_allowed_step_loss_delta": MICROBATCH_MAX_LOSS_DELTA,
        "candidate_reserved_memory_fraction": reserved_fraction,
        "maximum_allowed_reserved_memory_fraction": MAX_RESERVED_MEMORY_FRACTION,
        "gpu_total_memory_bytes": total_memory,
        "results": results,
    }
    if speedup < MICROBATCH_MIN_SPEEDUP:
        raise LaunchFailure(
            "Microbatch 4 did not deliver the required 5% median throughput gain: "
            + json.dumps(verdict, indent=2)
        )
    if max(loss_deltas) > MICROBATCH_MAX_LOSS_DELTA:
        raise LaunchFailure(
            "Microbatch 4 changed the eight-step loss trajectory beyond the "
            "bounded execution-grouping tolerance: " + json.dumps(verdict, indent=2)
        )
    if reserved_fraction > MAX_RESERVED_MEMORY_FRACTION:
        raise LaunchFailure(
            "Microbatch 4 leaves insufficient T4 memory headroom: "
            + json.dumps(verdict, indent=2)
        )
    verdict["status"] = "passed"
    return verdict


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_arguments(argv)
    state: dict[str, Any] = {
        "schema_version": 1,
        "started_utc": common.now(),
        "status": "initializing",
        "authorization": "20m_100m_data_scaling_setup",
        "launch_commit": args.launch_commit,
        "dataset_run_id": DATASET_RUN_ID,
        "profile": PROFILE,
        "model_parameters": 20_637_592,
        "effective_sequences_per_update": 16,
        "selected_microbatch_size": MICROBATCH_CANDIDATE,
        "checkpoint_dir": str(CHECKPOINTS),
        "evidence_dir": str(EVIDENCE),
        "wandb_run_id": WANDB_RUN_ID,
        "cadence": {
            "local_checkpoint_every_steps": LOCAL_CHECKPOINT_EVERY,
            "evaluation_every_steps": EVALUATION_EVERY,
            "remote_publish_every_steps": REMOTE_PUBLISH_EVERY,
            "reason": "10x step scaling preserves approximately the 10M run's relative dataset-progress cadence",
        },
    }
    try:
        if CHECKPOINTS.exists():
            raise LaunchFailure(
                f"{CHECKPOINTS} already exists; this entry point starts from scratch"
            )
        environment = common.check_environment()
        controller_head = repo_head(args.launch_commit)
        wandb_key = common.secret("WANDB_API_KEY")
        hf_token = common.secret("HF_TOKEN")
        hf_repo = common.secret("SMALL_LLM_HF_REPO_ID")
        entity = common.secret("WANDB_ENTITY", required=False)
        assert wandb_key and hf_token and hf_repo

        EVIDENCE.mkdir(parents=True, exist_ok=False)
        state.update(
            status="preparing",
            environment=environment,
            controller_head=controller_head,
            remote_checkpoint_repo=hf_repo,
        )
        common.write_json(SUMMARY, state)
        prepare_worktree(args.launch_commit)
        dataset, inspected = find_dataset(args.dataset_dir)
        state.update(dataset_dir=str(dataset), datasets_inspected=inspected)

        common.run(
            [sys.executable, "-m", "pip", "install", "-q", "uv"],
            name="install-uv",
            evidence=EVIDENCE,
            cwd=WORKTREE,
        )
        uv = common.find_uv()
        common.run(
            [uv, "python", "install", "3.13"],
            name="install-python-3.13",
            evidence=EVIDENCE,
            cwd=WORKTREE,
        )
        common.run(
            [uv, "run", "--python", "3.13", "python", "--version"],
            name="python-version",
            evidence=EVIDENCE,
            cwd=WORKTREE,
        )
        base = [uv, "run", "--python", "3.13"]
        common.run(
            base
            + [
                "--with-requirements",
                "dataset/requirements-remote.txt",
                "python",
                "-m",
                "dataset.main",
                "verify",
                "--output-dir",
                str(dataset),
                "--full-scan",
            ],
            name="dataset-full-scan",
            evidence=EVIDENCE,
            cwd=WORKTREE,
        )
        plan_path = EVIDENCE / "qualification_plan.json"
        common.run(
            base
            + [
                "python",
                "-m",
                "dataset.qualification_100m_report",
                "--dataset-dir",
                str(dataset),
                "--drive-manifest",
                str(dataset / "drive_manifest.json"),
                "--output",
                str(plan_path),
            ],
            name="qualification-plan",
            evidence=EVIDENCE,
            cwd=WORKTREE,
        )
        plan = validate_plan(plan_path)
        state["plan"] = plan
        common.write_json(SUMMARY, state)

        runtime_env = {
            "WANDB_API_KEY": wandb_key,
            "HF_TOKEN": hf_token,
            "SMALL_LLM_HF_REPO_ID": hf_repo,
            "UV_LINK_MODE": "copy",
            "PYTHONUNBUFFERED": "1",
        }
        state["microbatch_qualification"] = qualify_microbatch(
            uv, dataset, plan, runtime_env
        )
        common.write_json(SUMMARY, state)

        trainer = plan["trainer"]
        assert isinstance(trainer, Mapping)
        command = trainer_command(
            uv,
            dataset,
            plan,
            checkpoint_dir=CHECKPOINTS,
            steps=int(trainer["steps"]),
            microbatch_size=MICROBATCH_CANDIDATE,
            wandb=True,
            entity=entity,
        )
        state.update(
            status="running",
            trainer_started_utc=common.now(),
            trainer_command=command,
        )
        common.write_json(SUMMARY, state)
        print(
            "\nSTARTING 20M-MODEL / 100M-TOKEN DATA-SCALING RUN\n"
            f"dataset: {dataset}\n"
            f"updates: {trainer['steps']}\n"
            f"microbatch: {MICROBATCH_CANDIDATE}\n"
            f"W&B: {WANDB_RUN_ID}\n",
            flush=True,
        )
        common.run(
            command,
            name=f"trainer-{trainer['steps']}-updates",
            evidence=EVIDENCE,
            cwd=WORKTREE,
            env=runtime_env,
        )
        state.update(status="completed", completed_utc=common.now())
        common.write_json(SUMMARY, state)
        print(f"100M-token run completed. Summary: {SUMMARY}")
        return 0
    except KeyboardInterrupt:
        state.update(status="interrupted", finished_utc=common.now())
        common.write_json(SUMMARY, state)
        return 130
    except Exception as error:  # noqa: BLE001 - launcher failure boundary
        state.update(
            status="failed",
            finished_utc=common.now(),
            error=f"{type(error).__name__}: {error}",
        )
        common.write_json(SUMMARY, state)
        print(f"LAUNCH FAILED CLOSED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
