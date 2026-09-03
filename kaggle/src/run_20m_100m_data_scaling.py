#!/usr/bin/env python3
"""Fail-closed Kaggle launcher for the 20M-model/100M-token scaling run.

Run the same command in every Kaggle session. The first session qualifies
microbatch 4 against microbatch 1 and starts from seed 17. Later sessions restore
the latest verified remote checkpoint and continue the exact one-pass schedule.
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
DEFAULT_COMMIT = "__PIN_20M_100M_LAUNCH_COMMIT__"
DATASET_RUN_ID = "20m-100m-dataset-001"
PROFILE = "20m-100m-data-scaling-v1"
ROOT = common.WORK / "small-llm-20m-100m-data-scaling"
WORKTREE = ROOT / "launch-worktree"
EVIDENCE = ROOT / ("evidence-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
CHECKPOINTS = ROOT / "checkpoints"
SUMMARY = common.WORK / "small_llm_20m_100m_data_scaling_summary.json"
WANDB_RUN_ID = "20m-100m-data-003"
WANDB_INIT_TIMEOUT_SECONDS = "30"
WANDB_PREFLIGHT = REPO / "kaggle" / "wandb_preflight.py"
MICROBATCH_BASELINE, MICROBATCH_CANDIDATE = 1, 4
PROBE_STEPS, PROBE_WARMUP = 8, 2
MIN_SPEEDUP = 1.05
MAX_LOSS_DELTA = 0.05
MAX_GRADIENT_RELATIVE_DELTA = 0.05
MAX_RESERVED_MEMORY_FRACTION = 0.90
LOCAL_EVERY, EVAL_EVERY, REMOTE_EVERY = 250, 500, 500
MAX_STEPS_PER_SESSION = 749
_CHECKPOINT_ID = re.compile(r"^step-(\d{8})$")


class LaunchFailure(common.GateFailure):
    pass


def read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as error:
        raise LaunchFailure(f"Cannot read {label}: {path}") from error
    if not isinstance(value, Mapping):
        raise LaunchFailure(f"{label} is not a JSON object: {path}")
    return dict(value)


def wandb_preflight_command(
    uv: str,
    evidence: Path,
    entity: str | None = None,
) -> tuple[list[str], Path, Path]:
    root = evidence / "wandb-preflight"
    result = root / "result.json"
    command = [
        uv,
        "run",
        "--python",
        "3.13",
        "--with",
        "wandb==0.26.1",
        "python",
        str(WANDB_PREFLIGHT),
        "--project",
        "Small-LLM",
        "--run-id",
        WANDB_RUN_ID,
        "--run-name",
        "20M model on 100M tokens",
        "--dir",
        str(root),
        "--result",
        str(result),
        "--init-timeout",
        WANDB_INIT_TIMEOUT_SECONDS,
    ]
    if entity:
        command += ["--entity", entity]
    return command, root, result


def validate_wandb_preflight_result(path: Path) -> dict[str, Any]:
    result = read_object(path, "W&B preflight result")
    if result.get("status") != "passed":
        classification = result.get("failure_classification") or "unclassified"
        raise LaunchFailure(
            f"W&B preflight failed ({classification}); see {path.parent}"
        )
    if result.get("run_id") != WANDB_RUN_ID:
        raise LaunchFailure("W&B preflight used the wrong run ID")
    if float(result.get("init_timeout_seconds", 0)) > float(WANDB_INIT_TIMEOUT_SECONDS):
        raise LaunchFailure("W&B preflight exceeded the healthy initialization budget")
    phases = result.get("phases")
    required = {
        "secret_propagation",
        "dns",
        "tls",
        "api_key_authentication",
        "local_wandb_core",
        "project_run_resume",
    }
    if not isinstance(phases, list):
        raise LaunchFailure("W&B preflight has no phase evidence")
    observed = {
        str(row.get("name")): row
        for row in phases
        if isinstance(row, Mapping)
    }
    if set(observed) != required or any(
        row.get("status") != "passed" for row in observed.values()
    ):
        raise LaunchFailure("W&B preflight did not pass every startup phase")
    online_elapsed = observed["project_run_resume"].get("elapsed_seconds")
    if not isinstance(online_elapsed, (int, float)) or float(online_elapsed) > float(
        WANDB_INIT_TIMEOUT_SECONDS
    ):
        raise LaunchFailure("W&B online initialization was not healthy")
    debug_logs = result.get("debug_logs")
    if not isinstance(debug_logs, Mapping):
        raise LaunchFailure("W&B preflight did not preserve debug logs")
    for name in ("debug.log", "debug-internal.log", "debug-core.log"):
        row = debug_logs.get(name)
        if not isinstance(row, Mapping):
            raise LaunchFailure(f"W&B preflight did not preserve {name}")
        debug_path = Path(str(row.get("path", "")))
        if not debug_path.is_file():
            raise LaunchFailure(f"Preserved W&B log is missing: {debug_path}")
    return result


def run_wandb_preflight(
    uv: str,
    wandb_key: str,
    entity: str | None,
) -> dict[str, Any]:
    command, root, result_path = wandb_preflight_command(uv, EVIDENCE, entity)
    environment = {
        "WANDB_API_KEY": wandb_key,
        "WANDB_INIT_TIMEOUT": WANDB_INIT_TIMEOUT_SECONDS,
        "WANDB_CONFIG_DIR": str(root / "config"),
        "WANDB_CACHE_DIR": str(root / "cache"),
        "PYTHONUNBUFFERED": "1",
        "UV_LINK_MODE": "copy",
    }
    if entity:
        environment["WANDB_ENTITY"] = entity
    try:
        common.run(
            command,
            name="wandb-preflight",
            evidence=EVIDENCE,
            cwd=REPO,
            env=environment,
        )
    except common.GateFailure as error:
        if result_path.is_file():
            result = read_object(result_path, "failed W&B preflight result")
            classification = result.get("failure_classification") or "unclassified"
            if classification == "deleted_run_id":
                raise LaunchFailure(
                    f"W&B run ID {WANDB_RUN_ID!r} is tombstoned; choose a new stable "
                    f"run ID before training. Debug logs: {root / 'preserved'}"
                ) from error
            raise LaunchFailure(
                f"W&B preflight failed ({classification}); debug logs: "
                f"{root / 'preserved'}"
            ) from error
        raise LaunchFailure(
            f"W&B preflight failed before producing diagnostics; see {root}"
        ) from error
    return validate_wandb_preflight_result(result_path)


def arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--launch-commit",
        default=os.environ.get("SMALL_LLM_100M_LAUNCH_COMMIT", DEFAULT_COMMIT),
    )
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--max-steps-this-session", type=int, default=MAX_STEPS_PER_SESSION)
    return parser.parse_args(argv)


def repo_head(commit: str) -> str:
    try:
        root = Path(subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], cwd=REPO, text=True
        ).strip()).resolve()
    except (OSError, subprocess.CalledProcessError) as error:
        raise LaunchFailure("Run from the cloned Small-LLM repository") from error
    if root != REPO.resolve():
        raise LaunchFailure(f"Repository root mismatch: {root}")
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO, text=True
    ).strip():
        raise LaunchFailure("The controlling clone has tracked modifications")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise LaunchFailure("The 100M launch commit is not pinned")
    if subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=REPO,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode:
        raise LaunchFailure(f"Frozen launch commit {commit} is missing")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def prepare_worktree(commit: str) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    if WORKTREE.exists():
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(WORKTREE)], cwd=REPO,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        shutil.rmtree(WORKTREE, ignore_errors=True)
    subprocess.run(["git", "worktree", "prune"], cwd=REPO, check=False)
    common.run(
        ["git", "worktree", "add", "--detach", str(WORKTREE), commit],
        name="git-launch-worktree", evidence=EVIDENCE, cwd=REPO,
    )
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=WORKTREE, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=WORKTREE, text=True).strip()
    if actual != commit or dirty:
        raise LaunchFailure(f"Frozen worktree mismatch: {actual}, dirty={bool(dirty)}")


def profile_match(root: Path) -> tuple[bool, dict[str, Any]]:
    manifest_path, drive_path = root / "manifest.json", root / "drive_manifest.json"
    row: dict[str, Any] = {
        "root": str(root), "manifest": manifest_path.is_file(),
        "drive_manifest": drive_path.is_file(), "train": (root / "train").is_dir(),
        "validation": (root / "validation").is_dir(),
    }
    if not all(row[key] for key in ("manifest", "drive_manifest", "train", "validation")):
        return False, row
    manifest = read_object(manifest_path, "dataset manifest")
    production = manifest.get("production")
    top = {
        "schema_version": 2, "sequence_format": "context_plus_one",
        "context_length": 2048, "stored_tokens_per_sequence": 2049,
        "sequences_per_block": 16, "target_shard_bytes": 8 * 1024 * 1024,
    }
    prod = {
        "run_id": DATASET_RUN_ID, "target_source_tokens": 100_000_000,
        "minimum_source_tokens": 90_000_000, "maximum_source_tokens": 110_000_000,
        "checkpoint_source_tokens": 20_000_000, "target_reached": True,
        "remote_required": True,
    }
    matched = all(manifest.get(k) == v for k, v in top.items())
    matched = matched and isinstance(production, Mapping)
    if isinstance(production, Mapping):
        matched = matched and all(production.get(k) == v for k, v in prod.items())
    row["run_id"] = production.get("run_id") if isinstance(production, Mapping) else None
    if matched:
        row["manifest_sha256"] = common.sha256(manifest_path)
        row["drive_manifest_sha256"] = common.sha256(drive_path)
    return bool(matched), row


def find_dataset(explicit: Path | None) -> tuple[Path, list[dict[str, Any]]]:
    roots = [explicit.resolve()] if explicit else sorted({p.parent for p in common.INPUT.rglob("manifest.json")})
    inspected, matches = [], []
    for root in roots:
        matched, row = profile_match(root)
        inspected.append(row)
        if matched:
            matches.append(root)
    if len(matches) != 1:
        raise LaunchFailure(
            f"Expected exactly one attached 100M dataset; found {len(matches)}.\n"
            + json.dumps(inspected, indent=2)
        )
    return matches[0], inspected


def validate_plan(path: Path) -> dict[str, Any]:
    plan = read_object(path, "100M trainer plan")
    trainer, train, identity = plan.get("trainer"), plan.get("train"), plan.get("identity")
    if plan.get("version") != 1 or plan.get("qualification_profile") != PROFILE:
        raise LaunchFailure("Trainer plan profile mismatch")
    if plan.get("context_length") != 2048 or plan.get("sequences_per_block") != 16:
        raise LaunchFailure("Trainer plan block geometry mismatch")
    if plan.get("target_shard_bytes") != 8 * 1024 * 1024:
        raise LaunchFailure("Trainer plan shard geometry mismatch")
    if not all(isinstance(x, Mapping) for x in (trainer, train, identity)):
        raise LaunchFailure("Trainer plan is missing required sections")
    assert isinstance(trainer, Mapping) and isinstance(train, Mapping) and isinstance(identity, Mapping)
    steps = trainer.get("steps")
    phases = [trainer.get("warmup_updates"), trainer.get("stable_updates"), trainer.get("decay_updates")]
    if not isinstance(steps, int) or steps <= 0 or not all(isinstance(v, int) and v > 0 for v in phases):
        raise LaunchFailure("Trainer plan has invalid update counts")
    if sum(int(v) for v in phases) != steps:
        raise LaunchFailure("Trainer plan phases do not sum to total steps")
    if trainer.get("passes") != 1 or trainer.get("schedule") != "wsd":
        raise LaunchFailure("Trainer plan is not one-pass WSD")
    if trainer.get("full_block_target_tokens") != 32768 or trainer.get("minimum_lr_ratio") != 0.1:
        raise LaunchFailure("Trainer plan changed optimization geometry")
    if train.get("block_ids") != list(range(steps)):
        raise LaunchFailure("Trainer plan block IDs are not contiguous")
    if not isinstance(trainer.get("validation_blocks"), int) or trainer["validation_blocks"] <= 0:
        raise LaunchFailure("Trainer plan has no validation blocks")
    if identity.get("drive_run_id") != DATASET_RUN_ID:
        raise LaunchFailure("Trainer plan Drive identity mismatch")
    for key in ("manifest_sha256", "drive_manifest_sha256"):
        if not isinstance(identity.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", identity[key]):
            raise LaunchFailure(f"Trainer plan has invalid {key}")
    return plan


def checkpoint_step(checkpoint_id: str) -> int:
    match = _CHECKPOINT_ID.fullmatch(checkpoint_id)
    if match is None:
        raise LaunchFailure(f"Invalid checkpoint ID: {checkpoint_id!r}")
    return int(match.group(1))


def trainer_command(
    uv: str, dataset: Path, plan: Mapping[str, Any], checkpoint_dir: Path,
    *, additional_steps: int, microbatch: int, online: bool,
    entity: str | None = None, resume: str | None = None,
) -> list[str]:
    p = plan["trainer"]
    assert isinstance(p, Mapping)
    cmd = [
        uv, "run", "--python", "3.13", "--extra", "model",
        "--with", "wandb==0.26.1", "--with-requirements", "dataset/requirements-remote.txt",
        "python", "-m", "trainer", "--dataset-dir", str(dataset),
        "--dataset-manifest", str(dataset / "manifest.json"),
        "--checkpoint-dir", str(checkpoint_dir), "--steps", str(additional_steps),
        "--sequences-per-block", "16", "--model-size", "smoke",
        "--architecture", "gdn2_hybrid", "--gdn-chunk-size", "32",
        "--initialization", "normal", "--optimizer", "hybrid_muon_adamw",
        "--device", "cuda", "--precision", "fp16", "--microbatch-size", str(microbatch),
        "--learning-rate", "3e-4", "--weight-decay", "0.1",
        "--muon-momentum", "0.95", "--muon-lr-multiplier", "1.0",
        "--muon-update-rms", "0.18", "--muon-weight-decay", "0.1",
        "--max-grad-norm", "1.0", "--schedule", "wsd",
        "--warmup-tokens", str(p["warmup_tokens"]),
        "--stable-tokens", str(p["stable_tokens"]),
        "--decay-tokens", str(p["decay_tokens"]),
        "--minimum-lr-ratio", "0.1", "--seed", "17",
    ]
    if resume:
        cmd += ["--resume", resume]
    if not online:
        return cmd + [
            "--checkpoint-every-steps", "0", "--evaluation-every-steps", "0",
            "--validation-blocks", "0", "--remote-publish-every-steps", "0",
            "--wandb-mode", "disabled",
        ]
    cmd += [
        "--checkpoint-every-steps", str(LOCAL_EVERY),
        "--evaluation-every-steps", str(EVAL_EVERY),
        "--validation-blocks", str(p["validation_blocks"]),
        "--remote-publish-every-steps", str(REMOTE_EVERY),
        "--remote-drive-manifest", str(dataset / "drive_manifest.json"),
        "--remote-token-env", "HF_TOKEN", "--wandb-mode", "online",
        "--wandb-project", "Small-LLM", "--wandb-run-id", WANDB_RUN_ID,
        "--wandb-run-name", "20M model on 100M tokens", "--wandb-tags",
        "20m", "100m-tokens", "t4", "data-scaling", "microbatch-4",
        "one-pass", "segmented-exact-resume",
    ]
    # A timed-out first init may already have created the fixed run server-side.
    cmd += ["--wandb-resume", "must" if resume else "allow"]
    if entity:
        cmd += ["--wandb-entity", entity]
    return cmd


def training_rows(path: Path, expected_steps: int) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
        except (TypeError, ValueError):
            continue
        if isinstance(item, Mapping) and all(k in item for k in ("step", "block_id", "loss", "tokens_per_second")):
            rows.append(dict(item))
    if len(rows) != expected_steps:
        raise LaunchFailure(f"Expected {expected_steps} probe rows, found {len(rows)}")
    for index, row in enumerate(rows, start=1):
        if row.get("step") != index or row.get("block_id") != index - 1:
            raise LaunchFailure("Probe step/block cursor mismatch")
        if row.get("sequences") != 16 or row.get("target_tokens") != 32768:
            raise LaunchFailure("Probe changed effective optimizer batch")
        if row.get("consumed_tokens") != index * 32768:
            raise LaunchFailure("Probe consumed-token cursor mismatch")
        for key in ("loss", "gradient_norm", "tokens_per_second", "grad_scaler_scale", "learning_rate"):
            value = row.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise LaunchFailure(f"Probe emitted non-finite {key}")
        if row.get("overflow_retries") != 0 or row.get("overflow_events_total") != 0:
            raise LaunchFailure("Probe encountered an FP16 overflow")
    return rows


def probe_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    measured = rows[PROBE_WARMUP:]
    tps = [float(row["tokens_per_second"]) for row in measured]
    return {
        "median_tokens_per_second": statistics.median(tps),
        "mean_tokens_per_second": statistics.fmean(tps),
        "maximum_peak_allocated_bytes": max(int(row["peak_memory_bytes"]) for row in rows),
        "maximum_peak_reserved_bytes": max(int(row["peak_reserved_memory_bytes"]) for row in rows),
        "final_loss": float(rows[-1]["loss"]),
        "clipped_updates": sum(bool(row.get("gradient_clipped")) for row in rows),
    }


def compare_probes(baseline: list[dict[str, Any]], candidate: list[dict[str, Any]], gpu_bytes: int) -> dict[str, Any]:
    left, right = probe_summary(baseline), probe_summary(candidate)
    loss_delta, grad_delta = [], []
    for a, b in zip(baseline, candidate):
        if any(a[k] != b[k] for k in ("block_id", "target_tokens", "consumed_tokens", "learning_rate")):
            raise LaunchFailure("Probes did not execute the same token schedule")
        loss_delta.append(abs(float(a["loss"]) - float(b["loss"])))
        ag, bg = abs(float(a["gradient_norm"])), abs(float(b["gradient_norm"]))
        grad_delta.append(abs(ag - bg) / max(ag, bg, 1e-12))
    speedup = float(right["median_tokens_per_second"]) / float(left["median_tokens_per_second"])
    memory_fraction = float(right["maximum_peak_reserved_bytes"]) / gpu_bytes
    verdict = {
        "status": "pending", "baseline_microbatch": 1, "selected_microbatch": 4,
        "observed_median_speedup": speedup, "minimum_required_speedup": MIN_SPEEDUP,
        "maximum_step_loss_delta": max(loss_delta), "maximum_allowed_step_loss_delta": MAX_LOSS_DELTA,
        "maximum_gradient_relative_delta": max(grad_delta),
        "maximum_allowed_gradient_relative_delta": MAX_GRADIENT_RELATIVE_DELTA,
        "candidate_reserved_memory_fraction": memory_fraction,
        "maximum_allowed_reserved_memory_fraction": MAX_RESERVED_MEMORY_FRACTION,
        "results": {"1": left, "4": right},
    }
    if speedup < MIN_SPEEDUP:
        raise LaunchFailure("Microbatch 4 did not improve throughput by at least 5%: " + json.dumps(verdict, indent=2))
    if max(loss_delta) > MAX_LOSS_DELTA:
        raise LaunchFailure("Microbatch 4 loss trajectory exceeded tolerance: " + json.dumps(verdict, indent=2))
    if max(grad_delta) > MAX_GRADIENT_RELATIVE_DELTA:
        raise LaunchFailure("Microbatch 4 gradient trajectory exceeded tolerance: " + json.dumps(verdict, indent=2))
    if memory_fraction > MAX_RESERVED_MEMORY_FRACTION:
        raise LaunchFailure("Microbatch 4 memory headroom failed: " + json.dumps(verdict, indent=2))
    verdict["status"] = "passed"
    return verdict


def qualify_microbatch(uv: str, dataset: Path, plan: Mapping[str, Any], env: Mapping[str, str]) -> dict[str, Any]:
    histories = {}
    for microbatch in (MICROBATCH_BASELINE, MICROBATCH_CANDIDATE):
        checkpoints = EVIDENCE / f"probe-{microbatch}-checkpoints"
        stage = common.run(
            trainer_command(
                uv, dataset, plan, checkpoints, additional_steps=PROBE_STEPS,
                microbatch=microbatch, online=False,
            ),
            name=f"microbatch-{microbatch}-probe", evidence=EVIDENCE,
            cwd=WORKTREE, env=env,
        )
        histories[microbatch] = training_rows(Path(stage["log"]), PROBE_STEPS)
        shutil.rmtree(checkpoints, ignore_errors=True)
    total_mib = int(subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"], text=True
    ).splitlines()[0].strip().replace(",", ""))
    return compare_probes(histories[1], histories[4], total_mib * 1024 * 1024)


RESTORE_SCRIPT = r'''
import json, os, sys
from pathlib import Path
from dataset.src.joint_checkpoint import restore_on_empty_vps
from dataset.src.remote import HuggingFaceCheckpointStore, TwoPhaseCheckpointPublisher, sha256_path
repo_id, run_id, destination, attached_manifest, output = sys.argv[1:6]
destination, attached_manifest, output = Path(destination), Path(attached_manifest), Path(output)
store = HuggingFaceCheckpointStore(repo_id, token=os.environ["HF_TOKEN"], private=True)
pointer_path = f"run/{run_id}/latest.json"
pointer = store.read_json(pointer_path)
if pointer is None:
    output.write_text(json.dumps({"status": "missing", "pointer_path": pointer_path}) + "\n")
    raise SystemExit(0)
root = restore_on_empty_vps(
    publisher=TwoPhaseCheckpointPublisher(store, run_id=run_id), store=None,
    run_id=run_id, destination=destination, checkpoint_pointer=pointer, prefetch_shards=0,
)
if sha256_path(root / "drive_manifest.json") != sha256_path(attached_manifest):
    raise RuntimeError("remote checkpoint Drive manifest differs from attached dataset")
checkpoint = json.loads((root / "checkpoint.json").read_text())
last = checkpoint.get("pipeline_state", {}).get("last_consumed_block_id")
if not isinstance(last, int):
    raise RuntimeError("restored checkpoint has no integer block cursor")
result = {"status": "restored", "checkpoint_id": pointer["checkpoint_id"],
          "checkpoint_root": str(root), "last_consumed_block_id": last}
output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps(result, sort_keys=True))
'''


def restore_latest(uv: str, dataset: Path, repo_id: str, env: Mapping[str, str]) -> dict[str, Any] | None:
    script, output = EVIDENCE / "restore.py", EVIDENCE / "restore.json"
    script.write_text(RESTORE_SCRIPT, encoding="utf-8")
    if CHECKPOINTS.exists():
        shutil.rmtree(CHECKPOINTS)
    common.run(
        [uv, "run", "--python", "3.13", "--extra", "model",
         "--with-requirements", "dataset/requirements-remote.txt", "python", str(script),
         repo_id, DATASET_RUN_ID, str(ROOT), str(dataset / "drive_manifest.json"), str(output)],
        name="restore-latest-checkpoint", evidence=EVIDENCE, cwd=WORKTREE, env=env,
    )
    result = read_object(output, "remote restore result")
    if result.get("status") == "missing":
        return None
    if result.get("status") != "restored" or not isinstance(result.get("checkpoint_id"), str):
        raise LaunchFailure("Unexpected remote restore result")
    step = checkpoint_step(result["checkpoint_id"])
    if result.get("last_consumed_block_id") != step - 1:
        raise LaunchFailure("Restored checkpoint and block cursor disagree")
    if Path(str(result.get("checkpoint_root"))) != CHECKPOINTS / result["checkpoint_id"]:
        raise LaunchFailure("Restored checkpoint path mismatch")
    result["step"] = step
    return result


def segment_plan(completed: int, total: int, maximum: int) -> dict[str, int | bool]:
    if maximum <= 0 or completed < 0 or completed > total:
        raise LaunchFailure("Invalid session bound or restored step")
    remaining = total - completed
    additional = min(remaining, maximum)
    if additional < remaining and (completed + additional) % REMOTE_EVERY == 0:
        additional -= 1
    if remaining and additional <= 0:
        raise LaunchFailure("Session cannot make forward progress")
    return {
        "completed_steps": completed, "remaining_steps_before_session": remaining,
        "additional_steps_this_session": additional, "expected_final_step": completed + additional,
        "complete_before_session": remaining == 0, "complete_after_session": additional == remaining,
    }


def verify_segment_log(path: Path, expected_step: int) -> str:
    checkpoint_id = f"step-{expected_step:08d}"
    final_line = remote_final = False
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
        except (TypeError, ValueError):
            continue
        if not isinstance(item, Mapping):
            continue
        final_line |= item.get("checkpoint_id") == checkpoint_id
        event = item.get("remote_publication")
        remote_final |= isinstance(event, Mapping) and event.get("checkpoint_id") == checkpoint_id and event.get("final") is True
    if not final_line or not remote_final:
        raise LaunchFailure(f"Missing final verified remote publication {checkpoint_id}")
    return checkpoint_id


def main(argv: Sequence[str] | None = None) -> int:
    args = arguments(argv)
    state: dict[str, Any] = {
        "schema_version": 2, "started_utc": common.now(), "status": "initializing",
        "authorization": "20m_100m_data_scaling_authorized", "launch_commit": args.launch_commit,
        "dataset_run_id": DATASET_RUN_ID, "profile": PROFILE, "model_parameters": 20_637_592,
        "effective_sequences_per_update": 16, "selected_microbatch_size": 4,
        "checkpoint_dir": str(CHECKPOINTS), "evidence_dir": str(EVIDENCE),
        "wandb_run_id": WANDB_RUN_ID, "maximum_steps_this_session": args.max_steps_this_session,
    }
    try:
        state["environment"] = common.check_environment()
        state["controller_head"] = repo_head(args.launch_commit)
        wandb_key, hf_token = common.secret("WANDB_API_KEY"), common.secret("HF_TOKEN")
        hf_repo = common.secret("SMALL_LLM_HF_REPO_ID")
        entity = common.secret("WANDB_ENTITY", required=False)
        assert wandb_key and hf_token and hf_repo
        EVIDENCE.mkdir(parents=True, exist_ok=False)
        common.write_json(SUMMARY, state)
        prepare_worktree(args.launch_commit)
        dataset, inspected = find_dataset(args.dataset_dir)
        state.update(dataset_dir=str(dataset), datasets_inspected=inspected, remote_checkpoint_repo=hf_repo)

        common.run([sys.executable, "-m", "pip", "install", "-q", "uv"],
                   name="install-uv", evidence=EVIDENCE, cwd=WORKTREE)
        uv = common.find_uv()
        common.run([uv, "python", "install", "3.13"],
                   name="install-python-3.13", evidence=EVIDENCE, cwd=WORKTREE)
        state["wandb_preflight"] = run_wandb_preflight(uv, wandb_key, entity)
        common.write_json(SUMMARY, state)
        base = [uv, "run", "--python", "3.13"]
        common.run(base + ["--with-requirements", "dataset/requirements-remote.txt",
                   "python", "-m", "dataset.main", "verify", "--output-dir", str(dataset), "--full-scan"],
                   name="dataset-full-scan", evidence=EVIDENCE, cwd=WORKTREE)
        plan_path = EVIDENCE / "qualification_plan.json"
        common.run(base + ["python", "-m", "dataset.qualification", "report", "--profile", "20m-100m",
                   "--dataset-dir", str(dataset), "--drive-manifest", str(dataset / "drive_manifest.json"),
                   "--output", str(plan_path)], name="qualification-plan", evidence=EVIDENCE, cwd=WORKTREE)
        plan = validate_plan(plan_path)
        state["plan"] = plan
        env = {
            "WANDB_API_KEY": wandb_key,
            "WANDB_INIT_TIMEOUT": WANDB_INIT_TIMEOUT_SECONDS,
            "HF_TOKEN": hf_token,
            "SMALL_LLM_HF_REPO_ID": hf_repo,
            "UV_LINK_MODE": "copy",
            "PYTHONUNBUFFERED": "1",
        }
        if entity:
            env["WANDB_ENTITY"] = entity
        restored = restore_latest(uv, dataset, hf_repo, env)
        state["remote_restore"] = restored or {"status": "fresh"}
        total = int(plan["trainer"]["steps"])
        completed = int(restored["step"]) if restored else 0
        segment = segment_plan(completed, total, args.max_steps_this_session)
        state["session_plan"] = segment
        if segment["complete_before_session"]:
            state.update(status="already_completed", completed_utc=common.now())
            common.write_json(SUMMARY, state)
            print(f"Run already complete at {completed}/{total}. Summary: {SUMMARY}")
            return 0
        state["microbatch_qualification"] = (
            qualify_microbatch(uv, dataset, plan, env) if restored is None
            else {"status": "inherited_from_verified_checkpoint", "selected_microbatch": 4}
        )
        resume = str(restored["checkpoint_id"]) if restored else None
        command = trainer_command(
            uv, dataset, plan, CHECKPOINTS,
            additional_steps=int(segment["additional_steps_this_session"]), microbatch=4,
            online=True, entity=entity, resume=resume,
        )
        state.update(status="running", trainer_started_utc=common.now(), trainer_command=command)
        common.write_json(SUMMARY, state)
        stage = common.run(
            command,
            name=f"trainer-{completed + 1:08d}-{int(segment['expected_final_step']):08d}",
            evidence=EVIDENCE, cwd=WORKTREE, env=env,
        )
        final_step = int(segment["expected_final_step"])
        checkpoint = verify_segment_log(Path(stage["log"]), final_step)
        remaining = total - final_step
        state.update(
            status="completed" if segment["complete_after_session"] else "segment_completed",
            segment_completed_utc=common.now(), final_step=final_step,
            final_checkpoint_id=checkpoint, remaining_steps=remaining,
        )
        common.write_json(SUMMARY, state)
        if remaining:
            print(f"Segment published at {checkpoint}; rerun the same entry point. Summary: {SUMMARY}")
        else:
            print(f"100M-token run completed. Summary: {SUMMARY}")
        return 0
    except KeyboardInterrupt:
        state.update(status="interrupted", finished_utc=common.now())
        common.write_json(SUMMARY, state)
        return 130
    except Exception as error:
        state.update(status="failed", finished_utc=common.now(), error=f"{type(error).__name__}: {error}")
        common.write_json(SUMMARY, state)
        print(f"LAUNCH FAILED CLOSED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
