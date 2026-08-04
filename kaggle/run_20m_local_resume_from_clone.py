#!/usr/bin/env python3
"""Qualify actual-process interruption and exact local resume on Kaggle.

Usage:
    %cd /kaggle/working/Small-LLM
    !git pull --ff-only
    !python kaggle/run_20m_local_resume_from_clone.py

The controller clone stays on ``main``. Evidence-producing trainer commands run
inside a clean detached worktree at the frozen launch commit. One uninterrupted
50-update reference is followed by a second trainer process that is terminated
as an actual process group immediately after its verified update-25 checkpoint.
A fresh process then resumes that checkpoint for updates 26 through 50. The
combined interrupted/resumed trajectory and semantic checkpoint state must match
the uninterrupted reference exactly. This test does not perform remote recovery
or authorize the complete 306-update run.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = Path(__file__).with_name("run_20m_repeatability_from_clone.py")
WORK = Path("/kaggle/working/small-llm-local-resume-controller")
LATEST = Path("/kaggle/working/small_llm_local_resume_summary.json")
COMMIT = "45d1da4a1ac3f18cf6ce02b8439672f10e2c8b4c"
STEPS = 50
BOUNDARY = 25
TOKENS_PER_STEP = 32_768


def load_helper():
    spec = importlib.util.spec_from_file_location("small_llm_repeatability_helper", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load repository helper: {HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


helper = load_helper()
helper.WORK = WORK
base = helper.base
GateFailure = base.GateFailure


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def run_id_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Same-T4 actual-process interruption and local-resume qualification."
    )
    parser.add_argument(
        "--launch-commit",
        default=os.environ.get("SMALL_LLM_LAUNCH_COMMIT", COMMIT),
    )
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument(
        "--wandb-run-prefix",
        default=os.environ.get("SMALL_LLM_WANDB_RESUME_PREFIX"),
    )
    return parser.parse_args(argv)


def set_flag(command: list[str], flag: str, value: str) -> None:
    try:
        index = command.index(flag)
    except ValueError as error:
        raise GateFailure(f"Expected trainer flag missing from helper command: {flag}") from error
    if index + 1 >= len(command):
        raise GateFailure(f"Trainer flag has no value: {flag}")
    command[index + 1] = value


def trainer_command(
    uv: str,
    dataset: Path,
    checkpoints: Path,
    *,
    run_id: str,
    run_name: str,
    entity: str | None,
    steps: int,
    resume: str | None = None,
) -> list[str]:
    command = helper.trainer_command(
        uv,
        dataset,
        checkpoints,
        run_id,
        run_name,
        entity,
    )
    set_flag(command, "--steps", str(steps))
    set_flag(command, "--wandb-resume", "must" if resume else "never")
    tags_index = command.index("--wandb-tags")
    command[tags_index + 1 :] = [
        "20m",
        "t4",
        "local-resume",
        "process-kill",
    ] + (["--wandb-entity", entity] if entity else [])
    if resume:
        command.extend(["--resume", resume])
    return command


def require_finite(value: Any, path: str = "root") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise GateFailure(f"Non-finite value at {path}: {value!r}")
    if isinstance(value, Mapping):
        for key, child in value.items():
            require_finite(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            require_finite(child, f"{path}[{index}]")


def parse_training_log(
    path: Path,
    *,
    expected_steps: Sequence[int],
    expected_validation_count: int,
    expected_final_checkpoint: str | None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    final_ids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
        except (ValueError, TypeError):
            continue
        if not isinstance(item, Mapping):
            continue
        if {"step", "block_id", "loss", "gradient_norm"}.issubset(item):
            row = dict(item)
            require_finite(row)
            rows.append(row)
        elif isinstance(item.get("validation"), Mapping):
            validation = dict(item["validation"])
            require_finite(validation)
            validations.append(validation)
        elif isinstance(item.get("local_checkpoint"), Mapping):
            checkpoints.append(dict(item["local_checkpoint"]))
        elif isinstance(item.get("checkpoint_id"), str):
            final_ids.append(str(item["checkpoint_id"]))

    actual_steps = [int(row["step"]) for row in rows]
    wanted_steps = list(expected_steps)
    if actual_steps != wanted_steps:
        raise GateFailure(
            f"Unexpected successful-update sequence in {path}: "
            f"expected {wanted_steps}, got {actual_steps}"
        )
    for row in rows:
        step = int(row["step"])
        expected = {
            "block_id": step - 1,
            "sequences": 16,
            "target_tokens": TOKENS_PER_STEP,
            "consumed_tokens": step * TOKENS_PER_STEP,
        }
        for key, wanted in expected.items():
            if row.get(key) != wanted:
                raise GateFailure(
                    f"Trajectory identity mismatch at step {step}, {key}: "
                    f"{row.get(key)!r} != {wanted!r}"
                )
    if len(validations) != expected_validation_count:
        raise GateFailure(
            f"Expected {expected_validation_count} validation event(s) in {path}, "
            f"found {len(validations)}"
        )
    if expected_final_checkpoint is None:
        if final_ids:
            raise GateFailure(f"Interrupted process unexpectedly reached a final checkpoint: {final_ids}")
    elif final_ids != [expected_final_checkpoint]:
        raise GateFailure(
            f"Expected final checkpoint {expected_final_checkpoint!r} in {path}, got {final_ids}"
        )
    return {
        "rows": rows,
        "validation": validations[0] if validations else None,
        "checkpoint_events": checkpoints,
        "final_ids": final_ids,
    }


def verify_checkpoint_boundary(root: Path, checkpoint_id: str) -> dict[str, Any]:
    checkpoint = root / checkpoint_id
    manifest_path = checkpoint / "local_manifest.json"
    payload_path = checkpoint / "checkpoint.json"
    if not manifest_path.is_file() or not payload_path.is_file():
        raise GateFailure(f"Checkpoint is not complete: {checkpoint}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping) or not isinstance(manifest.get("files"), list):
        raise GateFailure(f"Invalid local checkpoint manifest: {manifest_path}")
    listed: set[str] = set()
    for entry in manifest["files"]:
        if not isinstance(entry, Mapping):
            raise GateFailure(f"Malformed checkpoint manifest entry in {manifest_path}")
        name = entry.get("name")
        digest = entry.get("sha256")
        if not isinstance(name, str) or not isinstance(digest, str):
            raise GateFailure(f"Malformed checkpoint file identity in {manifest_path}")
        candidate = checkpoint / name
        if not candidate.is_file() or base.sha256(candidate) != digest:
            raise GateFailure(f"Checkpoint file verification failed: {candidate}")
        listed.add(name)
    if not {"trainer_state.pkl", "checkpoint.json"}.issubset(listed):
        raise GateFailure(f"Checkpoint manifest is incomplete: {manifest_path}")

    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    if payload.get("checkpoint_id") != checkpoint_id:
        raise GateFailure(f"Checkpoint ID mismatch in {payload_path}")
    if payload.get("optimizer_step_complete") is not True:
        raise GateFailure(f"Checkpoint is not at an optimizer boundary: {payload_path}")
    pipeline = payload.get("pipeline_state")
    if not isinstance(pipeline, Mapping):
        raise GateFailure(f"Checkpoint has no pipeline state: {payload_path}")
    expected_last = BOUNDARY - 1
    if pipeline.get("last_consumed_block_id") != expected_last:
        raise GateFailure(
            f"Checkpoint cursor mismatch: expected {expected_last}, "
            f"got {pipeline.get('last_consumed_block_id')!r}"
        )
    if int(pipeline.get("gradient_accumulation_position", 0)) != 0:
        raise GateFailure("Checkpoint contains a partial gradient-accumulation window")
    return {
        "path": str(checkpoint),
        "checkpoint_id": checkpoint_id,
        "last_consumed_block_id": expected_last,
        "manifest_sha256": base.sha256(manifest_path),
        "checkpoint_json_sha256": base.sha256(payload_path),
        "trainer_state_sha256": base.sha256(checkpoint / "trainer_state.pkl"),
    }


def checkpoint_event(line: str, checkpoint_id: str) -> bool:
    try:
        item = json.loads(line)
    except (ValueError, TypeError):
        return False
    if not isinstance(item, Mapping) or not isinstance(item.get("local_checkpoint"), Mapping):
        return False
    return item["local_checkpoint"].get("checkpoint_id") == checkpoint_id


def wait_for_group_exit(pgid: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        time.sleep(0.1)
    return False


def run_interrupted_process(
    command: Sequence[str],
    *,
    evidence: Path,
    cwd: Path,
    env: Mapping[str, str],
    checkpoint_root: Path,
) -> dict[str, Any]:
    name = "trainer-interrupted-at-25"
    log_path = evidence / f"{name}.log"
    exit_path = evidence / f"{name}.exit-code"
    print(f"\n=== {name} ===\n$ {base.safe_cmd(command)}", flush=True)
    merged_env = os.environ.copy()
    merged_env.update(env)
    started = time.perf_counter()
    detected: dict[str, Any] | None = None
    kill_sent_utc: str | None = None
    forced_kill = False

    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            list(command),
            cwd=str(cwd),
            env=merged_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        assert process.stdout is not None
        pgid = process.pid
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
            if checkpoint_event(line, "step-00000025"):
                detected = verify_checkpoint_boundary(checkpoint_root, "step-00000025")
                if process.poll() is not None:
                    raise GateFailure("Trainer exited before the interruption signal was sent")
                kill_sent_utc = utc_now()
                os.killpg(pgid, signal.SIGTERM)
                break
        if detected is None:
            code = process.wait()
            exit_path.write_text(f"{code}\n", encoding="utf-8")
            raise GateFailure(
                "Trainer exited without emitting and verifying the update-25 checkpoint event"
            )
        try:
            code = process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            forced_kill = True
            os.killpg(pgid, signal.SIGKILL)
            code = process.wait(timeout=15)
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        log.flush()

    exit_path.write_text(f"{code}\n", encoding="utf-8")
    if code == 0:
        raise GateFailure("The intended interruption did not terminate the trainer")
    group_gone = wait_for_group_exit(pgid, 5.0)
    if not group_gone:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            group_gone = True
        else:
            group_gone = wait_for_group_exit(pgid, 5.0)
    if not group_gone:
        raise GateFailure("The trainer process group remained alive after interruption")

    detected_after_kill = verify_checkpoint_boundary(checkpoint_root, "step-00000025")
    if detected != detected_after_kill:
        raise GateFailure("The update-25 checkpoint changed after process termination")
    return {
        "name": name,
        "exit_code": code,
        "expected_nonzero_exit": True,
        "elapsed_seconds": time.perf_counter() - started,
        "log": str(log_path),
        "log_sha256": base.sha256(log_path),
        "signal": "SIGTERM",
        "forced_sigkill": forced_kill,
        "kill_sent_utc": kill_sent_utc,
        "process_group_id": pgid,
        "process_group_gone": True,
        "checkpoint": detected_after_kill,
    }


SEMANTIC_COMPARATOR = r'''
from __future__ import annotations
import json, math, pickle, sys
from collections.abc import Mapping
from pathlib import Path
import torch

left_root, right_root, output = map(Path, sys.argv[1:4])
differences = []
counts = {"tensors": 0, "tensor_elements": 0, "scalars": 0, "containers": 0}


def record(path, left, right, reason):
    if len(differences) < 100:
        differences.append({
            "path": path,
            "reason": reason,
            "left_type": type(left).__name__,
            "right_type": type(right).__name__,
            "left": repr(left)[:300],
            "right": repr(right)[:300],
        })


def compare(left, right, path="root"):
    if type(left) is not type(right):
        record(path, left, right, "type_mismatch")
        return
    if torch.is_tensor(left):
        counts["tensors"] += 1
        counts["tensor_elements"] += left.numel()
        if left.shape != right.shape or left.dtype != right.dtype:
            record(path, left, right, "tensor_metadata_mismatch")
            return
        if not torch.equal(left.detach().cpu(), right.detach().cpu()):
            delta = None
            if left.is_floating_point() or left.is_complex():
                delta = float((left.detach().cpu() - right.detach().cpu()).abs().max().item())
            record(path, left.shape, right.shape, f"tensor_value_mismatch:max_abs={delta}")
        return
    if isinstance(left, Mapping):
        counts["containers"] += 1
        if set(left) != set(right):
            record(path, sorted(map(repr, left)), sorted(map(repr, right)), "mapping_keys_mismatch")
            return
        for key in sorted(left, key=repr):
            compare(left[key], right[key], f"{path}[{key!r}]")
        return
    if isinstance(left, (list, tuple)):
        counts["containers"] += 1
        if len(left) != len(right):
            record(path, len(left), len(right), "sequence_length_mismatch")
            return
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            compare(a, b, f"{path}[{index}]")
        return
    counts["scalars"] += 1
    if isinstance(left, float) and math.isnan(left) and math.isnan(right):
        return
    try:
        equal = left == right
    except Exception:
        equal = repr(left) == repr(right)
    if isinstance(equal, torch.Tensor):
        equal = bool(torch.all(equal).item())
    if not bool(equal):
        record(path, left, right, "value_mismatch")


left_json = json.loads((left_root / "checkpoint.json").read_text())
right_json = json.loads((right_root / "checkpoint.json").read_text())
compare(left_json, right_json, "checkpoint_json")
with (left_root / "trainer_state.pkl").open("rb") as handle:
    left_state = pickle.load(handle)
with (right_root / "trainer_state.pkl").open("rb") as handle:
    right_state = pickle.load(handle)
compare(left_state, right_state, "trainer_state")
result = {
    "semantic_exact": not differences,
    "difference_count": len(differences),
    "differences": differences,
    "counts": counts,
}
output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps(result, sort_keys=True))
raise SystemExit(0)
'''


def compare_checkpoint_semantics(
    uv: str,
    worktree: Path,
    evidence: Path,
    env: Mapping[str, str],
    left: Path,
    right: Path,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    script = evidence / f"compare-{label}.py"
    output = evidence / f"compare-{label}.json"
    script.write_text(SEMANTIC_COMPARATOR, encoding="utf-8")
    stage = base.run(
        [
            uv,
            "run",
            "--python",
            "3.13",
            "--extra",
            "model",
            "python",
            str(script),
            str(left),
            str(right),
            str(output),
        ],
        name=f"compare-checkpoint-semantics-{label}",
        evidence=evidence,
        cwd=worktree,
        env=env,
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    if result.get("semantic_exact") is not True:
        raise GateFailure(f"Semantic checkpoint mismatch for {label}")
    return result, stage


def main(argv: Sequence[str] | None = None) -> int:
    args = arguments(argv)
    WORK.mkdir(parents=True, exist_ok=True)
    evidence = WORK / f"small-llm-local-resume-{stamp()}"
    evidence.mkdir(parents=True, exist_ok=False)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "started_utc": utc_now(),
        "status": "running",
        "authorization": "none",
        "scope": "same-T4 actual-process update-25 interruption and exact local resume",
        "evidence": str(evidence),
        "gradient_clipping_policy": {
            "accepted_for_qualification": True,
            "max_grad_norm": 1.0,
            "basis": "user accepted the bounded, exactly repeatable universal clipping pattern",
        },
        "stages": [],
    }
    try:
        summary["environment"] = base.check_environment()
        wandb_key = base.secret("WANDB_API_KEY")
        entity = base.secret("WANDB_ENTITY", required=False)
        if not wandb_key:
            raise GateFailure("Missing required Kaggle Secret: WANDB_API_KEY")

        worktree, checkout, checkout_stage = helper.prepare_worktree(
            args.launch_commit,
            evidence,
        )
        summary["checkout"] = checkout
        summary["stages"].append(checkout_stage)
        env = {
            "WANDB_API_KEY": wandb_key,
            "UV_LINK_MODE": "copy",
            "PYTHONUNBUFFERED": "1",
        }

        summary["stages"].append(
            base.run(
                [sys.executable, "-m", "pip", "install", "-q", "uv"],
                name="install-uv",
                evidence=evidence,
            )
        )
        uv = base.find_uv()
        summary["stages"].append(
            base.run(
                [uv, "python", "install", "3.13"],
                name="install-python-3.13",
                evidence=evidence,
                cwd=worktree,
                env=env,
            )
        )
        summary["stages"].append(
            base.run(
                [uv, "run", "--python", "3.13", "python", "--version"],
                name="python-version",
                evidence=evidence,
                cwd=worktree,
                env=env,
            )
        )

        dataset, inspected = base.find_dataset(args.dataset_dir)
        summary["dataset"] = {"path": str(dataset), "inspected": inspected}
        summary["stages"].append(
            base.run(
                [
                    uv,
                    "run",
                    "--python",
                    "3.13",
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
                evidence=evidence,
                cwd=worktree,
                env=env,
            )
        )
        plan_path = evidence / "qualification_plan.json"
        summary["stages"].append(
            base.run(
                [
                    uv,
                    "run",
                    "--python",
                    "3.13",
                    "python",
                    "-m",
                    "dataset.qualification_20m_report",
                    "--dataset-dir",
                    str(dataset),
                    "--drive-manifest",
                    str(dataset / "drive_manifest.json"),
                    "--output",
                    str(plan_path),
                ],
                name="qualification-plan",
                evidence=evidence,
                cwd=worktree,
                env=env,
            )
        )
        summary["plan"] = base.validate_plan(plan_path)

        prefix = args.wandb_run_prefix or f"20m-t4-resume-{run_id_stamp()}"
        reference_id = f"{prefix}-reference"
        resumed_id = f"{prefix}-interrupted-resumed"
        reference_checkpoints = evidence / "reference" / "checkpoints"
        resume_checkpoints = evidence / "interrupted-resumed" / "checkpoints"

        reference_command = trainer_command(
            uv,
            dataset,
            reference_checkpoints,
            run_id=reference_id,
            run_name="20M T4 local-resume reference",
            entity=entity,
            steps=STEPS,
        )
        summary["stages"].append(
            base.run(
                reference_command,
                name="trainer-reference-50",
                evidence=evidence,
                cwd=worktree,
                env=env,
            )
        )
        summary["stages"].append(
            helper.verify_checkpoints(
                uv,
                worktree,
                evidence,
                reference_checkpoints,
                "reference",
                env,
            )
        )

        interrupted_command = trainer_command(
            uv,
            dataset,
            resume_checkpoints,
            run_id=resumed_id,
            run_name="20M T4 actual-process local resume",
            entity=entity,
            steps=STEPS,
        )
        interruption = run_interrupted_process(
            interrupted_command,
            evidence=evidence,
            cwd=worktree,
            env=env,
            checkpoint_root=resume_checkpoints,
        )
        summary["stages"].append(interruption)

        resumed_command = trainer_command(
            uv,
            dataset,
            resume_checkpoints,
            run_id=resumed_id,
            run_name="20M T4 actual-process local resume",
            entity=entity,
            steps=STEPS - BOUNDARY,
            resume="step-00000025",
        )
        summary["stages"].append(
            base.run(
                resumed_command,
                name="trainer-resumed-25-to-50",
                evidence=evidence,
                cwd=worktree,
                env=env,
            )
        )
        summary["stages"].append(
            helper.verify_checkpoints(
                uv,
                worktree,
                evidence,
                resume_checkpoints,
                "interrupted-resumed",
                env,
            )
        )

        reference = parse_training_log(
            evidence / "trainer-reference-50.log",
            expected_steps=range(1, 51),
            expected_validation_count=1,
            expected_final_checkpoint="step-00000050",
        )
        interrupted = parse_training_log(
            evidence / "trainer-interrupted-at-25.log",
            expected_steps=range(1, 26),
            expected_validation_count=0,
            expected_final_checkpoint=None,
        )
        resumed = parse_training_log(
            evidence / "trainer-resumed-25-to-50.log",
            expected_steps=range(26, 51),
            expected_validation_count=1,
            expected_final_checkpoint="step-00000050",
        )
        combined_rows = list(interrupted["rows"]) + list(resumed["rows"])
        metric_comparison = helper.compare_rows(reference["rows"], combined_rows)
        if not metric_comparison.get("numeric_trajectory_exact"):
            raise GateFailure("Interrupted/resumed numerical trajectory differs from reference")
        if reference["validation"] != resumed["validation"]:
            raise GateFailure("Interrupted/resumed validation differs from reference")

        checkpoint_semantics: dict[str, Any] = {}
        checkpoint_stages: list[dict[str, Any]] = []
        for checkpoint_id, label in [
            ("step-00000025", "step-25"),
            ("step-00000050", "step-50"),
        ]:
            result, stage = compare_checkpoint_semantics(
                uv,
                worktree,
                evidence,
                env,
                reference_checkpoints / checkpoint_id,
                resume_checkpoints / checkpoint_id,
                label,
            )
            checkpoint_semantics[label] = result
            checkpoint_stages.append(stage)
        summary["stages"].extend(checkpoint_stages)

        reference_summary = helper.summarize(reference)
        resumed_complete = {
            "rows": combined_rows,
            "validation": resumed["validation"],
            "checkpoint_events": interrupted["checkpoint_events"] + resumed["checkpoint_events"],
        }
        resumed_summary = helper.summarize(resumed_complete)
        summary["runs"] = {
            "reference": {
                "wandb_run_id": reference_id,
                "checkpoint_dir": str(reference_checkpoints),
                "metrics": reference_summary,
            },
            "interrupted_resumed": {
                "wandb_run_id": resumed_id,
                "checkpoint_dir": str(resume_checkpoints),
                "metrics": resumed_summary,
                "interruption": interruption,
            },
        }
        summary["comparison"] = {
            "metric_trajectory": metric_comparison,
            "validation_exact": True,
            "checkpoint_semantics": checkpoint_semantics,
            "checkpoint_tree_exact": {
                "step_25": helper.checkpoint_digest(reference_checkpoints / "step-00000025")["tree_sha256"]
                == helper.checkpoint_digest(resume_checkpoints / "step-00000025")["tree_sha256"],
                "step_50": helper.checkpoint_digest(reference_checkpoints / "step-00000050")["tree_sha256"]
                == helper.checkpoint_digest(resume_checkpoints / "step-00000050")["tree_sha256"],
            },
            "resume_class": "exact_local_resume",
        }
        summary["checkpoint_trees"] = {
            "reference": {
                "step_25": helper.checkpoint_digest(reference_checkpoints / "step-00000025"),
                "step_50": helper.checkpoint_digest(reference_checkpoints / "step-00000050"),
            },
            "interrupted_resumed": {
                "step_25": helper.checkpoint_digest(resume_checkpoints / "step-00000025"),
                "step_50": helper.checkpoint_digest(resume_checkpoints / "step-00000050"),
            },
        }
        summary["status"] = "passed_local_interruption_resume"
        summary["authorization"] = "remote_recovery_only"
        summary["finished_utc"] = utc_now()
        base.write_json(evidence / "summary.json", summary)
        base.write_json(LATEST, summary)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return 0
    except BaseException as error:
        summary["status"] = "failed"
        summary["authorization"] = "none"
        summary["finished_utc"] = utc_now()
        summary["error"] = f"{type(error).__name__}: {error}"
        base.write_json(evidence / "summary.json", summary)
        base.write_json(LATEST, summary)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
