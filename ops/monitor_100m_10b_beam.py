#!/usr/bin/env python3
"""Hourly, fail-closed supervisor for the Beam 100M/10B run.

The supervisor treats the Hugging Face checkpoint pointer as the resume
authority, Beam's active-task list as the allocation authority, and W&B's
runtime counter as the compute-time estimate since the local billing baseline
was reset.  It never launches when the estimated spend is at or above the
configured cap, and it refuses to launch if either remote control plane cannot
be read.

When ``SMALL_LLM_BEAM_BILLING_MODE=account_zero`` is set, the account's
reported zero charge is printed as the account-cost basis.  The notional
serverless estimate remains the hard cap.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from typing import Any


RUN_ID = "100m-10b-data-001"
MODEL = "100M"
TOKENS = "10B"
GPU = "RTX4090"
MICROBATCH = 4
TOTAL_STEPS = 76_294
BLOCK_TOKENS = 64 * 2_048
SOURCE_COMMIT = "1f9dff920ecc45ce2fdb43fd875514a18391273d"

# Beam's published serverless rates: RTX4090 $0.69/GPU-hour,
# CPU $0.19008/core-hour, and RAM $0.02016/GiB-hour.  The function requests
# one GPU, four CPU cores, and 32 GiB RAM.
GPU_USD_PER_HOUR = Decimal("0.69")
CPU_USD_PER_CORE_HOUR = Decimal("0.19008")
RAM_USD_PER_GIB_HOUR = Decimal("0.02016")
GPU_COUNT = Decimal("1")
CPU_CORES = Decimal("4")
RAM_GIB = Decimal("32")
HOURLY_COMPUTE_USD = (
    GPU_COUNT * GPU_USD_PER_HOUR
    + CPU_CORES * CPU_USD_PER_CORE_HOUR
    + RAM_GIB * RAM_USD_PER_GIB_HOUR
)

BUDGET_USD = Decimal(os.environ.get("SMALL_LLM_BEAM_BUDGET_USD", "30"))
BILLING_MODE = os.environ.get("SMALL_LLM_BEAM_BILLING_MODE", "serverless_estimate")
# The notional serverless estimate is the hard cap even when the account has
# promotional/free billing.  This keeps the experiment bounded by the user's
# requested $30 resource budget rather than by the account invoice alone.
CAP_BASIS = os.environ.get("SMALL_LLM_BEAM_CAP_BASIS", "notional")
# W&B runtime does not include every short CPU gate and can lag during a live
# task.  Keep a conservative 10% estimate margin and reserve $0.50 for gates.
SAFETY_FACTOR = Decimal("1.10")
GATE_RESERVE_USD = Decimal("0.50")
# This is deliberately below the observed ~25--33k target tokens/s.  It turns
# the remaining dollar budget into a bounded step segment with headroom for a
# slower retry; the hourly monitor never assumes the full 10B run is free.
CONSERVATIVE_TOKENS_PER_SECOND = Decimal("20000")

DEFAULT_SOURCE_ROOT = Path("/tmp/small-llm-beam-resume-1f9dff9")
DEFAULT_STATE_ROOT = Path("/tmp/small-llm-beam-monitor")
DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
BILLING_RESET_FILENAME = "billing_reset.json"


def _json_command(command: list[str]) -> Any:
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def _beam_tasks(beam: str) -> list[dict[str, Any]]:
    value = _json_command([beam, "task", "list", "--limit", "1000", "--format", "json"])
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise RuntimeError("Beam task list was not a JSON array of objects")
    return value


def _is_relevant_task(task: dict[str, Any]) -> bool:
    stub_name = str(task.get("stub_name", ""))
    return "beam.launch" in stub_name or "beam.vps_train" in stub_name


def _active_tasks(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    active_statuses = {"PENDING", "QUEUED", "RUNNING", "STARTING", "IN_PROGRESS"}
    return [
        task
        for task in tasks
        if _is_relevant_task(task) and str(task.get("status", "")).upper() in active_statuses
    ]


def _stop_tasks(beam: str, tasks: list[dict[str, Any]]) -> None:
    task_ids = [str(task.get("id")) for task in tasks if task.get("id")]
    if not task_ids:
        return
    result = subprocess.run(
        [beam, "task", "stop", *task_ids],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"Beam task stop failed: {detail}")


def _wandb_run() -> tuple[dict[str, Any], float]:
    import wandb

    api = wandb.Api()
    entity = os.environ.get("WANDB_ENTITY") or api.default_entity
    run = api.run(f"{entity}/Small-LLM/{RUN_ID}")
    summary = dict(run.summary._json_dict)
    runtime = float(summary.get("_runtime") or 0.0)
    if runtime < 0:
        raise RuntimeError(f"W&B runtime is invalid: {runtime}")
    return {
        "entity": entity,
        "id": str(run.id),
        "name": str(run.name),
        "state": str(run.state),
        "url": str(run.url),
        "summary_runtime_seconds": runtime,
        "summary_step_field": summary.get("_step"),
    }, runtime


def _latest_checkpoint() -> dict[str, Any]:
    from huggingface_hub import hf_hub_download

    repo = os.environ["SMALL_LLM_HF_REPO_ID"]
    token = os.environ["HF_TOKEN"]
    path = hf_hub_download(
        repo_id=repo,
        filename=f"run/{RUN_ID}/latest.json",
        token=token,
        force_download=False,
    )
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    checkpoint_id = value.get("checkpoint_id")
    if not isinstance(checkpoint_id, str) or not checkpoint_id.startswith("step-"):
        raise RuntimeError("HF latest pointer has no valid checkpoint_id")
    try:
        step = int(checkpoint_id.removeprefix("step-"))
    except ValueError as error:
        raise RuntimeError(f"HF latest pointer has invalid checkpoint_id: {checkpoint_id}") from error
    if not 0 <= step <= TOTAL_STEPS:
        raise RuntimeError(f"HF latest step is outside the frozen plan: {step}")
    return {"checkpoint_id": checkpoint_id, "step": step}


def _estimated_cost(runtime_seconds: float) -> Decimal:
    runtime_cost = Decimal(str(runtime_seconds)) / Decimal("3600") * HOURLY_COMPUTE_USD
    return runtime_cost * SAFETY_FACTOR


def _billing_reset_path(state_root: Path) -> Path:
    return state_root / BILLING_RESET_FILENAME


def _load_billing_reset(state_root: Path) -> dict[str, Any] | None:
    path = _billing_reset_path(state_root)
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("billing reset state is not a JSON object")
    try:
        baseline_runtime = float(value["baseline_wandb_runtime_seconds"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("billing reset state has no valid W&B runtime baseline") from error
    if baseline_runtime < 0:
        raise RuntimeError("billing reset state has a negative W&B runtime baseline")
    reset_at = value.get("reset_at")
    if not isinstance(reset_at, str):
        raise RuntimeError("billing reset state has no reset timestamp")
    return {**value, "baseline_wandb_runtime_seconds": baseline_runtime}


def _write_billing_reset(
    state_root: Path,
    *,
    run: dict[str, Any],
    runtime: float,
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    state_root.mkdir(parents=True, exist_ok=True)
    value = {
        "reset_at": datetime.now(timezone.utc).isoformat(),
        "reason": "new_beam_account",
        "baseline_wandb_runtime_seconds": runtime,
        "baseline_checkpoint": checkpoint,
        "wandb_run": {"entity": run["entity"], "id": run["id"]},
    }
    path = _billing_reset_path(state_root)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return value


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _active_elapsed_since_reset(active: list[dict[str, Any]], reset_at: str) -> float:
    reset_time = _parse_timestamp(reset_at)
    if reset_time is None:
        raise RuntimeError("billing reset state has an invalid reset timestamp")
    now = datetime.now(timezone.utc)
    elapsed = 0.0
    for task in active:
        started = None
        for field in ("started_at", "start_time", "created_at", "submitted_at"):
            started = _parse_timestamp(task.get(field))
            if started is not None:
                break
        if started is None:
            continue
        elapsed = max(elapsed, max(0.0, (now - max(started, reset_time)).total_seconds()))
    return elapsed


def _runtime_since_reset(
    runtime: float,
    active: list[dict[str, Any]],
    billing_reset: dict[str, Any],
) -> float:
    baseline = float(billing_reset["baseline_wandb_runtime_seconds"])
    wandb_delta = max(0.0, runtime - baseline)
    active_elapsed = _active_elapsed_since_reset(active, str(billing_reset["reset_at"]))
    return max(wandb_delta, active_elapsed)


def _round_money(value: Decimal) -> str:
    return str(value.quantize(Decimal("0.01")))


def _pid_alive(path: Path) -> bool:
    try:
        pid = int(path.read_text(encoding="utf-8").strip())
        os.kill(pid, 0)
    except (FileNotFoundError, OSError, ValueError):
        return False
    return True


def _source_is_exact(source_root: Path) -> None:
    if not source_root.is_dir():
        raise RuntimeError(f"exact resume source checkout is missing: {source_root}")
    actual = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != SOURCE_COMMIT:
        raise RuntimeError(f"resume source drift: expected {SOURCE_COMMIT}, found {actual}")
    dirty = subprocess.check_output(
        ["git", "-C", str(source_root), "status", "--porcelain"], text=True
    ).strip()
    if dirty:
        raise RuntimeError("exact resume source checkout is dirty")


def _ensure_exact_source(source_root: Path, project_root: Path) -> None:
    if not source_root.exists():
        source_root.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--no-local", str(project_root), str(source_root)],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "-C", str(source_root), "checkout", "--detach", SOURCE_COMMIT],
            check=True,
            capture_output=True,
            text=True,
        )
    _source_is_exact(source_root)


def _launch_bounded_segment(
    *,
    source_root: Path,
    project_root: Path,
    state_root: Path,
    steps: int,
) -> dict[str, Any]:
    _ensure_exact_source(source_root, project_root)
    state_root.mkdir(parents=True, exist_ok=True)
    log_path = state_root / f"launch-{int(time.time())}.log"
    python = os.environ.get("SMALL_LLM_PYTHON", str(project_root / ".venv/bin/python"))
    command = [
        python,
        str(source_root / "beam/vps_train.py"),
        "--model",
        MODEL,
        "--tokens",
        TOKENS,
        "--gpu",
        GPU,
        "--microbatch-size",
        str(MICROBATCH),
        "--max-steps-this-session",
        str(steps),
    ]
    handle = log_path.open("a", encoding="utf-8")
    try:
        process = subprocess.Popen(
            command,
            cwd=source_root,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    finally:
        handle.close()
    (state_root / "launcher.pid").write_text(str(process.pid) + "\n", encoding="utf-8")
    return {"pid": process.pid, "log": str(log_path), "steps_requested": steps}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="report the decision without launching")
    parser.add_argument(
        "--reset-billing",
        action="store_true",
        help="reset the notional billing baseline to the current W&B runtime",
    )
    args = parser.parse_args()

    state_root = Path(os.environ.get("SMALL_LLM_BEAM_MONITOR_STATE", str(DEFAULT_STATE_ROOT)))
    project_root = Path(os.environ.get("SMALL_LLM_PROJECT_ROOT", str(DEFAULT_PROJECT_ROOT)))
    source_root = Path(os.environ.get("SMALL_LLM_BEAM_SOURCE_ROOT", str(DEFAULT_SOURCE_ROOT)))
    beam = os.environ.get("SMALL_LLM_BEAM_CLI", str(project_root / ".venv/bin/beam"))
    lock_path = state_root.with_suffix(".lock")
    state_root.mkdir(parents=True, exist_ok=True)

    with lock_path.open("w", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps({"decision": "check_already_running"}, sort_keys=True))
            return 0

        pid_path = state_root / "launcher.pid"
        launcher_alive = _pid_alive(pid_path)
        pid_path.unlink(missing_ok=True)

        try:
            tasks = _beam_tasks(beam)
            active = _active_tasks(tasks)
            run, runtime = _wandb_run()
            checkpoint = _latest_checkpoint()
        except Exception as error:  # fail closed: an unreadable control plane must not allocate
            print(json.dumps({"decision": "blocked_control_plane_error", "error": f"{type(error).__name__}: {error}"}, sort_keys=True))
            return 2

        if args.reset_billing:
            if active or launcher_alive:
                print(
                    json.dumps(
                        {
                            "decision": "billing_reset_blocked_active_training",
                            "active_tasks": [task.get("id") for task in active],
                            "launcher_process_alive": launcher_alive,
                        },
                        sort_keys=True,
                    )
                )
                return 2
            billing_reset = _write_billing_reset(
                state_root,
                run=run,
                runtime=runtime,
                checkpoint=checkpoint,
            )
            print(
                json.dumps(
                    {
                        "decision": "billing_reset",
                        "reset_at": billing_reset["reset_at"],
                        "baseline_wandb_runtime_seconds": runtime,
                        "baseline_checkpoint": checkpoint,
                        "budget_usd": _round_money(BUDGET_USD),
                        "cap_basis": CAP_BASIS,
                        "account_billing_mode": BILLING_MODE,
                    },
                    sort_keys=True,
                )
            )
            return 0

        try:
            billing_reset = _load_billing_reset(state_root)
        except Exception as error:
            print(json.dumps({"decision": "billing_reset_state_error", "error": f"{type(error).__name__}: {error}"}, sort_keys=True))
            return 2
        if billing_reset is None:
            print(
                json.dumps(
                    {
                        "decision": "billing_reset_required",
                        "message": "Run with --reset-billing after a Beam account change; refusing to use pre-reset W&B runtime.",
                    },
                    sort_keys=True,
                )
            )
            return 2

        runtime_since_reset = _runtime_since_reset(runtime, active, billing_reset)
        notional_cost = _estimated_cost(runtime_since_reset)
        account_cost = Decimal("0") if BILLING_MODE == "account_zero" else notional_cost
        cap_cost = notional_cost if CAP_BASIS == "notional" else account_cost
        base = {
            "billing_mode": BILLING_MODE,
            "cap_basis": CAP_BASIS,
            "budget_usd": _round_money(BUDGET_USD),
            "checkpoint": checkpoint,
            "account_cost_basis_usd": _round_money(account_cost),
            "cap_cost_basis_usd": _round_money(cap_cost),
            "hourly_compute_usd": _round_money(HOURLY_COMPUTE_USD),
            "notional_cost_estimate_usd": _round_money(notional_cost),
            "billing_reset_at": billing_reset["reset_at"],
            "baseline_wandb_runtime_seconds": billing_reset["baseline_wandb_runtime_seconds"],
            "runtime_seconds_since_reset": runtime_since_reset,
            "active_tasks": [
                {"id": task.get("id"), "status": task.get("status"), "stub_name": task.get("stub_name")}
                for task in active
            ],
            "run": run,
            "runtime_seconds": runtime,
        }
        if cap_cost >= BUDGET_USD:
            if active:
                try:
                    _stop_tasks(beam, active)
                except Exception as error:
                    print(json.dumps({**base, "decision": "budget_exceeded_stop_failed", "error": f"{type(error).__name__}: {error}"}, sort_keys=True, default=str))
                    return 2
                print(json.dumps({**base, "decision": "budget_exceeded_active_tasks_stopped"}, sort_keys=True, default=str))
                return 0
            print(json.dumps({**base, "decision": "budget_exhausted"}, sort_keys=True, default=str))
            return 0
        if active:
            print(json.dumps({**base, "decision": "active_task_present"}, sort_keys=True, default=str))
            return 0
        if launcher_alive:
            print(json.dumps({**base, "decision": "launcher_process_still_running"}, sort_keys=True, default=str))
            return 0
        if checkpoint["step"] >= TOTAL_STEPS:
            print(json.dumps({**base, "decision": "training_complete"}, sort_keys=True, default=str))
            return 0

        remaining_for_segment = BUDGET_USD - cap_cost - GATE_RESERVE_USD
        if remaining_for_segment <= 0:
            print(json.dumps({**base, "decision": "budget_reserve_exhausted"}, sort_keys=True, default=str))
            return 0
        safe_seconds = remaining_for_segment / HOURLY_COMPUTE_USD * Decimal("3600")
        estimated_steps = int(
            (safe_seconds * CONSERVATIVE_TOKENS_PER_SECOND / Decimal(BLOCK_TOKENS)).to_integral_value(
                rounding=ROUND_DOWN
            )
        )
        steps = min(estimated_steps, TOTAL_STEPS - checkpoint["step"])
        if steps < 1:
            print(json.dumps({**base, "decision": "no_budget_for_one_step"}, sort_keys=True, default=str))
            return 0
        if args.dry_run:
            print(json.dumps({**base, "decision": "would_launch", "steps_requested": steps}, sort_keys=True, default=str))
            return 0

        try:
            launch = _launch_bounded_segment(
                source_root=source_root,
                project_root=project_root,
                state_root=state_root,
                steps=steps,
            )
        except Exception as error:
            print(json.dumps({**base, "decision": "launch_blocked", "error": f"{type(error).__name__}: {error}"}, sort_keys=True, default=str))
            return 2
        print(json.dumps({**base, "decision": "launched_bounded_segment", "launch": launch}, sort_keys=True, default=str))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
