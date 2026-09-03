#!/usr/bin/env python3
"""Hourly crash and billing guard for the aggressive WSqD continuation.

This supervisor is deliberately separate from the older 100M/10B monitor.  It
only recognizes the aggressive launcher, uses its W&B run, and relaunches the
same exact command after a crash while the notional Beam budget remains below
the cap.  An unreadable control plane blocks relaunch rather than guessing.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any


RUN_ID = "100m-10b-aggressive-wsqd-from-step15500"
TASK_NEEDLE = "beam.aggressive_wsqd_10b_from_15500"
GPU = "RTX4090"
TMUX_SESSION = "aggressive-wsqd-10b"
SOURCE_ROOT = Path("/tmp/small-llm-beam-aggressive-54d8b1a")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = SOURCE_ROOT / "beam/aggressive_wsqd_10b_from_15500.py"
STATE_ROOT = Path(os.environ.get("SMALL_LLM_AGGRESSIVE_MONITOR_STATE", "/tmp/small-llm-aggressive-monitor"))
RESET_FILE = STATE_ROOT / "billing_reset.json"
LOCK_FILE = STATE_ROOT.with_suffix(".lock")

GPU_USD_PER_HOUR = Decimal("0.66")
CPU_USD_PER_CORE_HOUR = Decimal("0.19008")
RAM_USD_PER_GIB_HOUR = Decimal("0.02016")
HOURLY_COMPUTE_USD = GPU_USD_PER_HOUR + 4 * CPU_USD_PER_CORE_HOUR + 32 * RAM_USD_PER_GIB_HOUR
BUDGET_USD = Decimal(os.environ.get("SMALL_LLM_BEAM_BUDGET_USD", "30"))
SAFETY_FACTOR = Decimal("1.10")


def _json_command(command: list[str]) -> Any:
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def _beam_tasks(beam: str) -> list[dict[str, Any]]:
    value = _json_command([beam, "task", "list", "--limit", "1000", "--format", "json"])
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise RuntimeError("Beam task list was not a JSON array of objects")
    return value


def _active_tasks(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    active_statuses = {"PENDING", "QUEUED", "RUNNING", "STARTING", "IN_PROGRESS"}
    return [
        task
        for task in tasks
        if TASK_NEEDLE in str(task.get("stub_name", ""))
        and str(task.get("status", "")).upper() in active_statuses
    ]


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return timestamp.replace(tzinfo=timezone.utc) if timestamp.tzinfo is None else timestamp.astimezone(timezone.utc)


def _active_elapsed(active: list[dict[str, Any]], reset_at: str) -> float:
    reset_time = _parse_timestamp(reset_at)
    if reset_time is None:
        raise RuntimeError("billing reset timestamp is invalid")
    now = datetime.now(timezone.utc)
    elapsed = 0.0
    for task in active:
        for field in ("started_at", "start_time", "created_at", "submitted_at"):
            started = _parse_timestamp(task.get(field))
            if started is not None:
                elapsed = max(elapsed, (now - max(started, reset_time)).total_seconds())
                break
    return max(0.0, elapsed)


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
    }, runtime


def _wandb_run_optional() -> tuple[dict[str, Any], float]:
    try:
        return _wandb_run()
    except Exception as error:
        return {"state": "unknown", "error": f"{type(error).__name__}: {error}"}, 0.0


def _load_reset() -> dict[str, Any] | None:
    if not RESET_FILE.exists():
        return None
    value = json.loads(RESET_FILE.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("reset_at"), str):
        raise RuntimeError("billing reset state is invalid")
    return value


def _write_reset() -> dict[str, Any]:
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    value = {
        "reset_at": datetime.now(timezone.utc).isoformat(),
        "reason": "new_beam_account",
        "baseline_wandb_runtime_seconds": 0.0,
        "billing_gpu": GPU,
        "rate_start_runtime_seconds": 0.0,
        "accrued_notional_cost_usd": "0",
        "run_id": RUN_ID,
    }
    temporary = RESET_FILE.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(RESET_FILE)
    return value


def _write_state(value: dict[str, Any]) -> None:
    temporary = RESET_FILE.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(RESET_FILE)


def _money_decimal(value: Any, *, field: str) -> Decimal:
    try:
        return Decimal(str(value))
    except Exception as error:
        raise RuntimeError(f"{field} is not a decimal amount") from error


def _billing_state(reset: dict[str, Any]) -> tuple[str, float, Decimal]:
    """Return the rate segment, migrating the pre-switch A10G state safely."""
    gpu = str(reset.get("billing_gpu") or "A10G").upper()
    if gpu not in {"A10G", "RTX4090", "RTX5090"}:
        raise RuntimeError(f"unsupported billing GPU in monitor state: {gpu}")
    baseline = float(reset.get("rate_start_runtime_seconds", 0.0))
    if baseline < 0:
        raise RuntimeError("billing rate-start runtime is invalid")
    accrued = _money_decimal(reset.get("accrued_notional_cost_usd", "0"), field="accrued notional cost")
    if accrued < 0:
        raise RuntimeError("accrued notional cost is invalid")
    return gpu, baseline, accrued


def _hourly_compute_cost(gpu: str) -> Decimal:
    gpu_prices = {
        "A10G": Decimal("1.05"),
        "RTX4090": Decimal("0.66"),
        "RTX5090": Decimal("0.71"),
    }
    try:
        gpu_price = gpu_prices[gpu]
    except KeyError as error:
        raise RuntimeError(f"unsupported billed GPU: {gpu}") from error
    return gpu_price + 4 * CPU_USD_PER_CORE_HOUR + 32 * RAM_USD_PER_GIB_HOUR


def _launcher_alive() -> bool:
    result = subprocess.run(["ps", "-eo", "pid,args"], check=True, capture_output=True, text=True)
    needle = f"aggressive_wsqd_10b_from_15500.py --gpu {GPU}"
    return any(needle in line and str(os.getpid()) not in line for line in result.stdout.splitlines())


def _stop_tasks(beam: str, active: list[dict[str, Any]]) -> None:
    ids = [str(task["id"]) for task in active if task.get("id")]
    if not ids:
        return
    result = subprocess.run([beam, "task", "stop", *ids], check=False, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout).strip())


def _tmux_exists() -> bool:
    return subprocess.run(["tmux", "has-session", "-t", TMUX_SESSION], check=False, capture_output=True).returncode == 0


def _relaunch() -> dict[str, Any]:
    if not SOURCE_ROOT.is_dir() or not LAUNCHER.is_file():
        raise RuntimeError(f"exact launcher is missing: {LAUNCHER}")
    if not _tmux_exists():
        raise RuntimeError(f"tmux session is missing: {TMUX_SESSION}")
    command = f"cd {SOURCE_ROOT} && {PROJECT_ROOT / '.venv/bin/python'} {LAUNCHER} --gpu {GPU}"
    result = subprocess.run(
        ["tmux", "send-keys", "-t", f"{TMUX_SESSION}:0.0", command, "C-m"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout).strip())
    return {"tmux_session": TMUX_SESSION, "command": command}


def _money(value: Decimal) -> str:
    return str(value.quantize(Decimal("0.01")))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reset-billing", action="store_true")
    parser.add_argument("--transition-gpu", choices=("A10G", "RTX4090", "RTX5090"))
    args = parser.parse_args()

    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    with LOCK_FILE.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps({"decision": "check_already_running"}, sort_keys=True))
            return 0

        reset = _write_reset() if args.reset_billing else _load_reset()
        if reset is None:
            print(json.dumps({"decision": "billing_reset_required"}, sort_keys=True))
            return 2

        beam = os.environ.get("SMALL_LLM_BEAM_CLI", str(PROJECT_ROOT / ".venv/bin/beam"))
        try:
            tasks = _beam_tasks(beam)
            active = _active_tasks(tasks)
            run, runtime = _wandb_run_optional()
            launcher_alive = _launcher_alive()
        except Exception as error:
            print(json.dumps({"decision": "blocked_control_plane_error", "error": f"{type(error).__name__}: {error}"}, sort_keys=True))
            return 2

        runtime_since_reset = max(
            0.0,
            runtime - float(reset.get("baseline_wandb_runtime_seconds", 0.0)),
            _active_elapsed(active, str(reset["reset_at"])),
        )
        billed_gpu, rate_start_runtime, accrued_cost = _billing_state(reset)
        segment_runtime = max(0.0, runtime_since_reset - rate_start_runtime)
        hourly_compute_usd = _hourly_compute_cost(billed_gpu)
        notional_cost = accrued_cost + Decimal(str(segment_runtime)) / Decimal("3600") * hourly_compute_usd * SAFETY_FACTOR
        base = {
            "run_id": RUN_ID,
            "run_state": run["state"],
            "active_tasks": [{"id": task.get("id"), "status": task.get("status"), "stub_name": task.get("stub_name")} for task in active],
            "launcher_alive": launcher_alive,
            "runtime_seconds_since_reset": runtime_since_reset,
            "billing_gpu": billed_gpu,
            "hourly_compute_usd": _money(hourly_compute_usd),
            "notional_cost_estimate_usd": _money(notional_cost),
            "budget_usd": _money(BUDGET_USD),
            "billing_reset_at": reset["reset_at"],
        }

        if args.reset_billing:
            print(json.dumps({**base, "decision": "billing_reset"}, sort_keys=True, default=str))
            return 0
        if args.transition_gpu:
            target_gpu = args.transition_gpu.upper()
            if target_gpu != GPU:
                raise RuntimeError(
                    f"monitor launcher requests {GPU}, so refusing accounting transition to {target_gpu}"
                )
            if target_gpu == billed_gpu:
                print(json.dumps({**base, "decision": "billing_gpu_already_current"}, sort_keys=True, default=str))
                return 0
            if args.dry_run:
                print(json.dumps({**base, "decision": "would_transition_billing_gpu", "next_billing_gpu": target_gpu}, sort_keys=True, default=str))
                return 0
            reset.update(
                {
                    "billing_gpu": target_gpu,
                    "rate_start_runtime_seconds": runtime_since_reset,
                    "accrued_notional_cost_usd": str(notional_cost),
                }
            )
            _write_state(reset)
            print(json.dumps({**base, "decision": "billing_gpu_transitioned", "next_billing_gpu": target_gpu}, sort_keys=True, default=str))
            return 0
        if notional_cost >= BUDGET_USD:
            if active and not args.dry_run:
                _stop_tasks(beam, active)
                print(json.dumps({**base, "decision": "budget_exceeded_active_tasks_stopped"}, sort_keys=True, default=str))
            else:
                print(json.dumps({**base, "decision": "budget_exhausted"}, sort_keys=True, default=str))
            return 0
        if active:
            print(json.dumps({**base, "decision": "active_task_present"}, sort_keys=True, default=str))
            return 0
        if run["state"].lower() in {"finished", "success", "completed"}:
            print(json.dumps({**base, "decision": "training_complete"}, sort_keys=True, default=str))
            return 0
        if run["state"] == "unknown":
            print(json.dumps({**base, "decision": "blocked_wandb_unavailable"}, sort_keys=True, default=str))
            return 2
        if launcher_alive:
            print(json.dumps({**base, "decision": "launcher_alive_without_active_task"}, sort_keys=True, default=str))
            return 0
        if args.dry_run:
            print(json.dumps({**base, "decision": "would_relaunch_after_crash"}, sort_keys=True, default=str))
            return 0
        launch = _relaunch()
        print(json.dumps({**base, "decision": "relaunched_after_crash", "launch": launch}, sort_keys=True, default=str))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
