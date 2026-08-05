"""Minimal human-readable console output for the 100M Kaggle launcher."""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Mapping, Sequence

_TRAINER_STAGE = re.compile(r"^trainer-(\d{8})-(\d{8})$")
_PROBE_STAGE = re.compile(r"^microbatch-(\d+)-probe$")


def _argument(command: Sequence[str], flag: str) -> str | None:
    try:
        index = list(command).index(flag)
    except ValueError:
        return None
    return str(command[index + 1]) if index + 1 < len(command) else None


def _seconds(value: float) -> str:
    if value < 1:
        return f"{value * 1000:.0f}ms"
    if value < 60:
        return f"{value:.1f}s"
    minutes, seconds = divmod(int(value), 60)
    return f"{minutes}m{seconds:02d}s"


def _bytes(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "?"
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if amount < 1024 or unit == "GiB":
            return f"{amount:.1f}{unit}" if unit != "B" else f"{int(amount)}B"
        amount /= 1024
    return f"{amount:.1f}GiB"


def _number(value: object, digits: int = 4) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "?"
    number = float(value)
    if not math.isfinite(number):
        return str(number)
    return f"{number:.{digits}f}"


def _rate(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "?"
    number = float(value)
    return f"{number / 1000:.1f}k" if abs(number) >= 1000 else f"{number:.0f}"


def _stage(name: str) -> tuple[str, str]:
    fixed = {
        "git-launch-worktree": ("setup", "load frozen experiment code"),
        "install-uv": ("setup", "install uv"),
        "install-python-3.13": ("setup", "prepare Python 3.13"),
        "dataset-full-scan": ("dataset", "verify every attached shard"),
        "qualification-plan": ("dataset", "derive the exact one-pass schedule"),
        "restore-latest-checkpoint": ("resume", "check the latest remote checkpoint"),
    }
    if name in fixed:
        return fixed[name]
    probe = _PROBE_STAGE.fullmatch(name)
    if probe:
        return "probe", f"microbatch {probe.group(1)} for 8 updates"
    trainer = _TRAINER_STAGE.fullmatch(name)
    if trainer:
        return "train", f"global steps {int(trainer.group(1))}-{int(trainer.group(2))}"
    return "stage", name.replace("-", " ")


class _Formatter:
    def __init__(self, name: str, command: Sequence[str]) -> None:
        self.name = name
        self.command = list(command)
        self.probe = _PROBE_STAGE.fullmatch(name)
        self.trainer = _TRAINER_STAGE.fullmatch(name)
        self.steps = int(_argument(command, "--steps") or 0)
        self.resume = _argument(command, "--resume")
        self.start_step = (
            int(self.resume.removeprefix("step-"))
            if self.resume and self.resume.startswith("step-")
            else 0
        )

    def render(self, line: str) -> str | None:
        text = line.strip()
        if not text:
            return None
        if text.startswith("W&B telemetry enabled:"):
            return "[wandb] " + text.removeprefix("W&B telemetry enabled: ")
        if any(word in text.lower() for word in ("error", "failed", "warning", "traceback")):
            return "[detail] " + text
        try:
            item = json.loads(text)
        except (TypeError, ValueError):
            return None
        if not isinstance(item, Mapping):
            return None
        if all(key in item for key in ("step", "block_id", "loss", "tokens_per_second")):
            return self._training(item)
        validation = item.get("validation")
        if isinstance(validation, Mapping):
            parts = [f"loss {_number(validation.get('loss'))}"]
            if isinstance(validation.get("perplexity"), (int, float)):
                parts.append(f"ppl {_number(validation.get('perplexity'), 2)}")
            batches = validation.get("batches") or validation.get("batch_count")
            if isinstance(batches, int):
                parts.append(f"{batches} batches")
            if isinstance(item.get("elapsed_seconds"), (int, float)):
                parts.append(_seconds(float(item["elapsed_seconds"])))
            return "[validation] " + " | ".join(parts)
        local = item.get("local_checkpoint")
        if isinstance(local, Mapping):
            return (
                f"[checkpoint] saved {local.get('checkpoint_id')} | "
                f"{_bytes(local.get('byte_size'))} | "
                f"{_seconds(float(local.get('elapsed_seconds', 0)))}"
            )
        remote = item.get("remote_publication")
        if isinstance(remote, Mapping):
            flags = []
            if remote.get("final") is True:
                flags.append("final")
            if remote.get("best_updated") is True:
                flags.append("new best")
            detail = f" | {', '.join(flags)}" if flags else ""
            loss = remote.get("validation_loss")
            if isinstance(loss, (int, float)):
                detail += f" | val loss {_number(loss)}"
            return (
                f"[checkpoint] published {remote.get('checkpoint_id')}"
                f"{detail} | {_seconds(float(remote.get('elapsed_seconds', 0)))}"
            )
        if isinstance(item.get("checkpoint_id"), str) and len(item) == 1:
            return f"[segment] complete at {item['checkpoint_id']}"
        if item.get("status") == "restored" and isinstance(item.get("checkpoint_id"), str):
            return (
                f"[resume] restored {item['checkpoint_id']} | "
                f"last block {item.get('last_consumed_block_id')}"
            )
        return None

    def _training(self, item: Mapping[str, object]) -> str:
        step = int(item["step"])
        session_step = max(1, step - self.start_step)
        total = self.steps or session_step
        percent = min(100.0, 100.0 * session_step / max(total, 1))
        label = f"probe mb={self.probe.group(1)}" if self.probe else "train"
        parts = [
            f"[{label}] {session_step}/{total} ({percent:.1f}%)",
            f"global {step}",
            f"block {item['block_id']}",
            f"loss {_number(item.get('loss'))}",
        ]
        lr = item.get("learning_rate")
        if isinstance(lr, (int, float)):
            parts.append(f"lr {float(lr):.2e}")
        parts.append(f"{_rate(item.get('tokens_per_second'))} tok/s")
        grad = item.get("gradient_norm")
        if isinstance(grad, (int, float)):
            clipped = " clipped" if item.get("gradient_clipped") else ""
            parts.append(f"grad {_number(grad, 3)}{clipped}")
        memory = item.get("peak_reserved_memory_bytes") or item.get("peak_memory_bytes")
        if isinstance(memory, (int, float)):
            parts.append(f"VRAM {_bytes(memory)}")
        overflows = item.get("overflow_events_total")
        if isinstance(overflows, int) and overflows:
            parts.append(f"overflows {overflows}")
        return " | ".join(parts)


def install_common_console(common: Any) -> None:
    """Replace only console presentation; raw evidence logs remain unchanged."""

    if getattr(common, "_small_llm_100m_console_installed", False):
        return

    original_environment = common.check_environment

    def informative_environment() -> dict[str, Any]:
        result = original_environment()
        print(f"[environment] {result.get('gpu', 'GPU detected')}", flush=True)
        return result

    def informative_run(
        command: Sequence[str],
        *,
        name: str,
        evidence: Path,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
        shown: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        del shown
        log_path = evidence / f"{name}.log"
        exit_path = evidence / f"{name}.exit-code"
        category, description = _stage(name)
        print(f"\n[{category}] {description}", flush=True)
        formatter = _Formatter(name, command)
        started = time.perf_counter()
        merged_env = os.environ.copy()
        if env:
            merged_env.update(env)
        tail: deque[str] = deque(maxlen=12)
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                list(command),
                cwd=str(cwd) if cwd else None,
                env=merged_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                tail.append(line.rstrip())
                rendered = formatter.render(line)
                if rendered:
                    print(rendered, flush=True)
            code = process.wait()
        elapsed = time.perf_counter() - started
        exit_path.write_text(f"{code}\n", encoding="utf-8")
        result = {
            "name": name,
            "exit_code": code,
            "elapsed_seconds": elapsed,
            "log": str(log_path),
            "log_sha256": common.sha256(log_path),
        }
        if code:
            print(f"[error] {description} failed; last log lines:", file=sys.stderr, flush=True)
            for line in tail:
                print(f"  {line}", file=sys.stderr, flush=True)
            raise common.GateFailure(f"{name} failed with exit code {code}; see {log_path}")
        print(f"[ok] {description} | {_seconds(elapsed)}", flush=True)
        return result

    common.check_environment = informative_environment
    common.run = informative_run
    common._small_llm_100m_console_installed = True


def install_experiment_console(experiment: Any) -> None:
    """Print the microbatch verdict once without changing its gate logic."""

    original = experiment.compare_probes

    def informative_compare(*args: Any, **kwargs: Any) -> dict[str, Any]:
        verdict = original(*args, **kwargs)
        print(
            "[probe] microbatch 4 accepted | "
            f"speedup {float(verdict['observed_median_speedup']):.2f}x | "
            f"max loss delta {float(verdict['maximum_step_loss_delta']):.4f} | "
            f"max grad delta {100 * float(verdict['maximum_gradient_relative_delta']):.2f}% | "
            f"VRAM {100 * float(verdict['candidate_reserved_memory_fraction']):.1f}%",
            flush=True,
        )
        return verdict

    experiment.compare_probes = informative_compare
