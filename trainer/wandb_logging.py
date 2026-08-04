"""Optional Weights & Biases telemetry for qualification runs."""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Mapping


def _plain(value: object) -> object:
    """Convert launch configuration to W&B-safe primitive containers."""

    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _flatten(prefix: str, value: Mapping[str, object]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, item in value.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(item, Mapping):
            result.update(_flatten(name, item))
        else:
            result[name] = _plain(item)
    return result


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    return commit if len(commit) == 40 else None


def _file_identity(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    result: dict[str, object] = {"path": str(path)}
    if path.is_file() and not path.is_symlink():
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        result["sha256"] = digest.hexdigest()
        result["byte_size"] = path.stat().st_size
    return result


@dataclass(slots=True)
class WandbTelemetry:
    """Small adapter that keeps W&B calls out of trainer mechanics."""

    run: Any

    def log_training(self, metrics: object) -> None:
        raw = dict(metrics.as_dict())
        step = int(raw.pop("step"))
        payload = {"trainer/global_step": step}
        payload.update(_flatten("train", raw))
        self.run.log(payload)

    def log_validation(
        self,
        *,
        step: int,
        metrics: Mapping[str, object],
        elapsed_seconds: float,
    ) -> None:
        payload = {
            "trainer/global_step": int(step),
            "validation/elapsed_seconds": float(elapsed_seconds),
        }
        payload.update(_flatten("validation", metrics))
        self.run.log(payload)

    def log_checkpoint(
        self,
        *,
        step: int,
        checkpoint_id: str,
        elapsed_seconds: float,
        byte_size: int | None,
    ) -> None:
        payload: dict[str, object] = {
            "trainer/global_step": int(step),
            "checkpoint/local_elapsed_seconds": float(elapsed_seconds),
            "checkpoint/local_id": checkpoint_id,
        }
        if byte_size is not None:
            payload["checkpoint/local_byte_size"] = int(byte_size)
        self.run.log(payload)

    def log_remote_publication(
        self,
        *,
        step: int,
        checkpoint_id: str,
        elapsed_seconds: float,
        final: bool,
    ) -> None:
        self.run.log(
            {
                "trainer/global_step": int(step),
                "checkpoint/remote_id": checkpoint_id,
                "checkpoint/remote_elapsed_seconds": float(elapsed_seconds),
                "checkpoint/remote_final": bool(final),
            }
        )

    def finish(self, *, exit_code: int) -> None:
        self.run.finish(exit_code=exit_code)


def configure_wandb(
    args: object,
    *,
    model_config: object,
    trainer_config: object,
) -> WandbTelemetry | None:
    """Start an opt-in W&B run without ever reading or logging the API key."""

    mode = str(args.wandb_mode)
    if mode == "disabled":
        return None
    try:
        import wandb
    except ImportError as error:
        raise SystemExit(
            "W&B telemetry requires the wandb package; launch with "
            "`uv run --with wandb==0.26.1 ...`"
        ) from error

    if args.wandb_resume != "never" and not args.wandb_run_id:
        raise SystemExit("--wandb-run-id is required when --wandb-resume is not 'never'")

    local_dir = Path(args.wandb_dir or (args.checkpoint_dir / "wandb"))
    local_dir.mkdir(parents=True, exist_ok=True)
    launch = {
        key: _plain(value)
        for key, value in vars(args).items()
        if key not in {"wandb_dir"}
    }
    config = {
        "launch": launch,
        "model": _plain(model_config),
        "trainer": _plain(trainer_config),
        "identity": {
            "git_commit": _git_commit(),
            "dataset_manifest": _file_identity(args.dataset_manifest),
            "remote_drive_manifest": _file_identity(args.remote_drive_manifest),
        },
    }
    init_kwargs: dict[str, object] = {
        "project": str(args.wandb_project),
        "entity": args.wandb_entity,
        "name": args.wandb_run_name,
        "mode": mode,
        "dir": str(local_dir),
        "config": config,
        "job_type": "pretraining-qualification",
        "tags": list(args.wandb_tags) if args.wandb_tags else None,
    }
    if args.wandb_run_id:
        init_kwargs["id"] = str(args.wandb_run_id)
    if args.wandb_resume != "never":
        init_kwargs["resume"] = str(args.wandb_resume)

    run = wandb.init(**init_kwargs)
    if run is None:
        raise RuntimeError("wandb.init did not return a run")
    run.define_metric("trainer/global_step")
    run.define_metric("*", step_metric="trainer/global_step")
    print(
        f"W&B telemetry enabled: project={args.wandb_project} "
        f"run_id={getattr(run, 'id', 'unknown')} mode={mode}",
        flush=True,
    )
    return WandbTelemetry(run)


__all__ = ["WandbTelemetry", "configure_wandb"]
