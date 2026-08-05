"""Bounded smoke-training CLI for immutable schema-v2 shards."""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Mapping

import torch

from .cli_args import parse_args
from .cli_setup import setup, validation_reader
from .remote_publication import configure_remote_publication
from .wandb_logging import configure_wandb


def _tree_byte_size(value: object) -> int | None:
    if not isinstance(value, (str, Path)):
        return None
    path = Path(value)
    if path.is_symlink() or not path.exists():
        return None
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _validation_metric(validation: Mapping[str, object] | None) -> float | None:
    """Return a higher-is-better remote metric from held-out loss."""

    if validation is None:
        return None
    loss = validation.get("loss")
    if isinstance(loss, bool) or not isinstance(loss, (int, float)):
        raise RuntimeError("validation result has no numeric loss for best-checkpoint selection")
    loss_value = float(loss)
    if loss_value < 0 or not math.isfinite(loss_value):
        raise RuntimeError("validation loss for best-checkpoint selection is invalid")
    return -loss_value


def _existing_remote_best_metric(remote: object | None) -> float | None:
    """Read the persisted best metric so resumed runs cannot overwrite it blindly."""

    if remote is None:
        return None
    publisher = getattr(remote, "publisher", None)
    store = getattr(publisher, "store", None)
    read_json = getattr(store, "read_json", None)
    drive_manifest = getattr(remote, "drive_manifest", None)
    if not callable(read_json) or not isinstance(drive_manifest, Mapping):
        return None
    run_id = drive_manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise RuntimeError("remote publication has no valid run_id")
    pointer = read_json(f"run/{run_id}/best.json")
    if pointer is None:
        return None
    if not isinstance(pointer, Mapping):
        raise RuntimeError("remote best.json is not a JSON object")
    metric = pointer.get("metric")
    if isinstance(metric, bool) or not isinstance(metric, (int, float)):
        raise RuntimeError("remote best.json has no numeric metric")
    value = float(metric)
    if not math.isfinite(value):
        raise RuntimeError("remote best.json metric is non-finite")
    return value


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    model_config, trainer_config, engine, session, coordinator = setup(args)
    remote = configure_remote_publication(args)
    best_remote_metric = _existing_remote_best_metric(remote)
    telemetry = configure_wandb(
        args,
        model_config=model_config,
        trainer_config=trainer_config,
        engine=engine,
    )
    validation: dict[str, object] | None = None
    saved: set[str] = set()
    remotely_published: set[str] = set()
    completed = 0

    def run_validation() -> dict[str, object]:
        started = time.perf_counter()
        reader = validation_reader(args, model_config)
        result = engine.evaluate(
            reader.iter_from_start(args.validation_blocks),
            maximum_batches=args.validation_blocks,
        )
        elapsed = time.perf_counter() - started
        print(
            json.dumps(
                {"validation": result, "elapsed_seconds": elapsed},
                sort_keys=True,
            ),
            flush=True,
        )
        if telemetry is not None:
            telemetry.log_validation(
                step=engine.global_step,
                metrics=result,
                elapsed_seconds=elapsed,
            )
        return dict(result)

    def ensure_local_checkpoint(checkpoint_id: str) -> None:
        if checkpoint_id in saved:
            return
        if completed == 0 and args.resume == checkpoint_id:
            saved.add(checkpoint_id)
            return
        started = time.perf_counter()
        checkpoint = session.save_checkpoint(
            coordinator,
            checkpoint_id,
            validation_metrics=validation,
        )
        elapsed = time.perf_counter() - started
        saved.add(checkpoint_id)
        event = {
            "checkpoint_id": checkpoint_id,
            "elapsed_seconds": elapsed,
            "byte_size": _tree_byte_size(checkpoint),
        }
        print(json.dumps({"local_checkpoint": event}, sort_keys=True), flush=True)
        if telemetry is not None:
            telemetry.log_checkpoint(
                step=engine.global_step,
                checkpoint_id=checkpoint_id,
                elapsed_seconds=elapsed,
                byte_size=event["byte_size"],
            )

    def publish_remote_checkpoint(checkpoint_id: str, *, final: bool) -> None:
        nonlocal best_remote_metric
        if remote is None or checkpoint_id in remotely_published:
            return
        ensure_local_checkpoint(checkpoint_id)
        metric = _validation_metric(validation)
        started = time.perf_counter()
        result = coordinator.publish(
            remote.publisher,
            checkpoint_id=checkpoint_id,
            drive_manifest=remote.drive_manifest,
            metric=metric,
            best_metric=best_remote_metric,
        )
        elapsed = time.perf_counter() - started
        best_updated = bool(result.get("best_updated", False))
        if best_updated:
            if metric is None:
                raise RuntimeError("remote publisher updated best without a validation metric")
            best_remote_metric = metric
        event = {
            "checkpoint_id": checkpoint_id,
            "elapsed_seconds": elapsed,
            "final": final,
            "best_updated": best_updated,
            "validation_loss": None if metric is None else -metric,
        }
        print(json.dumps({"remote_publication": event}, sort_keys=True), flush=True)
        if telemetry is not None:
            telemetry.log_remote_publication(
                step=engine.global_step,
                checkpoint_id=checkpoint_id,
                elapsed_seconds=elapsed,
                final=final,
            )
        remotely_published.add(checkpoint_id)

    try:
        for _ in range(args.steps):
            try:
                metrics = session.step()
            except StopIteration:
                break
            completed += 1
            print(json.dumps(metrics.as_dict(), sort_keys=True), flush=True)
            if telemetry is not None:
                telemetry.log_training(metrics)
            if (
                args.validation_blocks
                and trainer_config.evaluation_every_steps
                and engine.global_step % trainer_config.evaluation_every_steps == 0
            ):
                validation = run_validation()
            checkpoint_id = f"step-{engine.global_step:08d}"
            if (
                trainer_config.checkpoint_every_steps
                and engine.global_step % trainer_config.checkpoint_every_steps == 0
            ):
                ensure_local_checkpoint(checkpoint_id)
            if remote is not None and engine.global_step % remote.every_steps == 0:
                publish_remote_checkpoint(checkpoint_id, final=False)

        if args.validation_blocks and (
            validation is None
            or trainer_config.evaluation_every_steps == 0
            or engine.global_step % trainer_config.evaluation_every_steps != 0
        ):
            validation = run_validation()
        checkpoint_id = f"step-{engine.global_step:08d}"
        ensure_local_checkpoint(checkpoint_id)
        publish_remote_checkpoint(checkpoint_id, final=True)
        print(json.dumps({"checkpoint_id": checkpoint_id}, sort_keys=True), flush=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except BaseException:
        if telemetry is not None:
            try:
                telemetry.finish(exit_code=1)
            except Exception as telemetry_error:  # noqa: BLE001 - preserve the primary failure
                sys.stderr.write(
                    "W&B finalization also failed while handling the training error: "
                    f"{type(telemetry_error).__name__}: {telemetry_error}\n"
                )
        raise

    if telemetry is not None:
        telemetry.finish(exit_code=0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
