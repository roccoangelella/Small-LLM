"""Bounded smoke-training CLI for immutable schema-v2 shards."""
from __future__ import annotations
import json
import time
import torch
from .cli_args import parse_args
from .cli_setup import setup, validation_reader
from .remote_publication import configure_remote_publication


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    model_config, trainer_config, engine, session, coordinator = setup(args)
    remote = configure_remote_publication(args)
    validation: dict[str, object] | None = None
    saved: set[str] = set()
    remotely_published: set[str] = set()
    completed = 0

    def run_validation() -> dict[str, object]:
        reader = validation_reader(args, model_config)
        result = engine.evaluate(reader.iter_from_start(args.validation_blocks),
                                 maximum_batches=args.validation_blocks)
        print(json.dumps({"validation": result}, sort_keys=True), flush=True)
        return dict(result)

    def ensure_local_checkpoint(checkpoint_id: str) -> None:
        if checkpoint_id in saved:
            return
        if completed == 0 and args.resume == checkpoint_id:
            saved.add(checkpoint_id)
            return
        session.save_checkpoint(
            coordinator,
            checkpoint_id,
            validation_metrics=validation,
        )
        saved.add(checkpoint_id)

    def publish_remote_checkpoint(checkpoint_id: str, *, final: bool) -> None:
        if remote is None or checkpoint_id in remotely_published:
            return
        ensure_local_checkpoint(checkpoint_id)
        started = time.perf_counter()
        result = coordinator.publish(
            remote.publisher,
            checkpoint_id=checkpoint_id,
            drive_manifest=remote.drive_manifest,
        )
        elapsed = time.perf_counter() - started
        print(
            json.dumps(
                {
                    "remote_publication": {
                        "checkpoint_id": checkpoint_id,
                        "elapsed_seconds": elapsed,
                        "final": final,
                        "best_updated": bool(result.get("best_updated", False)),
                    }
                },
                sort_keys=True,
            ),
            flush=True,
        )
        remotely_published.add(checkpoint_id)

    for _ in range(args.steps):
        try:
            metrics = session.step()
        except StopIteration:
            break
        completed += 1
        print(json.dumps(metrics.as_dict(), sort_keys=True), flush=True)
        if (args.validation_blocks and trainer_config.evaluation_every_steps and
                engine.global_step % trainer_config.evaluation_every_steps == 0):
            validation = run_validation()
        checkpoint_id = f"step-{engine.global_step:08d}"
        if (trainer_config.checkpoint_every_steps and
                engine.global_step % trainer_config.checkpoint_every_steps == 0):
            ensure_local_checkpoint(checkpoint_id)
        if remote is not None and engine.global_step % remote.every_steps == 0:
            publish_remote_checkpoint(checkpoint_id, final=False)

    if args.validation_blocks and (validation is None or trainer_config.evaluation_every_steps == 0):
        validation = run_validation()
    checkpoint_id = f"step-{engine.global_step:08d}"
    ensure_local_checkpoint(checkpoint_id)
    publish_remote_checkpoint(checkpoint_id, final=True)
    print(json.dumps({"checkpoint_id": checkpoint_id}, sort_keys=True), flush=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
