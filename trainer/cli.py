"""Bounded smoke-training CLI for immutable schema-v2 shards."""
from __future__ import annotations
import json
import torch
from .cli_args import parse_args
from .cli_setup import setup, validation_reader

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    model_config, trainer_config, engine, session, coordinator = setup(args)
    validation: dict[str, object] | None = None
    saved: set[str] = set()
    completed = 0

    def run_validation() -> dict[str, object]:
        reader = validation_reader(args, model_config)
        result = engine.evaluate(reader.iter_from_start(args.validation_blocks),
                                 maximum_batches=args.validation_blocks)
        print(json.dumps({"validation": result}, sort_keys=True), flush=True)
        return dict(result)

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
        if (trainer_config.checkpoint_every_steps and
                engine.global_step % trainer_config.checkpoint_every_steps == 0):
            checkpoint_id = f"step-{engine.global_step:08d}"
            session.save_checkpoint(coordinator, checkpoint_id,
                                    validation_metrics=validation)
            saved.add(checkpoint_id)

    if args.validation_blocks and (validation is None or trainer_config.evaluation_every_steps == 0):
        validation = run_validation()
    checkpoint_id = f"step-{engine.global_step:08d}"
    if checkpoint_id not in saved and not (completed == 0 and args.resume == checkpoint_id):
        session.save_checkpoint(coordinator, checkpoint_id,
                                validation_metrics=validation)
    print(json.dumps({"checkpoint_id": checkpoint_id}, sort_keys=True), flush=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
