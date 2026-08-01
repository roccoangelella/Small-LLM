"""Bind a trainer and block source at completed optimizer boundaries."""
from __future__ import annotations
from typing import Callable, Mapping, Protocol
from .metrics import StepMetrics
from .types import BatchSource

class CheckpointCoordinatorLike(Protocol):
    def save(self, *, checkpoint_id: str, trainer: object,
             pipeline_state: Mapping[str, object], optimizer_step_complete: bool,
             validation_metrics: Mapping[str, object] | None = None) -> object: ...
    def load(self, checkpoint_id: str, trainer: object) -> dict[str, object]: ...

class TrainingSession:
    def __init__(self, engine: object, source: BatchSource) -> None:
        self.engine, self.source = engine, source

    def step(self, timeout: float | None = None) -> StepMetrics:
        batch = self.source.next_batch(timeout=timeout)
        metrics = self.engine.train_batch(batch)
        self.source.acknowledge(batch.block_id)
        return metrics

    def save_checkpoint(self, coordinator: CheckpointCoordinatorLike, checkpoint_id: str, *,
                        pipeline_state: Mapping[str, object] | None = None,
                        validation_metrics: Mapping[str, object] | None = None) -> object:
        source_state = self.source.pipeline_state()
        if pipeline_state is None:
            payload = source_state
        else:
            payload = dict(pipeline_state)
            consumed = source_state["last_consumed_block_id"]
            if payload.get("last_consumed_block_id", consumed) != consumed:
                raise RuntimeError("dataset and trainer consumed-block cursors disagree")
            payload["last_consumed_block_id"] = consumed
            payload["gradient_accumulation_position"] = 0
            payload["consumer"] = source_state.get("consumer")
        return coordinator.save(checkpoint_id=checkpoint_id, trainer=self.engine,
            pipeline_state=payload, optimizer_step_complete=True,
            validation_metrics=validation_metrics)

    def load_checkpoint(self, coordinator: CheckpointCoordinatorLike,
                        checkpoint_id: str) -> dict[str, object]:
        pipeline_state = coordinator.load(checkpoint_id, self.engine)
        self.source.load_pipeline_state(pipeline_state)
        return pipeline_state

    def run(self, maximum_steps: int, *,
            on_step: Callable[[StepMetrics], None] | None = None) -> list[StepMetrics]:
        if maximum_steps <= 0:
            raise ValueError("maximum_steps must be positive")
        results: list[StepMetrics] = []
        for _ in range(maximum_steps):
            try:
                metrics = self.step()
            except StopIteration:
                break
            results.append(metrics)
            if on_step is not None:
                on_step(metrics)
        return results
