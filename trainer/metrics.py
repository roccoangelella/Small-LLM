"""Inspectable optimizer-step metrics."""
from dataclasses import asdict, dataclass

@dataclass(frozen=True, slots=True)
class StepMetrics:
    step: int
    block_id: int
    loss: float
    learning_rate: float
    gradient_norm: float
    sequences: int
    target_tokens: int
    consumed_tokens: int
    elapsed_seconds: float
    tokens_per_second: float
    overflow_retries: int
    peak_memory_bytes: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)
