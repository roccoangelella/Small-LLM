"""Inspectable optimizer-step metrics."""

from dataclasses import asdict, dataclass, field


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
    grad_scaler_scale: float = 1.0
    gradient_clipped: bool = False
    overflow_events_total: int = 0
    peak_reserved_memory_bytes: int = 0
    data_wait_seconds: float = 0.0
    optimizer_gradient_norms: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)
