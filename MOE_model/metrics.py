"""MoE-specific extension of the standard training-step metrics."""

from __future__ import annotations

from dataclasses import dataclass, field

from trainer.metrics import StepMetrics


@dataclass(frozen=True, slots=True)
class MoEStepMetrics(StepMetrics):
    objective_loss: float = 0.0
    router_z_loss: float = 0.0
    router_z_loss_coefficient: float = 1e-4
    moe: dict[str, object] = field(default_factory=dict)


__all__ = ["MoEStepMetrics"]
