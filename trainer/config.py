"""Validated training-system configuration.

The values in this module are operational defaults for bounded qualification,
not frozen scientific hyperparameters. Every value is stored in checkpoints so
resume cannot silently change the optimizer or training recipe.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Literal

Precision = Literal["fp32", "fp16", "bf16"]
ScheduleKind = Literal["constant", "wsd"]
OptimizerKind = Literal["adamw", "hybrid_muon_adamw"]


@dataclass(frozen=True, slots=True)
class TrainerConfig:
    """Configuration for one deterministic single-process trainer."""

    optimizer: OptimizerKind = "adamw"
    microbatch_size: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    adam_epsilon: float = 1e-8
    muon_momentum: float = 0.95
    muon_lr_multiplier: float = 1.0
    muon_update_rms: float = 0.18
    muon_weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    precision: Precision = "fp16"
    schedule: ScheduleKind = "constant"
    warmup_tokens: int = 0
    stable_tokens: int = 0
    decay_tokens: int = 0
    minimum_lr_ratio: float = 0.1
    seed: int = 17
    max_overflow_retries: int = 3
    checkpoint_every_steps: int = 0
    evaluation_every_steps: int = 0
    log_every_steps: int = 1

    def __post_init__(self) -> None:
        integer_fields = (
            "microbatch_size",
            "warmup_tokens",
            "stable_tokens",
            "decay_tokens",
            "max_overflow_retries",
            "checkpoint_every_steps",
            "evaluation_every_steps",
            "log_every_steps",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.microbatch_size == 0:
            raise ValueError("microbatch_size must be positive")
        if self.log_every_steps == 0:
            raise ValueError("log_every_steps must be positive")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")

        positive_floats = (
            "learning_rate",
            "adam_epsilon",
            "muon_lr_multiplier",
            "muon_update_rms",
            "max_grad_norm",
        )
        for name in positive_floats:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a positive finite number")
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{name} must be a positive finite number")

        for name in ("weight_decay", "muon_weight_decay"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0
            ):
                raise ValueError(f"{name} must be a finite non-negative number")

        for name in ("beta1", "beta2", "muon_momentum"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0 <= float(value) < 1
            ):
                raise ValueError(f"{name} must be finite and in [0, 1)")

        if (
            isinstance(self.minimum_lr_ratio, bool)
            or not isinstance(self.minimum_lr_ratio, (int, float))
            or not math.isfinite(float(self.minimum_lr_ratio))
            or not 0 <= float(self.minimum_lr_ratio) <= 1
        ):
            raise ValueError("minimum_lr_ratio must be finite and in [0, 1]")
        if self.optimizer not in {"adamw", "hybrid_muon_adamw"}:
            raise ValueError("optimizer must be adamw or hybrid_muon_adamw")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be fp32, fp16, or bf16")
        if self.schedule not in {"constant", "wsd"}:
            raise ValueError("schedule must be constant or wsd")
        if self.schedule == "constant" and any(
            (self.warmup_tokens, self.stable_tokens, self.decay_tokens)
        ):
            raise ValueError("constant schedule cannot define warmup/stable/decay tokens")
        if self.schedule == "wsd" and self.decay_tokens <= 0:
            raise ValueError("wsd schedule requires a positive decay_tokens value")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


__all__ = ["OptimizerKind", "Precision", "ScheduleKind", "TrainerConfig"]
