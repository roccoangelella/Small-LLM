"""Validated training-system configuration.

The values in this module are operational defaults for smoke qualification, not
frozen scientific hyperparameters.  The training policy remains explicit in
checkpoints so later experiments cannot silently resume with changed settings.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Literal

Precision = Literal["fp32", "fp16", "bf16"]
ScheduleKind = Literal["constant", "wsd"]


@dataclass(frozen=True, slots=True)
class TrainerConfig:
    """Configuration for one deterministic single-process trainer."""

    microbatch_size: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    adam_epsilon: float = 1e-8
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
            "max_grad_norm",
        )
        for name in positive_floats:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a positive finite number")
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{name} must be a positive finite number")
        if (
            isinstance(self.weight_decay, bool)
            or not isinstance(self.weight_decay, (int, float))
            or not math.isfinite(float(self.weight_decay))
            or self.weight_decay < 0
        ):
            raise ValueError("weight_decay must be a finite non-negative number")
        for name in ("beta1", "beta2"):
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


__all__ = ["Precision", "ScheduleKind", "TrainerConfig"]
