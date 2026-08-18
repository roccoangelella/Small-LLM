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
ScheduleKind = Literal["constant", "wsd", "wsqd"]
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
    schedule_anchor_tokens: int = 0
    cooldown_start_tokens: int = 0
    settle_tokens: int = 0
    settle_lr_ratio: float = 1.0
    base_power: float = 0.5
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
            "schedule_anchor_tokens",
            "cooldown_start_tokens",
            "settle_tokens",
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
            "base_power",
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
        if (
            isinstance(self.settle_lr_ratio, bool)
            or not isinstance(self.settle_lr_ratio, (int, float))
            or not math.isfinite(float(self.settle_lr_ratio))
            or not 0 < float(self.settle_lr_ratio) <= 1
        ):
            raise ValueError("settle_lr_ratio must be finite and in (0, 1]")
        if self.optimizer not in {"adamw", "hybrid_muon_adamw"}:
            raise ValueError("optimizer must be adamw or hybrid_muon_adamw")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be fp32, fp16, or bf16")
        if self.schedule not in {"constant", "wsd", "wsqd"}:
            raise ValueError("schedule must be constant, wsd, or wsqd")

        if self.schedule == "constant":
            if any(
                (
                    self.warmup_tokens,
                    self.stable_tokens,
                    self.decay_tokens,
                    self.schedule_anchor_tokens,
                    self.cooldown_start_tokens,
                    self.settle_tokens,
                )
            ) or self.settle_lr_ratio != 1.0 or self.base_power != 0.5:
                raise ValueError("constant schedule cannot define schedule token spans")
            return

        if self.schedule == "wsd":
            if self.decay_tokens <= 0:
                raise ValueError("wsd schedule requires a positive decay_tokens value")
            if (
                self.schedule_anchor_tokens
                or self.cooldown_start_tokens
                or self.settle_tokens
                or self.settle_lr_ratio != 1.0
                or self.base_power != 0.5
            ):
                raise ValueError("wsd schedule cannot define WSqD continuation parameters")
            return

        if self.warmup_tokens or self.stable_tokens:
            raise ValueError("wsqd continuation schedule does not use warmup/stable tokens")
        if self.schedule_anchor_tokens <= 0:
            raise ValueError("wsqd schedule requires positive schedule_anchor_tokens")
        if self.cooldown_start_tokens <= self.schedule_anchor_tokens:
            raise ValueError("wsqd cooldown_start_tokens must exceed its anchor")
        if self.decay_tokens <= 0:
            raise ValueError("wsqd schedule requires a positive decay_tokens value")
        if self.settle_tokens:
            settle_end = self.schedule_anchor_tokens + self.settle_tokens
            if settle_end >= self.cooldown_start_tokens:
                raise ValueError("wsqd settling phase must end before terminal cooldown")
            base_anchor = settle_end
            base_scale = float(self.settle_lr_ratio)
        else:
            if self.settle_lr_ratio != 1.0:
                raise ValueError("wsqd settle_lr_ratio requires positive settle_tokens")
            base_anchor = self.schedule_anchor_tokens
            base_scale = 1.0
        base_ratio_at_cooldown = base_scale * (
            base_anchor / self.cooldown_start_tokens
        ) ** float(self.base_power)
        if self.minimum_lr_ratio > base_ratio_at_cooldown:
            raise ValueError(
                "wsqd minimum_lr_ratio cannot exceed the base LR ratio at cooldown start"
            )

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        # These fields were introduced after the long-lived WSD checkpoints were
        # written. Omitting their defaults outside WSqD preserves exact identity
        # and resume compatibility for historical constant/WSD checkpoints.
        if self.schedule != "wsqd":
            payload.pop("schedule_anchor_tokens", None)
            payload.pop("cooldown_start_tokens", None)
            payload.pop("settle_tokens", None)
            payload.pop("settle_lr_ratio", None)
            payload.pop("base_power", None)
        else:
            if not self.settle_tokens:
                # Preserve the serialized identity of the first WSqD implementation.
                payload.pop("settle_tokens", None)
                payload.pop("settle_lr_ratio", None)
            if self.base_power == 0.5:
                # Preserve all pre-power-parameter WSqD checkpoint identities.
                payload.pop("base_power", None)
        return payload


__all__ = ["OptimizerKind", "Precision", "ScheduleKind", "TrainerConfig"]
