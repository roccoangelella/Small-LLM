"""Fresh-from-zero aggressive LR policy shared by pretraining and SFT.

The step-15,500 continuation was the experiment that calibrated the project
policy family.  Fresh runs do not inherit that continuation anchor: their
optimizer clock starts at zero, warms up briefly, settles immediately, follows
a long calibrated power law, and finishes with a short linear cooldown.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


FRESH_WARMUP_FRACTION = 0.05
FRESH_SETTLE_FRACTION = 0.03
FRESH_COOLDOWN_FRACTION = 0.04
SETTLE_LR_RATIO = 1.0 / 3.0
COOLDOWN_START_LR_RATIO = 1.0 / 30.0
FINAL_LR_RATIO = 1.0 / 60.0


@dataclass(frozen=True, slots=True)
class FreshAggressiveDecayPlan:
    total_tokens: int
    warmup_tokens: int
    settle_tokens: int
    cooldown_start_tokens: int
    decay_tokens: int
    settle_lr_ratio: float
    cooldown_start_lr_ratio: float
    minimum_lr_ratio: float
    base_power: float

    @property
    def schedule_anchor_tokens(self) -> int:
        return self.warmup_tokens

    @property
    def settle_end_tokens(self) -> int:
        return self.warmup_tokens + self.settle_tokens

    def trainer_kwargs(self) -> dict[str, object]:
        return {
            "schedule": "wsqd",
            "warmup_tokens": self.warmup_tokens,
            "stable_tokens": 0,
            "decay_tokens": self.decay_tokens,
            "minimum_lr_ratio": self.minimum_lr_ratio,
            "schedule_anchor_tokens": self.schedule_anchor_tokens,
            "cooldown_start_tokens": self.cooldown_start_tokens,
            "settle_tokens": self.settle_tokens,
            "settle_lr_ratio": self.settle_lr_ratio,
            "base_power": self.base_power,
        }

    def lr_landmarks(self, peak_lr: float) -> dict[str, float]:
        if not math.isfinite(float(peak_lr)) or peak_lr <= 0:
            raise ValueError("peak_lr must be positive and finite")
        return {
            "peak_lr": float(peak_lr),
            "settle_lr": float(peak_lr) * self.settle_lr_ratio,
            "cooldown_start_lr": float(peak_lr) * self.cooldown_start_lr_ratio,
            "final_lr": float(peak_lr) * self.minimum_lr_ratio,
        }


def fresh_aggressive_decay_plan(
    total_tokens: int,
    *,
    warmup_fraction: float = FRESH_WARMUP_FRACTION,
    settle_fraction: float = FRESH_SETTLE_FRACTION,
    cooldown_fraction: float = FRESH_COOLDOWN_FRACTION,
    settle_lr_ratio: float = SETTLE_LR_RATIO,
    cooldown_start_lr_ratio: float = COOLDOWN_START_LR_RATIO,
    final_lr_ratio: float = FINAL_LR_RATIO,
) -> FreshAggressiveDecayPlan:
    """Return the token-zero schedule geometry for one fresh training horizon."""

    if isinstance(total_tokens, bool) or not isinstance(total_tokens, int) or total_tokens <= 0:
        raise ValueError("total_tokens must be a positive integer")
    fractions = (warmup_fraction, settle_fraction, cooldown_fraction)
    if any(not math.isfinite(float(value)) or not 0 < float(value) < 1 for value in fractions):
        raise ValueError("fresh decay phase fractions must lie in (0, 1)")
    if sum(float(value) for value in fractions) >= 1:
        raise ValueError("fresh decay phases leave no power-law span")
    ratios = (settle_lr_ratio, cooldown_start_lr_ratio, final_lr_ratio)
    if any(not math.isfinite(float(value)) or not 0 < float(value) <= 1 for value in ratios):
        raise ValueError("fresh decay LR ratios must lie in (0, 1]")
    if not 1.0 > settle_lr_ratio > cooldown_start_lr_ratio > final_lr_ratio > 0.0:
        raise ValueError("fresh decay LR ratios must decrease peak -> settle -> cooldown -> final")

    warmup_tokens = max(1, math.floor(total_tokens * warmup_fraction))
    settle_tokens = max(1, math.floor(total_tokens * settle_fraction))
    decay_tokens = max(1, math.floor(total_tokens * cooldown_fraction))
    cooldown_start_tokens = total_tokens - decay_tokens
    settle_end_tokens = warmup_tokens + settle_tokens
    if settle_end_tokens >= cooldown_start_tokens:
        raise ValueError("fresh decay horizon is too short for settle and power-law phases")

    base_power = math.log(settle_lr_ratio / cooldown_start_lr_ratio) / math.log(
        cooldown_start_tokens / settle_end_tokens
    )
    if not math.isfinite(base_power) or base_power <= 0:
        raise RuntimeError("fresh decay calibration produced an invalid power exponent")

    return FreshAggressiveDecayPlan(
        total_tokens=total_tokens,
        warmup_tokens=warmup_tokens,
        settle_tokens=settle_tokens,
        cooldown_start_tokens=cooldown_start_tokens,
        decay_tokens=decay_tokens,
        settle_lr_ratio=float(settle_lr_ratio),
        cooldown_start_lr_ratio=float(cooldown_start_lr_ratio),
        minimum_lr_ratio=float(final_lr_ratio),
        base_power=base_power,
    )


__all__ = [
    "COOLDOWN_START_LR_RATIO",
    "FINAL_LR_RATIO",
    "FRESH_COOLDOWN_FRACTION",
    "FRESH_SETTLE_FRACTION",
    "FRESH_WARMUP_FRACTION",
    "FreshAggressiveDecayPlan",
    "SETTLE_LR_RATIO",
    "fresh_aggressive_decay_plan",
]
