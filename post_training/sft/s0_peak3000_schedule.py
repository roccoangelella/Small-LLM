"""Step-anchored aggressive LR schedule for the 100M/2B 10% S0 rerun.

This experiment deliberately keeps the already-qualified optimizer and LR
landmarks while changing only schedule geometry: a short warmup, peak LR held
through update 3000, a fast settle, then a long calibrated power-law decay and
terminal cooldown.
"""
from __future__ import annotations

import math

from post_training.sft import config as sft_config
from trainer.fresh_decay import FreshAggressiveDecayPlan


S0_PEAK3000_WARMUP_STEPS = 64
S0_PEAK3000_PEAK_END_STEP = 3_000
S0_PEAK3000_SETTLE_STEPS = 128
S0_PEAK3000_COOLDOWN_FRACTION = 0.04


def build_s0_peak3000_trainer_config(
    schedule: sft_config.SFTSchedulePlan,
    *,
    microbatch_size: int = 1,
    precision: str = "fp16",
    seed: int = 17,
    learning_rate: float = sft_config.S0_AGGRESSIVE_PEAK_LR,
    checkpoint_every_steps: int = 0,
    evaluation_every_steps: int = 0,
):
    """Build the exact step-anchored schedule accepted for the 10% rerun."""

    if not math.isclose(
        float(learning_rate),
        sft_config.S0_AGGRESSIVE_PEAK_LR,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("peak-3000 S0 peak LR is frozen at 3e-5")

    counts = schedule.block_target_counts
    settle_end_step = S0_PEAK3000_PEAK_END_STEP + S0_PEAK3000_SETTLE_STEPS
    if len(counts) <= settle_end_step:
        raise ValueError(
            "peak-3000 S0 schedule requires more than "
            f"{settle_end_step} optimizer updates"
        )

    total_tokens = sum(counts)
    warmup_tokens = sum(counts[:S0_PEAK3000_WARMUP_STEPS])
    peak_tokens = sum(
        counts[S0_PEAK3000_WARMUP_STEPS:S0_PEAK3000_PEAK_END_STEP]
    )
    settle_tokens = sum(
        counts[S0_PEAK3000_PEAK_END_STEP:settle_end_step]
    )
    decay_tokens = max(1, math.floor(total_tokens * S0_PEAK3000_COOLDOWN_FRACTION))
    cooldown_start_tokens = total_tokens - decay_tokens
    settle_end_tokens = warmup_tokens + peak_tokens + settle_tokens
    if settle_end_tokens >= cooldown_start_tokens:
        raise ValueError("peak-3000 S0 schedule leaves no power-law decay span")

    settle_lr_ratio = (
        sft_config.S0_AGGRESSIVE_SETTLE_LR / sft_config.S0_AGGRESSIVE_PEAK_LR
    )
    cooldown_start_lr_ratio = (
        sft_config.S0_AGGRESSIVE_COOLDOWN_START_LR
        / sft_config.S0_AGGRESSIVE_PEAK_LR
    )
    final_lr_ratio = (
        sft_config.S0_AGGRESSIVE_FINAL_LR / sft_config.S0_AGGRESSIVE_PEAK_LR
    )
    base_power = math.log(settle_lr_ratio / cooldown_start_lr_ratio) / math.log(
        cooldown_start_tokens / settle_end_tokens
    )
    if not math.isfinite(base_power) or base_power <= 0:
        raise RuntimeError("peak-3000 S0 calibration produced an invalid power exponent")

    plan = FreshAggressiveDecayPlan(
        total_tokens=total_tokens,
        warmup_tokens=warmup_tokens,
        peak_tokens=peak_tokens,
        settle_tokens=settle_tokens,
        cooldown_start_tokens=cooldown_start_tokens,
        decay_tokens=decay_tokens,
        settle_lr_ratio=settle_lr_ratio,
        cooldown_start_lr_ratio=cooldown_start_lr_ratio,
        minimum_lr_ratio=final_lr_ratio,
        base_power=base_power,
    )

    landmarks = plan.lr_landmarks(learning_rate)
    expected = {
        "settle_lr": sft_config.S0_AGGRESSIVE_SETTLE_LR,
        "cooldown_start_lr": sft_config.S0_AGGRESSIVE_COOLDOWN_START_LR,
        "final_lr": sft_config.S0_AGGRESSIVE_FINAL_LR,
    }
    for name, value in expected.items():
        if not math.isclose(landmarks[name], value, rel_tol=0.0, abs_tol=1e-15):
            raise RuntimeError(
                f"peak-3000 S0 LR landmark drifted: {name}={landmarks[name]} expected={value}"
            )

    return sft_config._s0_optimizer_config(
        microbatch_size=microbatch_size,
        precision=precision,
        seed=seed,
        learning_rate=learning_rate,
        checkpoint_every_steps=checkpoint_every_steps,
        evaluation_every_steps=evaluation_every_steps,
        schedule_kwargs=plan.trainer_kwargs(),
    )


__all__ = [
    "S0_PEAK3000_COOLDOWN_FRACTION",
    "S0_PEAK3000_PEAK_END_STEP",
    "S0_PEAK3000_SETTLE_STEPS",
    "S0_PEAK3000_WARMUP_STEPS",
    "build_s0_peak3000_trainer_config",
]
