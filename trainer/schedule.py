"""Token-count learning-rate schedules."""

from __future__ import annotations

import math
from typing import Mapping

from torch.optim import Optimizer

from .config import TrainerConfig


class TokenLRScheduler:
    """Set LR from committed non-padding target tokens, not dataloader steps."""

    VERSION = 1

    def __init__(self, optimizer: Optimizer, config: TrainerConfig) -> None:
        self.optimizer = optimizer
        self.config = config
        self.committed_tokens = 0
        self.last_lr = self._lr_at(0)
        self._set_lr(self.last_lr)

    def _lr_at(self, tokens: int) -> float:
        peak = float(self.config.learning_rate)
        if self.config.schedule == "constant":
            return peak

        if self.config.schedule == "wsd":
            warmup = self.config.warmup_tokens
            stable_end = warmup + self.config.stable_tokens
            decay_end = stable_end + self.config.decay_tokens
            if warmup and tokens < warmup:
                return peak * max(tokens, 1) / warmup
            if tokens <= stable_end:
                return peak
            progress = min(1.0, max(0.0, (tokens - stable_end) / self.config.decay_tokens))
            minimum = peak * self.config.minimum_lr_ratio
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            lr = minimum + (peak - minimum) * cosine
            return minimum if tokens >= decay_end else lr

        # Continuation-oriented WSqD variant. The exact fork point is an LR
        # anchor rather than a new warmup. An optional short cosine settling
        # phase can reduce LR quickly before the long inverse-sqrt base. The
        # terminal cooldown remains linear to the configured nonzero floor.
        anchor = self.config.schedule_anchor_tokens
        cooldown_start = self.config.cooldown_start_tokens
        decay_end = cooldown_start + self.config.decay_tokens
        if tokens <= anchor:
            return peak

        settle_tokens = self.config.settle_tokens
        if settle_tokens:
            settle_end = anchor + settle_tokens
            settle_lr = peak * self.config.settle_lr_ratio
            if tokens <= settle_end:
                progress = min(1.0, max(0.0, (tokens - anchor) / settle_tokens))
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                return settle_lr + (peak - settle_lr) * cosine
            base_anchor = settle_end
            base_peak = settle_lr
        else:
            base_anchor = anchor
            base_peak = peak

        base = base_peak * math.sqrt(base_anchor / tokens)
        if tokens <= cooldown_start:
            return base
        cooldown_base = base_peak * math.sqrt(base_anchor / cooldown_start)
        minimum = peak * self.config.minimum_lr_ratio
        progress = min(
            1.0,
            max(0.0, (tokens - cooldown_start) / self.config.decay_tokens),
        )
        lr = minimum + (cooldown_base - minimum) * (1.0 - progress)
        return minimum if tokens >= decay_end else lr

    def _set_lr(self, value: float) -> None:
        for group in self.optimizer.param_groups:
            scale = group.get("lr_scale", 1.0)
            if isinstance(scale, bool) or not isinstance(scale, (int, float)) or scale <= 0:
                raise ValueError("optimizer lr_scale must be a positive number")
            group["lr"] = value * float(scale)

    def prepare_step(self, next_committed_tokens: int) -> float:
        """Set the base LR for a candidate step without committing schedule state."""

        if next_committed_tokens <= self.committed_tokens:
            raise ValueError("next committed token count must advance")
        value = self._lr_at(next_committed_tokens)
        self._set_lr(value)
        return value

    def commit(self, committed_tokens: int) -> float:
        if committed_tokens <= self.committed_tokens:
            raise ValueError("committed token count must advance")
        self.committed_tokens = committed_tokens
        self.last_lr = self._lr_at(committed_tokens)
        self._set_lr(self.last_lr)
        return self.last_lr

    def state_dict(self) -> dict[str, object]:
        return {
            "version": self.VERSION,
            "config": self.config.as_dict(),
            "committed_tokens": self.committed_tokens,
            "last_lr": self.last_lr,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if state.get("version") != self.VERSION:
            raise ValueError("unsupported token scheduler state version")
        if state.get("config") != self.config.as_dict():
            raise ValueError("scheduler configuration does not match this trainer")
        tokens = state.get("committed_tokens")
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
            raise ValueError("scheduler committed token count is invalid")
        expected = self._lr_at(tokens)
        stored = state.get("last_lr")
        if not isinstance(stored, (int, float)) or not math.isclose(
            float(stored), expected, rel_tol=1e-12, abs_tol=0.0
        ):
            raise ValueError("scheduler state LR does not match its token count")
        self.committed_tokens = tokens
        self.last_lr = expected
        self._set_lr(expected)


__all__ = ["TokenLRScheduler"]
