"""Scalable configuration and exact target-token planning for SFT."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Mapping

from trainer.config import TrainerConfig

DEFAULT_INSTRUCTION_SOURCE_SHARES = {
    "smol-magpie-ultra-short": 0.75,
    "smol-contraints": 0.10,
    "smollm-rewrite-30k": 0.075,
    "smol-summarize-20k": 0.075,
}


def _validate_share(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite share")
    if not math.isfinite(float(value)) or float(value) < 0:
        raise ValueError(f"{name} must be a finite non-negative share")


@dataclass(frozen=True, slots=True)
class SFTDataConfig:
    """All dataset size and geometry values are data, never code constants."""

    target_loss_tokens: int = 4_000_000
    optimizer_target_tokens: int = 32_768
    context_length: int = 2_048
    maximum_assistant_tokens: int = 512
    eos_token_id: int = 50_256
    instruction_share: float = 0.85
    replay_share: float = 0.15
    instruction_source_shares: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_INSTRUCTION_SOURCE_SHARES)
    )
    shuffle_buffer_records: int = 4_096
    shard_target_bytes: int = 64 * 1024 * 1024
    seed: int = 17

    def __post_init__(self) -> None:
        for name in (
            "target_loss_tokens",
            "optimizer_target_tokens",
            "context_length",
            "maximum_assistant_tokens",
            "shuffle_buffer_records",
            "shard_target_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not 0 <= self.eos_token_id <= 65_535:
            raise ValueError("eos_token_id must be uint16-compatible")
        if self.maximum_assistant_tokens > self.context_length:
            raise ValueError("maximum_assistant_tokens cannot exceed context_length")

        _validate_share("instruction_share", self.instruction_share)
        _validate_share("replay_share", self.replay_share)
        if not math.isclose(
            self.instruction_share + self.replay_share,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("instruction_share and replay_share must sum to one")
        if not self.instruction_source_shares:
            raise ValueError("instruction_source_shares cannot be empty")
        for source, share in self.instruction_source_shares.items():
            if not source:
                raise ValueError("instruction source names must be non-empty")
            _validate_share(f"instruction_source_shares[{source!r}]", share)
        if not math.isclose(
            sum(float(value) for value in self.instruction_source_shares.values()),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("instruction_source_shares must sum to one")

    @property
    def complete_source_shares(self) -> dict[str, float]:
        result = {
            source: self.instruction_share * float(share)
            for source, share in self.instruction_source_shares.items()
        }
        result["climbmix-replay"] = self.replay_share
        return result

    def as_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["instruction_source_shares"] = dict(self.instruction_source_shares)
        result["complete_source_shares"] = self.complete_source_shares
        return result


@dataclass(frozen=True, slots=True)
class SFTSchedulePlan:
    block_target_counts: tuple[int, ...]
    warmup_tokens: int
    stable_tokens: int
    decay_tokens: int

    @classmethod
    def from_block_target_counts(
        cls,
        block_target_counts: tuple[int, ...],
        *,
        minimum_warmup_updates: int = 16,
        warmup_fraction: float = 0.05,
        decay_fraction: float = 0.20,
    ) -> "SFTSchedulePlan":
        if not block_target_counts or any(value <= 0 for value in block_target_counts):
            raise ValueError("block_target_counts must be positive and non-empty")
        steps = len(block_target_counts)
        warmup_steps = min(
            steps,
            max(minimum_warmup_updates, math.ceil(steps * warmup_fraction)),
        )
        remaining = steps - warmup_steps
        decay_steps = min(remaining, math.ceil(steps * decay_fraction))
        stable_steps = steps - warmup_steps - decay_steps

        warmup_tokens = sum(block_target_counts[:warmup_steps])
        stable_tokens = sum(
            block_target_counts[warmup_steps : warmup_steps + stable_steps]
        )
        decay_tokens = sum(block_target_counts[-decay_steps:]) if decay_steps else 0
        if decay_tokens <= 0:
            decay_tokens = block_target_counts[-1]
            if stable_tokens >= decay_tokens:
                stable_tokens -= decay_tokens
            elif warmup_tokens > decay_tokens:
                warmup_tokens -= decay_tokens
        if warmup_tokens + stable_tokens + decay_tokens != sum(block_target_counts):
            raise RuntimeError("SFT schedule plan does not cover every target token")
        return cls(
            block_target_counts=block_target_counts,
            warmup_tokens=warmup_tokens,
            stable_tokens=stable_tokens,
            decay_tokens=decay_tokens,
        )


def build_s0_trainer_config(
    schedule: SFTSchedulePlan,
    *,
    microbatch_size: int = 1,
    precision: str = "fp16",
    seed: int = 17,
    learning_rate: float = 3e-5,
) -> TrainerConfig:
    """Preserve the pretraining optimizer/scheduler mechanics with S0 values."""

    return TrainerConfig(
        optimizer="hybrid_muon_adamw",
        microbatch_size=microbatch_size,
        learning_rate=learning_rate,
        weight_decay=0.0,
        beta1=0.9,
        beta2=0.95,
        adam_epsilon=1e-8,
        muon_momentum=0.95,
        muon_lr_multiplier=1.0,
        muon_update_rms=0.18,
        muon_weight_decay=0.0,
        max_grad_norm=1.0,
        precision=precision,  # type: ignore[arg-type]
        schedule="wsd",
        warmup_tokens=schedule.warmup_tokens,
        stable_tokens=schedule.stable_tokens,
        decay_tokens=schedule.decay_tokens,
        minimum_lr_ratio=0.1,
        seed=seed,
    )


__all__ = [
    "DEFAULT_INSTRUCTION_SOURCE_SHARES",
    "SFTDataConfig",
    "SFTSchedulePlan",
    "build_s0_trainer_config",
]
