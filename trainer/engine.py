"""Single-device schema-v2 pretraining engine."""

from __future__ import annotations

import random
from typing import Iterable, Mapping

import torch
from torch import nn
from torch.optim import Optimizer

from .config import TrainerConfig
from .evaluation import evaluate_batches, generate_token_ids
from .metrics import StepMetrics
from .optimizer import build_adamw, build_optimizer
from .schedule import TokenLRScheduler
from .session import TrainingSession
from .state import engine_state_dict, load_engine_state
from .step import train_step
from .types import TokenBatch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TrainerEngine:
    """Train one prepared block as one atomic optimizer update."""

    STATE_VERSION = 1

    def __init__(
        self,
        model: nn.Module,
        config: TrainerConfig,
        *,
        device: str | torch.device | None = None,
        optimizer: Optimizer | None = None,
    ) -> None:
        self.config = config
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if config.precision == "fp16" and self.device.type != "cuda":
            raise ValueError("fp16 training requires a CUDA device")
        if config.precision == "bf16" and self.device.type not in {"cuda", "cpu"}:
            raise ValueError("bf16 training requires a CUDA or CPU device")
        self.model = model.to(self.device)
        self.optimizer = optimizer if optimizer is not None else build_optimizer(
            self.model, config
        )
        self.scheduler = TokenLRScheduler(self.optimizer, config)
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=config.precision == "fp16"
        )
        self.global_step = self.consumed_tokens = self.overflow_events = 0
        self.best_validation_loss: float | None = None

    def train_batch(self, batch: TokenBatch) -> StepMetrics:
        return train_step(self, batch)

    def evaluate(
        self,
        batches: Iterable[TokenBatch],
        *,
        maximum_batches: int | None = None,
    ) -> dict[str, float | int]:
        return evaluate_batches(self, batches, maximum_batches=maximum_batches)

    def state_dict(self) -> dict[str, object]:
        return engine_state_dict(self)

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        load_engine_state(self, state)


__all__ = [
    "StepMetrics",
    "TrainerEngine",
    "TrainingSession",
    "build_adamw",
    "build_optimizer",
    "generate_token_ids",
    "seed_everything",
]
