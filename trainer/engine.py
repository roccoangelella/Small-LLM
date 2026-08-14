"""Single-device schema-v2 pretraining engine."""

from __future__ import annotations

import os
from pathlib import Path
import pickle
import random
from typing import Iterable, Mapping

import torch
from torch import nn
from torch.optim import Optimizer

from .config import TrainerConfig
from .evaluation import evaluate_batches, generate_token_ids
from .metrics import StepMetrics
from .optimizer import _classify_parameters, build_adamw, build_optimizer
from .optimizer_telemetry import InstrumentedHybridMuonAdamW
from .schedule import TokenLRScheduler
from .session import TrainingSession
from .state import (
    engine_state_dict,
    load_engine_checkpoint_state,
    load_engine_state,
    save_engine_checkpoint_state,
)
from .step import train_step
from .types import TokenBatch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _build_training_optimizer(model: nn.Module, config: TrainerConfig) -> Optimizer:
    """Build the selected optimizer, optionally omitting diagnostic telemetry.

    ``InstrumentedHybridMuonAdamW`` is scientifically equivalent to the base
    hybrid optimizer, but it clones parameter tensors to derive per-step update
    diagnostics.  Memory-constrained execution environments may opt out of that
    non-checkpointed telemetry without changing optimizer state or updates.
    """

    if config.optimizer == "hybrid_muon_adamw":
        if _env_flag("SMALL_LLM_DISABLE_OPTIMIZER_TELEMETRY"):
            return build_optimizer(model, config)
        return InstrumentedHybridMuonAdamW(_classify_parameters(model), config)
    return build_optimizer(model, config)


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
        self.optimizer = (
            optimizer
            if optimizer is not None
            else _build_training_optimizer(self.model, config)
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

    def _checkpoint_model(self) -> tuple[nn.Module, nn.Module]:
        wrapper = self.model
        raw = getattr(self, "_small_llm_raw_model", wrapper)
        return wrapper, raw

    def save_checkpoint_state(self, path: Path | str) -> None:
        """Write exact-resume state, streaming only the DDP production path."""

        checkpoint_path = Path(path)
        wrapper, raw = self._checkpoint_model()
        if raw is wrapper:
            # Preserve the established plain-pickle format for ordinary
            # single-device pretraining. The low-memory streamed format is an
            # execution adaptation for DDP where host headroom is constrained.
            state = dict(self.state_dict())
            with checkpoint_path.open("wb") as handle:
                pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.flush()
                os.fsync(handle.fileno())
            return
        self.model = raw
        try:
            save_engine_checkpoint_state(self, checkpoint_path)
        finally:
            self.model = wrapper

    def load_checkpoint_state(self, path: Path | str) -> None:
        """Load exact-resume state while keeping DDP wrappers out of model keys."""

        wrapper, raw = self._checkpoint_model()
        if raw is wrapper:
            load_engine_checkpoint_state(self, path)
            return
        self.model = raw
        try:
            load_engine_checkpoint_state(self, path)
        finally:
            self.model = wrapper


__all__ = [
    "StepMetrics",
    "TrainerEngine",
    "TrainingSession",
    "build_adamw",
    "build_optimizer",
    "generate_token_ids",
    "seed_everything",
]
