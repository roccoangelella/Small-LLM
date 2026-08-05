"""Portable model/optimizer/scheduler/scaler and RNG checkpoint state."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import math
import random
from typing import Mapping

import torch
from torch import Tensor
from torch.optim import Optimizer


def cpu_tree(value: object) -> object:
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    return value


def move_optimizer_state(optimizer: Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, Tensor):
                state[key] = value.to(device=device)


def _model_config_dict(engine: object) -> dict[str, object] | None:
    model = getattr(engine, "model", None)
    config = getattr(model, "config", None)
    if config is None:
        return None
    as_dict_method = getattr(config, "as_dict", None)
    if callable(as_dict_method):
        raw = as_dict_method()
    elif is_dataclass(config):
        raw = asdict(config)
    else:
        raise TypeError("model config must be a dataclass or expose as_dict()")
    if not isinstance(raw, Mapping):
        raise TypeError("model config serialization must return a mapping")
    return dict(raw)


def engine_state_dict(engine: object) -> dict[str, object]:
    state: dict[str, object] = {
        "version": 1,
        "config": engine.config.as_dict(),
        "model": cpu_tree(engine.model.state_dict()),
        "optimizer": cpu_tree(engine.optimizer.state_dict()),
        "scheduler": engine.scheduler.state_dict(),
        "scaler": engine.scaler.state_dict(),
        "global_step": engine.global_step,
        "consumed_tokens": engine.consumed_tokens,
        "overflow_events": engine.overflow_events,
        "best_validation_loss": engine.best_validation_loss,
        "python_rng_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state().clone(),
        "cuda_rng_states": (
            [state.clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else []
        ),
    }
    model_config = _model_config_dict(engine)
    if model_config is not None:
        state["model_config"] = model_config
    return state


def load_engine_state(engine: object, state: Mapping[str, object]) -> None:
    if state.get("version") != 1 or state.get("config") != engine.config.as_dict():
        raise ValueError("trainer checkpoint version or configuration mismatch")
    saved_model_config = state.get("model_config")
    current_model_config = _model_config_dict(engine)
    if saved_model_config is not None and saved_model_config != current_model_config:
        raise ValueError("trainer checkpoint model configuration mismatch")
    model_state, optimizer_state = state.get("model"), state.get("optimizer")
    scheduler_state, scaler_state = state.get("scheduler"), state.get("scaler")
    if not all(
        isinstance(item, Mapping)
        for item in (model_state, optimizer_state, scheduler_state, scaler_state)
    ):
        raise ValueError("trainer checkpoint has invalid component state")
    engine.model.load_state_dict(model_state, strict=True)
    engine.optimizer.load_state_dict(dict(optimizer_state))
    move_optimizer_state(engine.optimizer, engine.device)
    engine.scheduler.load_state_dict(scheduler_state)
    engine.scaler.load_state_dict(dict(scaler_state))
    for name in ("global_step", "consumed_tokens", "overflow_events"):
        value = state.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"trainer checkpoint has invalid {name}")
        setattr(engine, name, value)
    if engine.scheduler.committed_tokens != engine.consumed_tokens:
        raise ValueError("scheduler and trainer consumed-token counters disagree")
    best = state.get("best_validation_loss")
    if best is not None and (
        not isinstance(best, (int, float))
        or not math.isfinite(float(best))
        or best < 0
    ):
        raise ValueError("trainer checkpoint has invalid best validation loss")
    engine.best_validation_loss = None if best is None else float(best)
    if state.get("python_rng_state") is not None:
        random.setstate(state["python_rng_state"])
    torch_rng = state.get("torch_rng_state")
    if not isinstance(torch_rng, Tensor):
        raise ValueError("trainer checkpoint has invalid torch RNG state")
    torch.set_rng_state(torch_rng.cpu())
    cuda_states = state.get("cuda_rng_states", [])
    if torch.cuda.is_available():
        if not isinstance(cuda_states, list) or not all(
            isinstance(item, Tensor) for item in cuda_states
        ):
            raise ValueError("trainer checkpoint has invalid CUDA RNG state")
        if cuda_states:
            torch.cuda.set_rng_state_all([item.cpu() for item in cuda_states])
