"""Portable model/optimizer/scheduler/scaler and RNG checkpoint state."""

from __future__ import annotations

import ctypes
from dataclasses import asdict, is_dataclass
import gc
import math
import pickle
from pathlib import Path
import random
from typing import Mapping
import zipfile

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


def release_host_memory() -> None:
    """Return collectable Python/glibc heap pages before memory-sensitive IO."""

    gc.collect()
    try:
        libc = ctypes.CDLL(None)
        malloc_trim = getattr(libc, "malloc_trim", None)
        if callable(malloc_trim):
            malloc_trim(0)
    except (AttributeError, OSError):
        pass


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


def engine_state_dict(engine: object, *, cpu: bool = True) -> dict[str, object]:
    """Return complete exact-resume state.

    The ordinary public state-dict path remains CPU-portable. Checkpoint writers
    may request device-native tensors so ``torch.save`` can stage one storage at
    a time instead of materializing a second full model+optimizer copy in host
    RAM before serialization.
    """

    model_state: object = engine.model.state_dict()
    optimizer_state: object = engine.optimizer.state_dict()
    if cpu:
        model_state = cpu_tree(model_state)
        optimizer_state = cpu_tree(optimizer_state)
    state: dict[str, object] = {
        "version": 1,
        "config": engine.config.as_dict(),
        "model": model_state,
        "optimizer": optimizer_state,
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


def _cpu_location(location: str | torch.device | None) -> bool:
    if location is None:
        return False
    try:
        return torch.device(location).type == "cpu"
    except (RuntimeError, TypeError):
        return False


def load_trainer_state_file(
    path: Path | str,
    *,
    map_location: str | torch.device | None = "cpu",
) -> dict[str, object]:
    """Load either the historical plain-pickle or streamed torch checkpoint format."""

    checkpoint_path = Path(path)
    if zipfile.is_zipfile(checkpoint_path):
        state = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=False,
            mmap=_cpu_location(map_location),
        )
    else:
        with checkpoint_path.open("rb") as handle:
            state = pickle.load(handle)
    if not isinstance(state, Mapping):
        raise RuntimeError("trainer checkpoint state is not a mapping")
    return dict(state)


def save_engine_checkpoint_state(engine: object, path: Path | str) -> None:
    """Serialize exact-resume state without a full host-side tensor clone."""

    checkpoint_path = Path(path)
    release_host_memory()
    if getattr(engine, "device", None) is not None and engine.device.type == "cuda":
        torch.cuda.synchronize(engine.device)
    state = engine_state_dict(engine, cpu=False)
    try:
        torch.save(
            state,
            checkpoint_path,
            pickle_protocol=pickle.HIGHEST_PROTOCOL,
        )
    finally:
        del state
        release_host_memory()


def load_engine_checkpoint_state(engine: object, path: Path | str) -> None:
    """Restore streamed or historical exact-resume state onto the live device."""

    state = load_trainer_state_file(path, map_location="cpu")
    try:
        load_engine_state(engine, state)
    finally:
        del state
        release_host_memory()


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


__all__ = [
    "cpu_tree",
    "engine_state_dict",
    "load_engine_checkpoint_state",
    "load_engine_state",
    "load_trainer_state_file",
    "move_optimizer_state",
    "release_host_memory",
    "save_engine_checkpoint_state",
]
