"""MoE-aware model loading for the canonical pretrained evaluation-v2 suite."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import torch

from model.config import ModelConfig
from trainer.state import load_trainer_state_file

from .config import MoEModelConfig
from .model import MoESmallLLM


def normalize_moe_model_config(raw: Mapping[str, object]) -> MoEModelConfig:
    values = dict(raw)
    dense_raw = values.pop("dense", None)
    if not isinstance(dense_raw, Mapping):
        raise RuntimeError("MoE model config must contain a dense ModelConfig mapping")
    dense_values = dict(dense_raw)
    pattern = dense_values.get("layer_pattern")
    if isinstance(pattern, list):
        dense_values["layer_pattern"] = tuple(pattern)
    layer_indices = values.get("moe_layer_indices")
    if isinstance(layer_indices, list):
        values["moe_layer_indices"] = tuple(int(index) for index in layer_indices)
    dense = ModelConfig(**dense_values)  # type: ignore[arg-type]
    return MoEModelConfig(dense=dense, **values)  # type: ignore[arg-type]


def _read_json_mapping(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as error:
        raise RuntimeError(f"MoE model config is not valid JSON: {path}") from error
    if not isinstance(payload, Mapping):
        raise RuntimeError("MoE model config JSON must contain an object")
    return dict(payload)


def load_moe_model(
    checkpoint_root: Path,
    *,
    device: torch.device,
    model_config_json: Path | None,
) -> tuple[MoESmallLLM, MoEModelConfig, Mapping[str, object]]:
    """Strict-load one self-describing Top-1 MoE checkpoint for evaluation."""

    state = load_trainer_state_file(
        checkpoint_root / "trainer_state.pkl",
        map_location="cpu",
    )
    if not isinstance(state, Mapping) or state.get("version") != 1:
        raise RuntimeError("trainer_state.pkl has an unsupported structure or version")
    model_state = state.get("model")
    if not isinstance(model_state, Mapping):
        raise RuntimeError("trainer_state.pkl has no model state mapping")

    if model_config_json is not None:
        raw_config = _read_json_mapping(model_config_json)
    else:
        raw = state.get("model_config")
        if not isinstance(raw, Mapping):
            raise RuntimeError("MoE checkpoint has no self-describing model_config")
        raw_config = dict(raw)

    config = normalize_moe_model_config(raw_config)
    model = MoESmallLLM(config)
    model.load_state_dict(model_state, strict=True)
    model.to(device)
    model.eval()
    return model, config, state


def main(argv: Sequence[str] | None = None) -> int:
    """Run the unchanged canonical eval-v2 suite with the MoE checkpoint loader."""

    from trainer import eval_suite_v2

    eval_suite_v2._load_model = load_moe_model
    from trainer import eval_entrypoint

    # eval_entrypoint holds the same imported eval_suite_v2 module object, so
    # only the architecture-specific checkpoint loader is replaced. Benchmark
    # definitions, prompt sets, decoding, and scoring remain unchanged.
    return eval_entrypoint.main(argv)


__all__ = ["load_moe_model", "main", "normalize_moe_model_config"]
