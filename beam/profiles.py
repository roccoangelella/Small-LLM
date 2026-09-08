"""Beam execution policy over the shared pretraining profile registry."""
from __future__ import annotations

from providers.profiles import (
    DEFAULT_PRECISION,
    DURABILITY_EVERY,
    LEGACY_WANDB_IDS,
    MODEL_PRESETS,
    SEQUENCES_PER_BLOCK,
    ModelPreset,
    ProfileRegistry,
    TokenPreset,
    canonical_run_id,
    format_quantity as _format_quantity,
    parse_quantity,
    run_name,
)

DEFAULT_GPU = "RTX5090"
MICROBATCH_CANDIDATES = (8, 12, 16)
SUPPORTED_GPUS = frozenset({"RTX5090", "RTX4090", "A10G"})

_REGISTRY = ProfileRegistry(volume_transport="beam_volume")
TOKEN_PRESETS = _REGISTRY.token_presets
resolve_presets = _REGISTRY.resolve_presets
resolve_profile_selection = _REGISTRY.resolve_selection

__all__ = [
    "DEFAULT_GPU",
    "DEFAULT_PRECISION",
    "DURABILITY_EVERY",
    "LEGACY_WANDB_IDS",
    "MICROBATCH_CANDIDATES",
    "MODEL_PRESETS",
    "SEQUENCES_PER_BLOCK",
    "SUPPORTED_GPUS",
    "TOKEN_PRESETS",
    "ModelPreset",
    "TokenPreset",
    "canonical_run_id",
    "parse_quantity",
    "resolve_presets",
    "resolve_profile_selection",
    "run_name",
]
