"""Modal execution policy over the shared pretraining profile registry."""
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

DEFAULT_GPU = "H100"
MICROBATCH_CANDIDATES = (16, 32, 48, 64)
SUPPORTED_GPUS = frozenset(
    {
        "T4", "L4", "A10", "L40S", "A100", "A100-40GB", "A100-80GB",
        "RTX-PRO-6000", "H100", "H100!", "H200", "B200", "B200+", "B300",
    }
)

_REGISTRY = ProfileRegistry(volume_transport="modal_volume")
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
