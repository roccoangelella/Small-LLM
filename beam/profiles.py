"""Pure profile resolution for the Beam training adapter."""
from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

DEFAULT_GPU = "RTX5090"
DEFAULT_PRECISION = "fp16"
SEQUENCES_PER_BLOCK = 64
DURABILITY_EVERY = 250
MICROBATCH_CANDIDATES = (8, 12, 16)

_QUANTITY = re.compile(r"^(\d+(?:\.\d+)?)([KMBT]?)$", re.IGNORECASE)
_MULTIPLIERS = {
    "": Decimal(1),
    "K": Decimal(1_000),
    "M": Decimal(1_000_000),
    "B": Decimal(1_000_000_000),
    "T": Decimal(1_000_000_000_000),
}


@dataclass(frozen=True, slots=True)
class ModelPreset:
    parameters: int
    label: str
    trainer_size: str


@dataclass(frozen=True, slots=True)
class TokenPreset:
    tokens: int
    label: str
    dataset_profile: str
    dataset_transport: str = "beam_volume"


MODEL_PRESETS: dict[int, ModelPreset] = {
    20_000_000: ModelPreset(20_000_000, "20M", "smoke"),
    100_000_000: ModelPreset(100_000_000, "100M", "substantive"),
}
TOKEN_PRESETS: dict[int, TokenPreset] = {
    100_000_000: TokenPreset(100_000_000, "100M", "20m-100m"),
    500_000_000: TokenPreset(500_000_000, "500M", "20m-500m"),
    2_000_000_000: TokenPreset(2_000_000_000, "2B", "modal-2b-b64"),
    10_000_000_000: TokenPreset(
        10_000_000_000,
        "10B",
        "modal-10b-b64",
        "hf_rolling_shards",
    ),
}
LEGACY_WANDB_IDS: dict[tuple[int, int], str] = {
    (20_000_000, 100_000_000): "20m-100m-data-004",
    (20_000_000, 500_000_000): "20m-500m-data-001",
    (20_000_000, 2_000_000_000): "20m-2b-data-001",
}
SUPPORTED_GPUS = frozenset({"RTX5090", "RTX4090", "A10G"})


def parse_quantity(value: str) -> int:
    compact = value.strip().replace("_", "").replace(",", "").replace(" ", "")
    match = _QUANTITY.fullmatch(compact)
    if match is None:
        raise ValueError(f"invalid size {value!r}; use forms such as 20M, 100M, 2B, or 10B")
    try:
        amount = Decimal(match.group(1)) * _MULTIPLIERS[match.group(2).upper()]
    except InvalidOperation as error:
        raise ValueError(f"invalid size {value!r}") from error
    if amount <= 0 or amount != amount.to_integral_value():
        raise ValueError(f"size must resolve to a positive whole number: {value!r}")
    return int(amount)


def _format_quantity(value: int) -> str:
    for suffix, scale in (("T", 10**12), ("B", 10**9), ("M", 10**6), ("K", 10**3)):
        if value >= scale and value % scale == 0:
            return f"{value // scale}{suffix}"
    return str(value)


def resolve_presets(model: str, tokens: str) -> tuple[ModelPreset, TokenPreset]:
    model_value, token_value = parse_quantity(model), parse_quantity(tokens)
    try:
        model_preset = MODEL_PRESETS[model_value]
    except KeyError as error:
        supported = ", ".join(p.label for p in MODEL_PRESETS.values())
        raise ValueError(f"unsupported model {_format_quantity(model_value)}; supported: {supported}") from error
    try:
        token_preset = TOKEN_PRESETS[token_value]
    except KeyError as error:
        supported = ", ".join(p.label for p in TOKEN_PRESETS.values())
        raise ValueError(f"unsupported token budget {_format_quantity(token_value)}; supported: {supported}") from error
    return model_preset, token_preset


def canonical_run_id(model: ModelPreset, tokens: TokenPreset) -> str:
    return LEGACY_WANDB_IDS.get(
        (model.parameters, tokens.tokens),
        f"{model.label.lower()}-{tokens.label.lower()}-data-001",
    )


def run_name(model: ModelPreset, tokens: TokenPreset) -> str:
    return f"{model.label} model on {tokens.label} tokens"
