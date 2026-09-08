"""Provider-neutral pretraining profile registry.

Beam and Modal intentionally retain different GPU and execution-microbatch
policies.  Model, token, dataset, and run identities are shared here so those
scientific contracts cannot drift between providers.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

DEFAULT_PRECISION = "fp16"
SEQUENCES_PER_BLOCK = 64
DURABILITY_EVERY = 250

_QUANTITY = re.compile(r"^(\d+(?:\.\d+)?)([KMBT]?)$", re.IGNORECASE)
_PROFILE = re.compile(
    r"^(\d+(?:\.\d+)?[KMBT]?)[/-](\d+(?:\.\d+)?[KMBT]?)$",
    re.IGNORECASE,
)
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
    dataset_transport: str


MODEL_PRESETS: dict[int, ModelPreset] = {
    20_000_000: ModelPreset(20_000_000, "20M", "smoke"),
    100_000_000: ModelPreset(100_000_000, "100M", "substantive"),
    200_000_000: ModelPreset(200_000_000, "200M", "expanded"),
}
LEGACY_WANDB_IDS: dict[tuple[int, int], str] = {
    (20_000_000, 100_000_000): "20m-100m-data-004",
    (20_000_000, 500_000_000): "20m-500m-data-001",
    (20_000_000, 2_000_000_000): "20m-2b-data-001",
}


def parse_quantity(value: str) -> int:
    compact = value.strip().replace("_", "").replace(",", "").replace(" ", "")
    match = _QUANTITY.fullmatch(compact)
    if match is None:
        raise ValueError(
            f"invalid size {value!r}; use forms such as 20M, 100M, 200M, 10B, or 100B"
        )
    try:
        amount = Decimal(match.group(1)) * _MULTIPLIERS[match.group(2).upper()]
    except InvalidOperation as error:
        raise ValueError(f"invalid size {value!r}") from error
    if amount <= 0 or amount != amount.to_integral_value():
        raise ValueError(f"size must resolve to a positive whole number: {value!r}")
    return int(amount)


def format_quantity(value: int) -> str:
    for suffix, scale in (("T", 10**12), ("B", 10**9), ("M", 10**6), ("K", 10**3)):
        if value >= scale and value % scale == 0:
            return f"{value // scale}{suffix}"
    return str(value)


def canonical_run_id(model: ModelPreset, tokens: TokenPreset) -> str:
    return LEGACY_WANDB_IDS.get(
        (model.parameters, tokens.tokens),
        f"{model.label.lower()}-{tokens.label.lower()}-data-001",
    )


def run_name(model: ModelPreset, tokens: TokenPreset) -> str:
    return f"{model.label} model on {tokens.label} tokens"


class ProfileRegistry:
    """Shared profile resolver with one provider-local volume transport name."""

    def __init__(self, *, volume_transport: str) -> None:
        if not volume_transport:
            raise ValueError("provider volume transport cannot be empty")
        self.model_presets = MODEL_PRESETS
        self.token_presets: dict[int, TokenPreset] = {
            100_000_000: TokenPreset(100_000_000, "100M", "20m-100m", volume_transport),
            500_000_000: TokenPreset(500_000_000, "500M", "20m-500m", volume_transport),
            2_000_000_000: TokenPreset(
                2_000_000_000, "2B", "modal-2b-b64", volume_transport
            ),
            10_000_000_000: TokenPreset(
                10_000_000_000,
                "10B",
                "modal-10b-b64",
                "hf_rolling_shards",
            ),
            100_000_000_000: TokenPreset(
                100_000_000_000,
                "100B",
                "100b-b64",
                "hf_rolling_shards",
            ),
        }

    def resolve_presets(self, model: str, tokens: str) -> tuple[ModelPreset, TokenPreset]:
        model_value, token_value = parse_quantity(model), parse_quantity(tokens)
        try:
            model_preset = self.model_presets[model_value]
        except KeyError as error:
            supported = ", ".join(p.label for p in self.model_presets.values())
            raise ValueError(
                f"unsupported model {format_quantity(model_value)}; supported: {supported}"
            ) from error
        try:
            token_preset = self.token_presets[token_value]
        except KeyError as error:
            supported = ", ".join(p.label for p in self.token_presets.values())
            raise ValueError(
                f"unsupported token budget {format_quantity(token_value)}; supported: {supported}"
            ) from error
        if model_preset.label == "200M" and token_preset.label != "100B":
            raise ValueError("the 200M production preset is frozen to the 100B trajectory")
        if token_preset.label == "100B" and model_preset.label != "200M":
            raise ValueError("the active 100B production trajectory is frozen to the 200M model")
        return model_preset, token_preset

    def resolve_selection(
        self,
        *,
        profile: str | None,
        model: str | None,
        tokens: str | None,
    ) -> tuple[str, str]:
        """Resolve either compact ``MODEL-TOKENS`` syntax or legacy split flags."""

        compact_profile = (profile or "").strip()
        split_model = (model or "").strip()
        split_tokens = (tokens or "").strip()
        if compact_profile:
            if split_model or split_tokens:
                raise ValueError("use either --profile or --model/--tokens, not both")
            match = _PROFILE.fullmatch(compact_profile)
            if match is None:
                raise ValueError("profile must use MODEL-TOKENS form, for example 200M-100B")
            split_model, split_tokens = match.groups()
        elif not split_model or not split_tokens:
            raise ValueError("pass --profile or both --model and --tokens")

        model_preset, token_preset = self.resolve_presets(split_model, split_tokens)
        return model_preset.label, token_preset.label


__all__ = [
    "DEFAULT_PRECISION",
    "DURABILITY_EVERY",
    "LEGACY_WANDB_IDS",
    "MODEL_PRESETS",
    "ModelPreset",
    "ProfileRegistry",
    "SEQUENCES_PER_BLOCK",
    "TokenPreset",
    "canonical_run_id",
    "format_quantity",
    "parse_quantity",
    "run_name",
]
