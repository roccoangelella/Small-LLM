"""Exact, tie-aware parameter accounting for assembled models."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass

from torch import nn


_CATEGORIES = ("embeddings", "gdn_mixers", "mha_mixers", "ffn", "norms", "other")


@dataclass(frozen=True)
class ParameterCounts(Mapping[str, int]):
    """Unique parameter totals by architecture category."""

    embeddings: int = 0
    gdn_mixers: int = 0
    mha_mixers: int = 0
    ffn: int = 0
    norms: int = 0
    other: int = 0

    @property
    def total(self) -> int:
        return sum(getattr(self, category) for category in _CATEGORIES)

    @property
    def categories(self) -> dict[str, int]:
        return {category: getattr(self, category) for category in _CATEGORIES}

    def __getitem__(self, key: str) -> int:
        if key == "total":
            return self.total
        if key not in _CATEGORIES:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter((*_CATEGORIES, "total"))

    def __len__(self) -> int:
        return len(_CATEGORIES) + 1

    def as_dict(self) -> dict[str, int]:
        return {key: self[key] for key in self}


@dataclass(frozen=True)
class GDNExceptionCounts:
    """Reference-required GDN offsets, reported within ``gdn_mixers``."""

    a_log: int = 0
    dt_bias: int = 0
    output_gate_bias: int = 0

    @property
    def total(self) -> int:
        return self.a_log + self.dt_bias + self.output_gate_bias


def _module_parameter_ids(module: nn.Module | None) -> set[int]:
    if module is None:
        return set()
    return {id(parameter) for parameter in module.parameters()}


def count_parameters(model: nn.Module) -> ParameterCounts:
    """Count every storage once, including a tied embedding/output matrix once."""

    embedding = getattr(model, "token_embedding", None)
    embedding_ids = _module_parameter_ids(embedding)
    gdn_ids: set[int] = set()
    mha_ids: set[int] = set()
    ffn_ids: set[int] = set()
    norm_ids: set[int] = set()
    for block in getattr(model, "blocks", ()):
        mixer = getattr(block, "mixer", None)
        mixer_name = type(mixer).__name__.lower()
        if "gateddeltanet" in mixer_name or mixer_name.startswith("gdn"):
            gdn_ids.update(_module_parameter_ids(mixer))
        elif "attention" in mixer_name or "mha" in mixer_name:
            mha_ids.update(_module_parameter_ids(mixer))
        else:
            # Assembly-level classification remains useful for compatible
            # third-party mixer wrappers whose class name is not standardized.
            if "gdn" in str(mixer).lower():
                gdn_ids.update(_module_parameter_ids(mixer))
            else:
                mha_ids.update(_module_parameter_ids(mixer))
        ffn_ids.update(_module_parameter_ids(getattr(block, "ffn", None)))
        norm_ids.update(_module_parameter_ids(getattr(block, "mixer_norm", None)))
        norm_ids.update(_module_parameter_ids(getattr(block, "ffn_norm", None)))
    norm_ids.update(_module_parameter_ids(getattr(model, "final_norm", None)))

    totals = {category: 0 for category in _CATEGORIES}
    seen: set[int] = set()
    for parameter in model.parameters():
        identity = id(parameter)
        if identity in seen:
            continue
        seen.add(identity)
        if identity in embedding_ids:
            category = "embeddings"
        elif identity in gdn_ids:
            category = "gdn_mixers"
        elif identity in mha_ids:
            category = "mha_mixers"
        elif identity in ffn_ids:
            category = "ffn"
        elif identity in norm_ids:
            category = "norms"
        else:
            category = "other"
        totals[category] += parameter.numel()
    return ParameterCounts(**totals)


def gdn_exception_counts(model: nn.Module) -> GDNExceptionCounts:
    """Report the named GDN exception parameters without double counting."""

    totals = {"a_log": 0, "dt_bias": 0, "output_gate_bias": 0}
    for name, parameter in model.named_parameters():
        if name.endswith(".A_log"):
            totals["a_log"] += parameter.numel()
        elif name.endswith(".dt_bias"):
            totals["dt_bias"] += parameter.numel()
        elif name.endswith(".output_gate.1.bias"):
            totals["output_gate_bias"] += parameter.numel()
    return GDNExceptionCounts(**totals)


def optimizer_no_weight_decay_parameter_names(model: nn.Module) -> frozenset[str]:
    """Return explicit decay exclusions for norms and GDN dynamics/offsets."""

    excluded = set()
    for name, _ in model.named_parameters():
        if (
            ("norm" in name.lower() and name.endswith(".weight"))
            or name.endswith(".A_log")
            or name.endswith(".dt_bias")
            or name.endswith(".output_gate.1.bias")
        ):
            excluded.add(name)
    return frozenset(excluded)


parameter_counts = count_parameters
count_model_parameters = count_parameters


__all__ = [
    "GDNExceptionCounts",
    "ParameterCounts",
    "count_model_parameters",
    "count_parameters",
    "gdn_exception_counts",
    "optimizer_no_weight_decay_parameter_names",
    "parameter_counts",
]
