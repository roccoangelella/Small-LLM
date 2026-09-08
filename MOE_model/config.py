"""Frozen configuration for the first 8-expert Top-1 MoE experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Literal

from model.config import ModelConfig

RouterScoring = Literal["softmax"]
RouterCombine = Literal["selected_softmax_probability"]
DispatchKind = Literal["dropless_grouped_by_expert"]
BalancingKind = Literal["none", "loss_free_sign"]


@dataclass(frozen=True, slots=True)
class MoEModelConfig:
    """Complete checkpoint-visible identity for the first MoE architecture."""

    dense: ModelConfig
    num_experts: int = 8
    top_k: int = 1
    expert_d_ff: int = 1_408
    expert_type: str = "swiglu_dense_copy"
    moe_layer_indices: tuple[int, ...] = tuple(range(20))
    router_scoring: RouterScoring = "softmax"
    router_combine: RouterCombine = "selected_softmax_probability"
    router_bias: bool = False
    router_fp32: bool = True
    router_init_std: float = 0.01
    router_jitter: float = 0.0
    router_z_loss_coefficient: float = 1e-4
    load_balancing: BalancingKind = "none"
    balancing_step_size: float = 0.0
    dispatch: DispatchKind = "dropless_grouped_by_expert"
    capacity_factor: None = None
    dropped_tokens_allowed: bool = False
    shared_expert: bool = False
    version: int = 2

    def __post_init__(self) -> None:
        if self.version != 2:
            raise ValueError("unsupported MoE configuration version; expected 2")
        if (
            self.router_scoring != "softmax"
            or self.router_combine != "selected_softmax_probability"
            or self.dispatch != "dropless_grouped_by_expert"
            or self.expert_type != "swiglu_dense_copy"
        ):
            raise ValueError("unsupported MoE scoring, combine, dispatch or expert type")
        if self.num_experts != 8:
            raise ValueError("first MoE experiment is frozen to 8 experts")
        if self.top_k != 1:
            raise ValueError("first MoE experiment is frozen to Top-1 routing")
        if self.expert_d_ff != self.dense.d_ff:
            raise ValueError("experts must be exact dense-FFN-width copies")
        if self.moe_layer_indices != tuple(range(self.dense.n_layers)):
            raise ValueError("first MoE experiment replaces the FFN in every decoder layer")
        if self.router_bias:
            raise ValueError("first MoE router is bias-free")
        if not self.router_fp32:
            raise ValueError("first MoE router must compute logits/probabilities in FP32")
        if self.router_jitter != 0.0:
            raise ValueError("first MoE experiment forbids routing jitter")
        if self.load_balancing not in {"none", "loss_free_sign"}:
            raise ValueError("unsupported load-balancing controller")
        if not math.isfinite(self.balancing_step_size) or self.balancing_step_size < 0:
            raise ValueError("balancing_step_size must be finite and non-negative")
        if self.load_balancing == "none" and self.balancing_step_size != 0:
            raise ValueError("M0 requires balancing_step_size=0")
        if self.capacity_factor is not None or self.dropped_tokens_allowed:
            raise ValueError("first MoE experiment is strictly dropless")
        if self.shared_expert:
            raise ValueError("first MoE experiment has no shared expert")
        if not math.isfinite(self.router_init_std) or self.router_init_std <= 0:
            raise ValueError("router_init_std must be positive")
        if not math.isfinite(self.router_z_loss_coefficient) or self.router_z_loss_coefficient <= 0:
            raise ValueError("router_z_loss_coefficient must be positive")

    @classmethod
    def substantive(cls, **dense_overrides: object) -> "MoEModelConfig":
        return cls(dense=ModelConfig.substantive(**dense_overrides))

    @classmethod
    def smoke(cls, **dense_overrides: object) -> "MoEModelConfig":
        dense = ModelConfig.smoke(**dense_overrides)
        return cls(
            dense=dense,
            expert_d_ff=dense.d_ff,
            moe_layer_indices=tuple(range(dense.n_layers)),
        )

    @property
    def semantic_vocab_size(self) -> int:
        return self.dense.semantic_vocab_size

    @property
    def padded_vocab_size(self) -> int:
        return self.dense.padded_vocab_size

    @property
    def max_seq_len(self) -> int:
        return self.dense.max_seq_len

    @property
    def d_model(self) -> int:
        return self.dense.d_model

    @property
    def n_layers(self) -> int:
        return self.dense.n_layers

    @property
    def d_ff(self) -> int:
        return self.dense.d_ff

    @property
    def architecture(self) -> str:
        return self.dense.architecture

    def as_dict(self) -> dict[str, object]:
        return asdict(self)
