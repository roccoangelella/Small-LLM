"""Checkpoint-visible configuration for the MoE experiments.

Version 2 is the frozen M0 identity (8 experts, Top-1, experts that are exact dense-FFN
copies). Version 3 opens the two axes the owner fixed on 2026-09-09 — expert count and
Top-k — and decouples expert width from the dense FFN width, without touching any
version-2 checkpoint contract.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Literal

from model.config import ModelConfig

RouterScoring = Literal["softmax"]
RouterCombine = Literal["selected_softmax_probability", "normalised_topk_softmax"]
DispatchKind = Literal["dropless_grouped_by_expert", "dropless_padded_batched_gemm"]
BalancingKind = Literal["none", "loss_free_sign"]


@dataclass(frozen=True, slots=True)
class MoEModelConfig:
    """Complete checkpoint-visible identity for the first MoE architecture."""

    dense: ModelConfig
    num_experts: int = 8
    top_k: int = 1
    expert_d_ff: int = 1_408
    expert_type: str = "swiglu_dense_copy"  # or "swiglu" when width is independent
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
        if self.version not in {2, 3}:
            raise ValueError("unsupported MoE configuration version; expected 2 or 3")
        if self.router_scoring != "softmax":
            raise ValueError("unsupported MoE router scoring")
        if self.expert_type not in {"swiglu_dense_copy", "swiglu"}:
            raise ValueError("unsupported MoE expert type")
        if self.dispatch not in {"dropless_grouped_by_expert", "dropless_padded_batched_gemm"}:
            raise ValueError("unsupported MoE dispatch")
        if self.version == 2:
            # The frozen M0 identity. Nothing below may drift without a version bump.
            if self.router_combine != "selected_softmax_probability":
                raise ValueError("version 2 combines the selected softmax probability")
            if self.dispatch != "dropless_grouped_by_expert":
                raise ValueError("version 2 uses the per-expert grouped dispatch")
            if self.expert_type != "swiglu_dense_copy":
                raise ValueError("version 2 experts are exact dense-FFN-width copies")
            if self.num_experts != 8:
                raise ValueError("version 2 is frozen to 8 experts")
            if self.top_k != 1:
                raise ValueError("version 2 is frozen to Top-1 routing")
        else:
            if self.num_experts < 2:
                raise ValueError("a routed MoE needs at least two experts")
            if not 1 <= self.top_k <= self.num_experts:
                raise ValueError("top_k must be between 1 and num_experts")
            if self.dispatch not in {
                "dropless_padded_batched_gemm", "dropless_grouped_by_expert"
            }:
                raise ValueError("unsupported version 3 dispatch")
            expected_combine = (
                "selected_softmax_probability" if self.top_k == 1 else "normalised_topk_softmax"
            )
            if self.router_combine != expected_combine:
                raise ValueError(f"top_k={self.top_k} requires router_combine={expected_combine!r}")
            if self.expert_d_ff <= 0:
                raise ValueError("expert_d_ff must be positive")
        if self.expert_type == "swiglu_dense_copy" and self.expert_d_ff != self.dense.d_ff:
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
    def accepted(cls, **overrides: object) -> "MoEModelConfig":
        """The geometry the owner fixed on 2026-09-09: 64 experts, Top-2, no shared.

        Eight decoder layers in the frozen (gdn, gdn, gdn, mha) pattern give six GDN-2
        blocks and two gated-MHA blocks; width 256 with four 64-wide heads; SwiGLU experts
        of width 352; an 8,000-entry tied vocabulary. Counting these parameters must give
        144,025,496 stored and 9,938,840 active per token, which is the independent check
        against the architecture options document.
        """

        dense_fields = {
            "semantic_vocab_size": 8_000, "padded_vocab_size": 8_000, "max_seq_len": 2_048,
            # dense d_ff=704 is the reference the owner's granularity ADR matches
            # (K*h = 2*352 = 704); experts are decoupled from it via expert_type="swiglu".
            "d_model": 256, "n_layers": 8, "d_ff": 704, "n_heads": 4, "head_dim": 64,
            "gdn_num_key_heads": 4, "gdn_num_value_heads": 4,
            "gdn_key_dim": 64, "gdn_value_dim": 64,
            "gdn_conv_kernel_size": 4, "gdn_chunk_size": 32,
        }
        dense_fields.update(
            {k: v for k, v in overrides.items() if k in ModelConfig.__dataclass_fields__}
        )
        moe_fields: dict[str, object] = {
            "num_experts": 64, "top_k": 2, "expert_d_ff": 352, "expert_type": "swiglu",
            "router_combine": "normalised_topk_softmax",
            "dispatch": "dropless_padded_batched_gemm",
            "moe_layer_indices": tuple(range(int(dense_fields["n_layers"]))),
            "version": 3,
        }
        moe_fields.update(
            {k: v for k, v in overrides.items() if k not in ModelConfig.__dataclass_fields__}
        )
        return cls(dense=ModelConfig(**dense_fields), **moe_fields)  # type: ignore[arg-type]

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
