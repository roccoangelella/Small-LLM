"""8-expert, dropless, Top-1 MoE variant of the Small-LLM architecture."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import Tensor, nn

from model.components import GatedMultiheadAttention, RMSNorm, SwiGLU, TiedEmbedding
from model.gdn2 import GDN2Cache
from model.gdn2_stable import StableGatedDeltaNet2 as GatedDeltaNet2

from .config import MoEModelConfig
from .router import SwitchTop1Router


@dataclass(frozen=True, slots=True)
class LayerMoETelemetry:
    expert_counts: Tensor
    selected_probability_sum: Tensor
    entropy_sum: Tensor
    token_count: int


@dataclass(frozen=True, slots=True)
class MoEForwardAux:
    z_loss: Tensor
    layers: tuple[LayerMoETelemetry, ...]


class DroplessTop1MoE(nn.Module):
    """Eight full SwiGLU experts; every token is executed by exactly one expert."""

    def __init__(self, config: MoEModelConfig) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.d_model = config.d_model
        self.experts = nn.ModuleList(
            SwiGLU(config.d_model, config.expert_d_ff)
            for _ in range(config.num_experts)
        )
        self.router = SwitchTop1Router(
            config.d_model,
            config.num_experts,
            init_std=config.router_init_std,
            balancing_step_size=config.balancing_step_size,
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, LayerMoETelemetry]:
        shape = x.shape
        if x.ndim != 3 or shape[-1] != self.d_model:
            raise ValueError("MoE FFN expects [batch, sequence, d_model]")
        flat = x.reshape(-1, self.d_model)
        route = self.router(flat)
        combined = torch.zeros_like(flat)
        selected_probability_sum = route.selected_probabilities.sum().detach()
        for expert_id, expert in enumerate(self.experts):
            token_indices = torch.nonzero(
                route.expert_indices == expert_id, as_tuple=False
            ).flatten()
            if token_indices.numel() == 0:
                continue
            expert_inputs = flat.index_select(0, token_indices)
            expert_outputs = expert(expert_inputs)
            gates = route.selected_probabilities.index_select(
                0, token_indices
            ).to(dtype=expert_outputs.dtype).unsqueeze(-1)
            combined = combined.index_copy(
                0, token_indices, (expert_outputs * gates).to(dtype=combined.dtype)
            )
        telemetry = LayerMoETelemetry(
            expert_counts=route.expert_counts.detach(),
            selected_probability_sum=selected_probability_sum,
            entropy_sum=route.entropy_sum.detach(),
            token_count=route.token_count,
        )
        return combined.reshape(shape), route.z_loss, telemetry


class MoEDecoderBlock(nn.Module):
    """Dense mixer branch plus an 8-expert Top-1 FFN branch."""

    def __init__(self, config: MoEModelConfig, mixer: nn.Module) -> None:
        super().__init__()
        dense = config.dense
        self.mixer = mixer
        self.mixer_norm = RMSNorm(dense.d_model, dense.rms_norm_eps)
        self.ffn_norm = RMSNorm(dense.d_model, dense.rms_norm_eps)
        self.ffn = DroplessTop1MoE(config)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, LayerMoETelemetry]:
        mixed = self.mixer(self.mixer_norm(x))
        if isinstance(mixed, tuple):
            mixed = mixed[0]
        x = x + mixed
        expert_out, z_loss, telemetry = self.ffn(self.ffn_norm(x))
        return x + expert_out, z_loss, telemetry


class MoESmallLLM(nn.Module):
    """GDN-2/MHA Small-LLM with every dense FFN replaced by 8 Top-1 experts."""

    def __init__(self, config: MoEModelConfig) -> None:
        super().__init__()
        self.config = config
        dense = config.dense
        kinds = tuple(str(kind).lower() for kind in dense.layer_kinds)
        self.layer_kinds = kinds
        self.token_embedding = TiedEmbedding(dense)
        blocks: list[MoEDecoderBlock] = []
        for kind in kinds:
            if kind in {"gdn", "gdn-2"}:
                mixer = GatedDeltaNet2(dense)
            elif kind == "swa":
                mixer = GatedMultiheadAttention(replace(dense, attention_window=512))
            elif kind == "mha":
                mixer = GatedMultiheadAttention(replace(dense, attention_window=None))
            else:
                raise NotImplementedError(
                    f"MoE experiment does not support mixer kind {kind!r}"
                )
            blocks.append(MoEDecoderBlock(config, mixer))
        self.blocks = nn.ModuleList(blocks)
        self.final_norm = RMSNorm(dense.d_model, dense.rms_norm_eps)

    @property
    def embedding(self) -> nn.Module:
        return self.token_embedding

    @property
    def layers(self) -> nn.ModuleList:
        return self.blocks

    @property
    def lm_head(self) -> nn.Module:
        return self.token_embedding

    def reset_router_parameters(self) -> None:
        for block in self.blocks:
            block.ffn.router.reset_router_parameters()

    def _logits(self, hidden: Tensor) -> Tensor:
        semantic = self.config.semantic_vocab_size
        for name in ("logits", "lm_head", "project"):
            projection = getattr(self.token_embedding, name, None)
            if callable(projection):
                logits = projection(hidden)
                if logits.shape[-1] > semantic:
                    logits = logits.narrow(-1, 0, semantic)
                if logits.shape[-1] != semantic:
                    raise ValueError("tied embedding projection returned wrong vocabulary width")
                return logits
        weight = getattr(self.token_embedding, "weight", None)
        if weight is None:
            raise AttributeError("tied embedding must expose logits(hidden) or weight")
        return torch.nn.functional.linear(hidden, weight[:semantic])

    def forward_with_aux(self, input_ids: Tensor) -> tuple[Tensor, MoEForwardAux]:
        hidden = self.token_embedding(input_ids)
        z_losses: list[Tensor] = []
        telemetry: list[LayerMoETelemetry] = []
        for block in self.blocks:
            hidden, layer_z, layer_telemetry = block(hidden)
            z_losses.append(layer_z)
            telemetry.append(layer_telemetry)
        if len(z_losses) != self.config.n_layers:
            raise RuntimeError("MoE forward did not produce one router z-loss per layer")
        z_loss = torch.stack(z_losses).mean()
        logits = self._logits(self.final_norm(hidden))
        return logits, MoEForwardAux(z_loss=z_loss, layers=tuple(telemetry))

    def forward(self, input_ids: Tensor, cache: Any | None = None) -> Tensor:
        if cache is not None:
            raise NotImplementedError("MoE cached decoding is not implemented")
        logits, _ = self.forward_with_aux(input_ids)
        return logits


__all__ = [
    "DroplessTop1MoE",
    "GDN2Cache",
    "LayerMoETelemetry",
    "MoEDecoderBlock",
    "MoEForwardAux",
    "MoESmallLLM",
]
