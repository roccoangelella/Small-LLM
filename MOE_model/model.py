"""Dropless MoE variants of the Small-LLM architecture.

``DroplessTop1MoE`` is the frozen M0 identity: eight SwiGLU expert modules, Top-1, sorted
dispatch with one host synchronisation per layer. ``DroplessTopKMoE`` is the version-3
geometry: expert weights stacked into one parameter per projection and executed as a
single padded batched GEMM per layer, so the number of kernel launches stops growing with
the expert count.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from model.components import GatedMultiheadAttention, RMSNorm, SwiGLU, TiedEmbedding
from model.gdn2 import GDN2Cache
from model.gdn2_stable import StableGatedDeltaNet2 as GatedDeltaNet2

from .config import MoEModelConfig
from .router import SwitchTop1Router, SwitchTopKRouter


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
        expert_indices = route.expert_indices
        selected_probabilities = route.selected_probabilities
        selected_probability_sum = selected_probabilities.sum().detach()
        # Sorted dispatch: one stable argsort groups tokens by expert (ascending
        # token order inside each group, exactly as the previous per-expert
        # ``nonzero`` scan produced), one host sync reads the E group
        # boundaries, every expert runs on a contiguous slice, and the outputs
        # are written back once with an in-place ``index_copy_`` over a
        # permutation (unique rows, no summation, no full-buffer copies).
        order = torch.argsort(expert_indices, stable=True)
        sorted_inputs = flat.index_select(0, order)
        sorted_gates = selected_probabilities.index_select(0, order)
        boundaries = route.expert_counts.cumsum(0).tolist()
        pieces: list[Tensor] = []
        start = 0
        for expert_id, expert in enumerate(self.experts):
            stop = int(boundaries[expert_id])
            if stop > start:
                expert_outputs = expert(sorted_inputs[start:stop])
                gates = sorted_gates[start:stop].to(dtype=expert_outputs.dtype).unsqueeze(-1)
                pieces.append((expert_outputs * gates).to(dtype=flat.dtype))
            start = stop
        combined = torch.zeros_like(flat)
        if pieces:
            combined = combined.index_copy_(0, order, torch.cat(pieces, dim=0))
        telemetry = LayerMoETelemetry(
            expert_counts=route.expert_counts.detach(),
            selected_probability_sum=selected_probability_sum,
            entropy_sum=route.entropy_sum.detach(),
            token_count=route.token_count,
        )
        return combined.reshape(shape), route.z_loss, telemetry


class DroplessTopKMoE(nn.Module):
    """Stacked SwiGLU experts executed as one padded batched GEMM per layer.

    Tokens are sorted by expert exactly as in the Top-1 path, then written into a dense
    ``[experts, capacity, d_model]`` buffer whose capacity is the largest expert load in
    this batch. Three ``bmm`` calls evaluate every expert at once, so a layer costs a fixed
    number of kernel launches instead of one group per expert. Padding rows compute values
    that are never read; their cost is proportional to routing imbalance and is reported as
    ``padded_fraction``.

    Recombination is deterministic: the permutation is inverted with ``index_copy_`` (unique
    destinations, no atomics) and the k contributions of a token are summed along a
    dedicated axis, so repeated runs of the same input give bit-identical output.
    """

    def __init__(self, config: MoEModelConfig) -> None:
        super().__init__()
        self.num_experts = int(config.num_experts)
        self.top_k = int(config.top_k)
        self.d_model = int(config.d_model)
        self.d_ff = int(config.expert_d_ff)
        # Two implementations of one contract: the batched GEMM is the production path,
        # the per-expert loop is the control that isolates what batching actually buys.
        self.batched = config.dispatch == "dropless_padded_batched_gemm"
        self.gate_weight = nn.Parameter(torch.empty(self.num_experts, self.d_model, self.d_ff))
        self.up_weight = nn.Parameter(torch.empty(self.num_experts, self.d_model, self.d_ff))
        self.down_weight = nn.Parameter(torch.empty(self.num_experts, self.d_ff, self.d_model))
        for weight in (self.gate_weight, self.up_weight, self.down_weight):
            nn.init.normal_(weight, mean=0.0, std=0.02)
        self.router = SwitchTopKRouter(
            config.d_model,
            config.num_experts,
            init_std=config.router_init_std,
            balancing_step_size=config.balancing_step_size,
            top_k=config.top_k,
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, LayerMoETelemetry]:
        shape = x.shape
        if x.ndim != 3 or shape[-1] != self.d_model:
            raise ValueError("MoE FFN expects [batch, sequence, d_model]")
        flat = x.reshape(-1, self.d_model)
        tokens = flat.shape[0]
        route = self.router(flat)
        assignments = route.expert_indices.reshape(-1)          # token-major, [tokens * k]
        counts = route.expert_counts
        order = torch.argsort(assignments, stable=True)
        expert_of = assignments.index_select(0, order)
        token_of = torch.div(order, self.top_k, rounding_mode="floor")
        starts = counts.cumsum(0) - counts
        rank = torch.arange(assignments.numel(), device=flat.device) - starts.index_select(0, expert_of)
        sorted_inputs = flat.index_select(0, token_of)

        if self.batched:
            capacity = int(counts.max())                        # the one host synchronisation
            slots = expert_of * capacity + rank
            buffer = flat.new_zeros(self.num_experts * capacity, self.d_model)
            buffer.index_copy_(0, slots, sorted_inputs)
            grouped = buffer.view(self.num_experts, capacity, self.d_model)
            hidden = F.silu(torch.bmm(grouped, self.gate_weight)) * torch.bmm(grouped, self.up_weight)
            produced = torch.bmm(hidden, self.down_weight)
            contributions = produced.reshape(self.num_experts * capacity, self.d_model).index_select(0, slots)
        else:
            boundaries = counts.cumsum(0).tolist()               # the same single host read
            pieces: list[Tensor] = []
            start = 0
            for expert in range(self.num_experts):
                stop = int(boundaries[expert])
                if stop > start:
                    chunk = sorted_inputs[start:stop]
                    hidden = F.silu(chunk @ self.gate_weight[expert]) * (chunk @ self.up_weight[expert])
                    pieces.append(hidden @ self.down_weight[expert])
                start = stop
            contributions = torch.cat(pieces, dim=0)
        restored = contributions.new_empty(assignments.numel(), self.d_model)
        restored.index_copy_(0, order, contributions)
        weights = route.selected_probabilities.to(dtype=restored.dtype).unsqueeze(-1)
        combined = (restored.view(tokens, self.top_k, self.d_model) * weights).sum(dim=1)

        telemetry = LayerMoETelemetry(
            expert_counts=counts.detach(),
            selected_probability_sum=route.selected_probabilities.sum().detach(),
            entropy_sum=route.entropy_sum.detach(),
            token_count=route.token_count,
        )
        return combined.to(dtype=flat.dtype).reshape(shape), route.z_loss, telemetry


def _build_ffn(config: MoEModelConfig) -> nn.Module:
    """Pick the dispatch the configuration declares; never infer it from the geometry."""

    if config.dispatch == "dropless_padded_batched_gemm":
        return DroplessTopKMoE(config)
    return DroplessTop1MoE(config)


class MoEDecoderBlock(nn.Module):
    """Dense mixer branch plus a routed expert FFN branch."""

    def __init__(self, config: MoEModelConfig, mixer: nn.Module) -> None:
        super().__init__()
        dense = config.dense
        self.mixer = mixer
        self.mixer_norm = RMSNorm(dense.d_model, dense.rms_norm_eps)
        self.ffn_norm = RMSNorm(dense.d_model, dense.rms_norm_eps)
        self.ffn = _build_ffn(config)

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

    def scale_expert_output_projections(self, factor: float) -> None:
        """Apply the residual output scale the dense initializer gives ``nn.Linear`` down
        projections to the stacked expert weights, which are bare parameters."""

        with torch.no_grad():
            for block in self.blocks:
                weight = getattr(block.ffn, "down_weight", None)
                if weight is not None:
                    weight.mul_(factor)

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

    def hidden_states_with_aux(self, input_ids: Tensor) -> tuple[Tensor, MoEForwardAux]:
        """Final-normed hidden states plus router aux, before the tied output
        projection; the MoE training step scores them in chunks."""

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
        return self.final_norm(hidden), MoEForwardAux(z_loss=z_loss, layers=tuple(telemetry))

    def forward_with_aux(self, input_ids: Tensor) -> tuple[Tensor, MoEForwardAux]:
        hidden, aux = self.hidden_states_with_aux(input_ids)
        return self._logits(hidden), aux

    def forward(self, input_ids: Tensor, cache: Any | None = None) -> Tensor:
        if cache is not None:
            raise NotImplementedError("MoE cached decoding is not implemented")
        logits, _ = self.forward_with_aux(input_ids)
        return logits


__all__ = [
    "DroplessTop1MoE",
    "DroplessTopKMoE",
    "GDN2Cache",
    "LayerMoETelemetry",
    "MoEDecoderBlock",
    "MoEForwardAux",
    "MoESmallLLM",
]
