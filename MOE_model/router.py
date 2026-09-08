"""FP32 Switch-style Top-1 routing for the first MoE experiment."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class Top1Routing:
    logits: Tensor
    expert_indices: Tensor
    selected_probabilities: Tensor
    z_loss: Tensor
    entropy_sum: Tensor
    token_count: int
    expert_counts: Tensor


class SwitchTop1Router(nn.Module):
    """Bias-free Top-1 router with differentiable selected softmax probability."""

    def __init__(
        self, d_model: int, num_experts: int, *, init_std: float,
        balancing_step_size: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model <= 0 or num_experts <= 1 or init_std <= 0:
            raise ValueError("invalid Top-1 router geometry")
        self.d_model = int(d_model)
        self.num_experts = int(num_experts)
        self.init_std = float(init_std)
        if not math.isfinite(balancing_step_size) or balancing_step_size < 0:
            raise ValueError("balancing_step_size must be finite and non-negative")
        self.balancing_step_size = float(balancing_step_size)
        self.projection = nn.Linear(d_model, num_experts, bias=False)
        self.register_buffer("selection_bias", torch.zeros(num_experts, dtype=torch.float32))
        self.register_buffer("token_exposure", torch.zeros(num_experts, dtype=torch.long))
        self.register_buffer("expert_updates", torch.zeros(num_experts, dtype=torch.long))
        self.reset_router_parameters()

    @torch.no_grad()
    def commit_load(self, counts: Tensor) -> None:
        """Commit one successful full batch; never called by forward/evaluation."""
        if counts.shape != self.token_exposure.shape or counts.dtype != torch.long:
            raise ValueError("load must contain one int64 count per expert")
        self.token_exposure.add_(counts)
        self.expert_updates.add_(counts > 0)
        if self.balancing_step_size:
            # Eq. 3 / Algorithm 1, arXiv:2408.15664v1, applied to softmax.
            # The Appendix C proportional-error controller is a different variant.
            error = counts.float().mean() - counts.float()
            self.selection_bias.add_(error.sign(), alpha=self.balancing_step_size)

    def reset_router_parameters(self) -> None:
        with torch.no_grad():
            nn.init.normal_(self.projection.weight, mean=0.0, std=self.init_std)
            self.selection_bias.zero_()
            self.token_exposure.zero_()
            self.expert_updates.zero_()

    def forward(self, x: Tensor) -> Top1Routing:
        if x.ndim != 2 or x.shape[-1] != self.d_model:
            raise ValueError("router expects [tokens, d_model]")
        device_type = x.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            logits = F.linear(x.float(), self.projection.weight.float())
            probabilities = torch.softmax(logits, dim=-1)
            expert_indices = (probabilities + self.selection_bias).argmax(dim=-1)
            selected_probabilities = probabilities.gather(
                -1, expert_indices.unsqueeze(-1)
            ).squeeze(-1)
            log_z = torch.logsumexp(logits, dim=-1)
            z_loss = log_z.square().mean()
            entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
            entropy_sum = entropy.sum()
            expert_counts = torch.bincount(expert_indices, minlength=self.num_experts)
        return Top1Routing(
            logits=logits.detach(),
            expert_indices=expert_indices,
            selected_probabilities=selected_probabilities,
            z_loss=z_loss,
            entropy_sum=entropy_sum,
            token_count=int(x.shape[0]),
            expert_counts=expert_counts,
        )
