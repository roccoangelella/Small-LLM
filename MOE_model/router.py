"""FP32 Switch-style routing: Top-1 for the frozen M0 identity, Top-k for version 3.

Top-1 behaviour is preserved exactly. With ``top_k == 1`` the selection is `topk(1)`,
which returns the same element as `argmax` including its lowest-index tie-break, and the
combine weight is the selected softmax probability, unnormalised — byte-for-byte the M0
contract. With ``top_k > 1`` the k selected probabilities are renormalised so the mixture
weights sum to one, which is the only point on which the two competing router ADRs in the
`main` history agree; the score function itself (softmax here, versus sigmoid or
sqrt-softplus there) remains an open owner decision and does not affect shapes, dispatch
or any systems measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class TopKRouting:
    logits: Tensor
    expert_indices: Tensor          # [tokens, k]
    selected_probabilities: Tensor  # [tokens, k], mixture weights
    z_loss: Tensor
    entropy_sum: Tensor
    token_count: int
    expert_counts: Tensor           # [num_experts], assignments not tokens


Top1Routing = TopKRouting


class SwitchTopKRouter(nn.Module):
    """Bias-free router with differentiable selected softmax probabilities."""

    def __init__(
        self, d_model: int, num_experts: int, *, init_std: float,
        balancing_step_size: float = 0.0, top_k: int = 1, scoring: str = "softmax",
    ) -> None:
        super().__init__()
        if d_model <= 0 or num_experts <= 1 or init_std <= 0:
            raise ValueError("invalid router geometry")
        if scoring not in {"softmax", "sqrt_softplus"}:
            raise ValueError("unsupported router scoring")
        if scoring == "sqrt_softplus" and int(top_k) < 2:
            raise ValueError("sqrt_softplus affinities require top_k >= 2")
        self.scoring = scoring
        if not 1 <= int(top_k) <= int(num_experts):
            raise ValueError("top_k must be between 1 and num_experts")
        self.d_model = int(d_model)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
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

    def forward(self, x: Tensor) -> TopKRouting:
        if x.ndim != 2 or x.shape[-1] != self.d_model:
            raise ValueError("router expects [tokens, d_model]")
        device_type = x.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            logits = F.linear(x.float(), self.projection.weight.float())
            if self.scoring == "softmax":
                # Version 2: probabilities normalised across experts; the classical
                # softmax z-loss keeps the logits from drifting.
                scores = torch.softmax(logits, dim=-1)
                z_loss = torch.logsumexp(logits, dim=-1).square().mean()
                distribution = scores
            else:
                # Accepted contract: elementwise affinity, no cross-expert normalisation
                # before selection. There is no softmax partition function here, so the
                # classical z-loss does not apply and is returned as an exact zero.
                scores = torch.sqrt(F.softplus(logits))
                z_loss = logits.new_zeros(())
                distribution = scores / scores.sum(dim=-1, keepdim=True).clamp_min(
                    torch.finfo(scores.dtype).tiny
                )
            # The balancing bias steers selection only; weights come from unbiased scores.
            expert_indices = (scores + self.selection_bias).topk(self.top_k, dim=-1).indices
            selected = scores.gather(-1, expert_indices)
            if self.top_k > 1:
                selected = selected / selected.sum(dim=-1, keepdim=True).clamp_min(
                    torch.finfo(selected.dtype).tiny
                )
            entropy_sum = -(
                distribution * distribution.clamp_min(1e-12).log()
            ).sum(dim=-1).sum()
            expert_counts = torch.bincount(
                expert_indices.reshape(-1), minlength=self.num_experts
            )
        return TopKRouting(
            logits=logits.detach(),
            expert_indices=expert_indices,
            selected_probabilities=selected,
            z_loss=z_loss,
            entropy_sum=entropy_sum,
            token_count=int(x.shape[0]),
            expert_counts=expert_counts,
        )


class SwitchTop1Router(SwitchTopKRouter):
    """The frozen M0 router: the Top-k router pinned to k = 1, with the M0 return shapes.

    Selection, weights and counts are those of the general router at k = 1; only the
    trailing singleton axis is dropped, because the M0 identity returns one index and one
    probability per token and its consumers and tests depend on that shape.
    """

    def __init__(
        self, d_model: int, num_experts: int, *, init_std: float,
        balancing_step_size: float = 0.0,
    ) -> None:
        super().__init__(
            d_model, num_experts, init_std=init_std,
            balancing_step_size=balancing_step_size, top_k=1,
        )

    def forward(self, x: Tensor) -> TopKRouting:
        route = super().forward(x)
        return TopKRouting(
            logits=route.logits,
            expert_indices=route.expert_indices.squeeze(-1),
            selected_probabilities=route.selected_probabilities.squeeze(-1),
            z_loss=route.z_loss,
            entropy_sum=route.entropy_sum,
            token_count=route.token_count,
            expert_counts=route.expert_counts,
        )


__all__ = ["SwitchTop1Router", "SwitchTopKRouter", "Top1Routing", "TopKRouting"]
