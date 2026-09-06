"""CPU-unit contracts for the first Top-1 MoE architecture."""

from __future__ import annotations

import torch

from model.config import ModelConfig
from MOE_model.config import MoEModelConfig
from MOE_model.model import DroplessTop1MoE
from MOE_model.optimizer import classify_moe_parameters
from MOE_model.router import SwitchTop1Router


def _tiny_moe_config() -> MoEModelConfig:
    dense = ModelConfig(
        semantic_vocab_size=32,
        padded_vocab_size=32,
        max_seq_len=8,
        d_model=64,
        n_layers=4,
        d_ff=96,
        n_heads=4,
        head_dim=16,
        gdn_num_key_heads=4,
        gdn_num_value_heads=4,
        gdn_key_dim=16,
        gdn_value_dim=16,
        gdn_conv_kernel_size=4,
        gdn_chunk_size=4,
    )
    return MoEModelConfig(
        dense=dense,
        expert_d_ff=96,
        moe_layer_indices=tuple(range(4)),
    )


def test_router_is_random_bias_free_top1_and_differentiable() -> None:
    torch.manual_seed(17)
    router = SwitchTop1Router(16, 8, init_std=0.01)
    assert router.projection.bias is None
    assert float(router.projection.weight.std()) > 0

    x = torch.randn(64, 16, requires_grad=True)
    route = router(x)
    assert route.expert_indices.shape == (64,)
    assert route.selected_probabilities.shape == (64,)
    assert int(route.expert_counts.sum()) == 64

    loss = route.selected_probabilities.mean() + 1e-4 * route.z_loss
    loss.backward()
    assert router.projection.weight.grad is not None
    assert bool(torch.isfinite(router.projection.weight.grad).all())


def test_dropless_moe_executes_exactly_one_expert_per_token() -> None:
    torch.manual_seed(3)
    config = _tiny_moe_config()
    moe = DroplessTop1MoE(config)
    x = torch.randn(2, 7, config.d_model, requires_grad=True)
    y, z_loss, telemetry = moe(x)
    assert y.shape == x.shape
    assert int(telemetry.expert_counts.sum()) == 14
    assert telemetry.token_count == 14
    assert torch.isfinite(z_loss)


def test_moe_optimizer_routes_experts_to_muon_and_router_to_adamw() -> None:
    config = _tiny_moe_config()

    class Wrapper(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.ffn = DroplessTop1MoE(config)
            self.token_embedding = torch.nn.Embedding(32, 64)

    wrapper = Wrapper()
    routing = classify_moe_parameters(wrapper).routing
    assert any(".experts.0.gate.weight" in name for name in routing.muon)
    assert any(
        name.endswith(".ffn.router.projection.weight")
        for name in routing.adamw_decay
    )
