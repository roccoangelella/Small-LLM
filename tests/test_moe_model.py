"""CPU-unit contracts for the first Top-1 MoE architecture."""

from __future__ import annotations

import unittest

import torch

from model.config import ModelConfig
from MOE_model.accounting import count_moe_parameters
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


class TestMoEModel(unittest.TestCase):
    def test_router_is_random_bias_free_top1_and_differentiable(self) -> None:
        torch.manual_seed(17)
        router = SwitchTop1Router(16, 8, init_std=0.01)
        self.assertIsNone(router.projection.bias)
        self.assertGreater(float(router.projection.weight.std()), 0.0)

        x = torch.randn(64, 16, requires_grad=True)
        first = router(x)
        second = router(x)
        self.assertTrue(torch.equal(first.expert_indices, second.expert_indices))
        self.assertTrue(
            torch.equal(first.selected_probabilities, second.selected_probabilities)
        )
        self.assertEqual(first.expert_indices.shape, (64,))
        self.assertEqual(first.selected_probabilities.shape, (64,))
        self.assertEqual(int(first.expert_counts.sum()), 64)

        loss = first.selected_probabilities.mean() + 1e-4 * first.z_loss
        loss.backward()
        self.assertIsNotNone(router.projection.weight.grad)
        self.assertTrue(bool(torch.isfinite(router.projection.weight.grad).all()))

    def test_dropless_moe_executes_exactly_one_expert_per_token(self) -> None:
        torch.manual_seed(3)
        config = _tiny_moe_config()
        moe = DroplessTop1MoE(config)
        x = torch.randn(2, 7, config.d_model, requires_grad=True)
        y, z_loss, telemetry = moe(x)
        self.assertEqual(y.shape, x.shape)
        self.assertEqual(int(telemetry.expert_counts.sum()), 14)
        self.assertEqual(telemetry.token_count, 14)
        self.assertTrue(bool(torch.isfinite(z_loss)))

    def test_only_selected_expert_gets_token_path_gradient(self) -> None:
        torch.manual_seed(5)
        config = _tiny_moe_config()
        moe = DroplessTop1MoE(config)
        with torch.no_grad():
            moe.router.projection.weight.zero_()
            moe.router.projection.weight[0].fill_(0.1)
        x = torch.ones(2, 5, config.d_model, requires_grad=True)
        y, z_loss, telemetry = moe(x)
        self.assertEqual(int(telemetry.expert_counts[0]), 10)
        self.assertEqual(int(telemetry.expert_counts[1:].sum()), 0)

        (y.square().mean() + 1e-4 * z_loss).backward()
        self.assertIsNotNone(moe.router.projection.weight.grad)
        self.assertIsNotNone(moe.experts[0].gate.weight.grad)
        for expert in moe.experts[1:]:
            self.assertIsNone(expert.gate.weight.grad)
            self.assertIsNone(expert.up.weight.grad)
            self.assertIsNone(expert.down.weight.grad)

    def test_config_fails_closed_on_top2_balancing_or_drops(self) -> None:
        config = _tiny_moe_config()
        dense = config.dense
        common = {
            "dense": dense,
            "expert_d_ff": dense.d_ff,
            "moe_layer_indices": tuple(range(dense.n_layers)),
        }
        with self.assertRaises(ValueError):
            MoEModelConfig(**common, top_k=2)
        with self.assertRaises(ValueError):
            MoEModelConfig(**common, load_balancing="none", capacity_factor=1.5)
        with self.assertRaises(ValueError):
            MoEModelConfig(**common, dropped_tokens_allowed=True)

    def test_moe_optimizer_routes_experts_to_muon_and_router_to_adamw(self) -> None:
        config = _tiny_moe_config()

        class Wrapper(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.ffn = DroplessTop1MoE(config)
                self.token_embedding = torch.nn.Embedding(32, 64)

        wrapper = Wrapper()
        routing = classify_moe_parameters(wrapper).routing
        self.assertTrue(
            any(".experts.0.gate.weight" in name for name in routing.muon)
        )
        self.assertIn("ffn.router.projection.weight", routing.adamw_decay)
        self.assertEqual(
            set(routing.all_names),
            {name for name, parameter in wrapper.named_parameters() if parameter.requires_grad},
        )

    def test_parameter_accounting_separates_stored_and_active_experts(self) -> None:
        config = _tiny_moe_config()

        class Wrapper(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.ffn = DroplessTop1MoE(config)
                self.token_embedding = torch.nn.Embedding(32, 64)

        wrapper = Wrapper()
        counts = count_moe_parameters(wrapper, num_experts=config.num_experts)
        self.assertGreater(counts.experts_stored, counts.experts_active_per_token)
        self.assertEqual(
            counts.experts_stored,
            counts.experts_active_per_token * config.num_experts,
        )
        self.assertEqual(
            counts.active_per_token,
            counts.backbone_non_expert + counts.router + counts.experts_active_per_token,
        )


if __name__ == "__main__":
    unittest.main()
