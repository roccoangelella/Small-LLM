"""CPU contracts for the accepted MoE geometry: 64 experts, Top-2, batched expert GEMM."""

from __future__ import annotations

import unittest

import torch
from torch import nn
from torch.nn import functional as F

from MOE_model.accounting import count_moe_parameters
from MOE_model.config import MoEModelConfig
from MOE_model.initialization import initialize_moe_model
from MOE_model.model import DroplessTopKMoE, MoESmallLLM
from MOE_model.router import SwitchTop1Router, SwitchTopKRouter
from model.config import ModelConfig
from trainer.config import TrainerConfig
from MOE_model.optimizer import build_moe_optimizer, classify_moe_parameters


def _tiny(num_experts: int = 6, top_k: int = 2, d_model: int = 16, d_ff: int = 10) -> MoEModelConfig:
    dense = ModelConfig(
        semantic_vocab_size=32, padded_vocab_size=32, max_seq_len=8,
        d_model=d_model, n_layers=4, d_ff=d_ff, n_heads=2, head_dim=d_model // 2,
        gdn_num_key_heads=2, gdn_num_value_heads=2,
        gdn_key_dim=d_model // 2, gdn_value_dim=d_model // 2,
        gdn_conv_kernel_size=4, gdn_chunk_size=4,
    )
    return MoEModelConfig(
        dense=dense, num_experts=num_experts, top_k=top_k, expert_d_ff=d_ff,
        expert_type="swiglu",
        router_combine="normalised_topk_softmax" if top_k > 1 else "selected_softmax_probability",
        dispatch="dropless_padded_batched_gemm",
        moe_layer_indices=tuple(range(4)), version=3,
    )


def _reference_moe(module: DroplessTopKMoE, x: torch.Tensor) -> torch.Tensor:
    """Oracle: evaluate each token's k experts one at a time and blend the outputs."""

    flat = x.reshape(-1, module.d_model)
    route = module.router(flat)
    out = torch.zeros_like(flat)
    for token in range(flat.shape[0]):
        for slot in range(module.top_k):
            expert = int(route.expert_indices[token, slot])
            weight = route.selected_probabilities[token, slot]
            gate = flat[token] @ module.gate_weight[expert]
            up = flat[token] @ module.up_weight[expert]
            out[token] = out[token] + weight * (F.silu(gate) * up) @ module.down_weight[expert]
    return out.reshape(x.shape)


class TestAcceptedConfiguration(unittest.TestCase):
    def test_accepted_geometry_matches_the_architecture_document(self) -> None:
        config = MoEModelConfig.accepted()
        self.assertEqual(
            (config.num_experts, config.top_k, config.shared_expert), (64, 2, False)
        )
        self.assertEqual(
            (config.dense.n_layers, config.dense.d_model, config.expert_d_ff), (8, 256, 352)
        )
        self.assertEqual(config.semantic_vocab_size, 8_000)
        self.assertEqual(config.dense.layer_pattern.count("gdn"), 3)
        self.assertEqual(config.dispatch, "dropless_padded_batched_gemm")

    def test_version_two_stays_frozen_on_every_opened_axis(self) -> None:
        dense = MoEModelConfig.substantive().dense
        for override in (
            {"num_experts": 64}, {"top_k": 2}, {"expert_type": "swiglu"},
            {"dispatch": "dropless_padded_batched_gemm"},
            {"router_combine": "normalised_topk_softmax"},
        ):
            with self.assertRaises(ValueError):
                MoEModelConfig(dense=dense, **override)

    def test_version_three_rejects_incoherent_routing(self) -> None:
        dense = _tiny().dense
        with self.assertRaises(ValueError):  # top_k above the expert count
            MoEModelConfig(dense=dense, num_experts=4, top_k=5, expert_d_ff=10,
                           expert_type="swiglu", router_combine="normalised_topk_softmax",
                           moe_layer_indices=tuple(range(4)), version=3)
        with self.assertRaises(ValueError):  # Top-2 with the Top-1 combine
            MoEModelConfig(dense=dense, num_experts=4, top_k=2, expert_d_ff=10,
                           expert_type="swiglu", router_combine="selected_softmax_probability",
                           moe_layer_indices=tuple(range(4)), version=3)


class TestAcceptedParameterCounts(unittest.TestCase):
    def test_counts_reproduce_the_independently_computed_totals(self) -> None:
        config = MoEModelConfig.accepted()
        model = initialize_moe_model(MoESmallLLM(config), "normal")
        counts = count_moe_parameters(model, num_experts=64, top_k=2)
        # Independent arithmetic from the 2026-09-09 architecture options document.
        self.assertEqual(counts.total, 144_025_496)
        self.assertEqual(counts.active_per_token, 9_938_840)
        self.assertEqual(counts.experts_stored, 3 * 8 * 256 * 64 * 352)
        self.assertEqual(counts.router, 8 * 256 * 64)


class TestTopKRouter(unittest.TestCase):
    def test_top1_router_keeps_the_frozen_shapes_and_values(self) -> None:
        torch.manual_seed(5)
        frozen = SwitchTop1Router(16, 8, init_std=0.01)
        general = SwitchTopKRouter(16, 8, init_std=0.01, top_k=1)
        general.load_state_dict(frozen.state_dict())
        x = torch.randn(32, 16)
        a, b = frozen(x), general(x)
        self.assertEqual(a.expert_indices.shape, (32,))
        self.assertEqual(a.selected_probabilities.shape, (32,))
        self.assertTrue(torch.equal(a.expert_indices, b.expert_indices.squeeze(-1)))
        self.assertTrue(
            torch.equal(a.selected_probabilities, b.selected_probabilities.squeeze(-1))
        )
        probabilities = torch.softmax(x.float() @ frozen.projection.weight.float().T, dim=-1)
        self.assertTrue(torch.equal(a.expert_indices, probabilities.argmax(dim=-1)))

    def test_topk_weights_are_normalised_and_counts_match_assignments(self) -> None:
        torch.manual_seed(6)
        router = SwitchTopKRouter(16, 6, init_std=0.01, top_k=3)
        route = router(torch.randn(40, 16))
        self.assertEqual(route.expert_indices.shape, (40, 3))
        torch.testing.assert_close(
            route.selected_probabilities.sum(dim=-1), torch.ones(40), rtol=1e-6, atol=1e-6
        )
        for row in route.expert_indices:
            self.assertEqual(len(set(int(v) for v in row)), 3)
        self.assertEqual(int(route.expert_counts.sum()), 40 * 3)

    def test_selection_bias_steers_selection_without_biasing_the_weights(self) -> None:
        torch.manual_seed(7)
        router = SwitchTopKRouter(16, 4, init_std=0.01, top_k=1, balancing_step_size=0.0)
        x = torch.randn(24, 16)
        before = router(x)
        with torch.no_grad():
            router.selection_bias[3] += 10.0
        after = router(x)
        self.assertTrue(bool((after.expert_indices == 3).all()))
        probabilities = torch.softmax(x.float() @ router.projection.weight.float().T, dim=-1)
        torch.testing.assert_close(
            after.selected_probabilities.squeeze(-1), probabilities[:, 3], rtol=1e-6, atol=1e-6
        )
        self.assertEqual(before.token_count, after.token_count)


class TestBatchedExpertGemm(unittest.TestCase):
    def test_matches_the_one_expert_at_a_time_oracle(self) -> None:
        for top_k in (1, 2, 3):
            with self.subTest(top_k=top_k):
                torch.manual_seed(9)
                module = DroplessTopKMoE(_tiny(num_experts=6, top_k=top_k))
                x = torch.randn(3, 7, 16)
                actual, z_loss, telemetry = module(x)
                expected = _reference_moe(module, x)
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                self.assertTrue(torch.isfinite(z_loss))
                self.assertEqual(int(telemetry.expert_counts.sum()), 21 * top_k)
                self.assertEqual(telemetry.token_count, 21)

    def test_batched_and_looped_dispatch_agree_on_identical_weights(self) -> None:
        torch.manual_seed(14)
        batched = DroplessTopKMoE(_tiny(num_experts=6, top_k=2))
        import dataclasses

        looped = DroplessTopKMoE(
            dataclasses.replace(
                _tiny(num_experts=6, top_k=2), dispatch="dropless_grouped_by_expert"
            )
        )
        looped.load_state_dict(batched.state_dict())
        self.assertTrue(batched.batched)
        self.assertFalse(looped.batched)
        x = torch.randn(2, 11, 16)
        a, _, ta = batched(x)
        b, _, tb = looped(x)
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
        self.assertTrue(torch.equal(ta.expert_counts, tb.expert_counts))

    def test_output_is_deterministic_across_repeated_calls(self) -> None:
        torch.manual_seed(10)
        module = DroplessTopKMoE(_tiny())
        x = torch.randn(2, 9, 16)
        first, _, _ = module(x)
        second, _, _ = module(x)
        self.assertTrue(torch.equal(first, second))

    def test_gradients_reach_every_expert_and_the_router(self) -> None:
        torch.manual_seed(11)
        module = DroplessTopKMoE(_tiny(num_experts=4, top_k=2))
        out, z_loss, _ = module(torch.randn(4, 8, 16))
        (out.square().mean() + z_loss).backward()
        for name in ("gate_weight", "up_weight", "down_weight"):
            grad = getattr(module, name).grad
            self.assertIsNotNone(grad)
            per_expert = grad.abs().flatten(1).sum(dim=1)
            self.assertTrue(bool((per_expert > 0).all()), f"{name} has an untouched expert")
        self.assertIsNotNone(module.router.projection.weight.grad)

    def test_extreme_imbalance_still_matches_the_oracle(self) -> None:
        torch.manual_seed(12)
        module = DroplessTopKMoE(_tiny(num_experts=5, top_k=1))
        with torch.no_grad():
            module.router.selection_bias[2] += 50.0
        x = torch.randn(2, 6, 16)
        actual, _, telemetry = module(x)
        torch.testing.assert_close(actual, _reference_moe(module, x), rtol=1e-5, atol=1e-6)
        self.assertEqual(int(telemetry.expert_counts[2]), 12)


class TestStackedExpertsUnderMuon(unittest.TestCase):
    def test_stacked_weights_are_routed_to_muon_and_orthogonalised_per_slice(self) -> None:
        config = _tiny(num_experts=4, top_k=2)
        model = initialize_moe_model(MoESmallLLM(config), "normal")
        routing = classify_moe_parameters(model).routing
        stacked = [n for n in routing.muon if n.endswith((".gate_weight", ".up_weight", ".down_weight"))]
        self.assertEqual(len(stacked), 3 * config.dense.n_layers)

        trainer_config = TrainerConfig(optimizer="hybrid_muon_adamw", precision="fp32",
                                       learning_rate=1e-2, muon_weight_decay=0.0, weight_decay=0.0)
        optimizer = build_moe_optimizer(model, trainer_config)
        generator = torch.Generator().manual_seed(13)
        for parameter in model.parameters():
            parameter.grad = torch.randn(parameter.shape, generator=generator)
        weight = model.blocks[0].ffn.gate_weight
        before = weight.detach().clone()
        gradient = weight.grad.detach().clone()
        optimizer.step()

        from trainer.optimizer import _newton_schulz_orthogonalize
        beta = float(trainer_config.muon_momentum)
        expected = torch.stack([
            _newton_schulz_orthogonalize(
                gradient[i].add(gradient[i], alpha=beta), target_rms=trainer_config.muon_update_rms
            )
            for i in range(gradient.shape[0])
        ])
        torch.testing.assert_close(
            weight, before - trainer_config.learning_rate * expected, rtol=1e-6, atol=1e-7
        )

    def test_full_model_step_keeps_every_parameter_finite(self) -> None:
        config = _tiny(num_experts=4, top_k=2)
        model = initialize_moe_model(MoESmallLLM(config), "normal")
        optimizer = build_moe_optimizer(
            model, TrainerConfig(optimizer="hybrid_muon_adamw", precision="fp32")
        )
        ids = torch.randint(0, config.semantic_vocab_size, (2, 8))
        logits, aux = model.forward_with_aux(ids)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids.reshape(-1)) + aux.z_loss
        loss.backward()
        optimizer.step()
        for name, parameter in model.named_parameters():
            self.assertTrue(bool(torch.isfinite(parameter).all()), name)


if __name__ == "__main__":
    unittest.main()
