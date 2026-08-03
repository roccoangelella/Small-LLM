"""CPU tests for explicit whole-matrix Muon + AdamW routing and state."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from trainer.config import TrainerConfig
from trainer.optimizer import (
    HybridMuonAdamW,
    build_hybrid_muon_adamw,
    optimizer_parameter_routing,
)
from trainer.schedule import TokenLRScheduler


class _Mixer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(4, 4, bias=False)
        self.q_conv = nn.Conv1d(4, 4, 3, groups=4, bias=False)
        self.output_gate = nn.Sequential(
            nn.Linear(4, 4, bias=False),
            nn.Linear(4, 4, bias=True),
        )
        self.output_norm = nn.LayerNorm(4, elementwise_affine=True, bias=False)
        self.A_log = nn.Parameter(torch.zeros(1))
        self.dt_bias = nn.Parameter(torch.zeros(1, 4))
        self.out_proj = nn.Linear(4, 4, bias=False)


class _FFN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate = nn.Linear(4, 8, bias=False)
        self.up = nn.Linear(4, 8, bias=False)
        self.down = nn.Linear(8, 4, bias=False)


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mixer = _Mixer()
        self.mixer_norm = nn.LayerNorm(4, elementwise_affine=True, bias=False)
        self.ffn_norm = nn.LayerNorm(4, elementwise_affine=True, bias=False)
        self.ffn = _FFN()


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(16, 4)
        self.blocks = nn.ModuleList([_Block()])
        self.final_norm = nn.LayerNorm(4, elementwise_affine=True, bias=False)


class HybridOptimizerTests(unittest.TestCase):
    def test_routing_is_explicit_and_exhaustive(self) -> None:
        model = _Model()
        routing = optimizer_parameter_routing(model)
        expected = {
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        }
        self.assertEqual(set(routing.all_names), expected)
        self.assertEqual(len(routing.all_names), len(set(routing.all_names)))
        self.assertIn("blocks.0.mixer.q_proj.weight", routing.muon)
        self.assertIn("blocks.0.ffn.down.weight", routing.muon)
        self.assertIn("token_embedding.weight", routing.adamw_decay)
        self.assertIn("blocks.0.mixer.q_conv.weight", routing.adamw_decay)
        self.assertIn("blocks.0.mixer.A_log", routing.adamw_no_decay)
        self.assertIn("final_norm.weight", routing.adamw_no_decay)

    def test_step_uses_fp32_state_for_both_branches(self) -> None:
        torch.manual_seed(3)
        model = _Model()
        config = TrainerConfig(
            optimizer="hybrid_muon_adamw",
            precision="fp32",
            learning_rate=1e-3,
            muon_lr_multiplier=0.5,
        )
        optimizer = build_hybrid_muon_adamw(model, config)
        self.assertIsInstance(optimizer, HybridMuonAdamW)
        before = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }
        for parameter in model.parameters():
            parameter.grad = torch.randn_like(parameter)
        optimizer.step()
        self.assertTrue(
            all(
                not torch.equal(before[name], parameter)
                for name, parameter in model.named_parameters()
            )
        )
        roles = {
            id(parameter): group["optimizer_role"]
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        for parameter, state in optimizer.state.items():
            if roles[id(parameter)] == "muon":
                self.assertEqual(state["momentum_buffer"].dtype, torch.float32)
            else:
                self.assertEqual(state["exp_avg"].dtype, torch.float32)
                self.assertEqual(state["exp_avg_sq"].dtype, torch.float32)

    def test_scheduler_preserves_muon_multiplier(self) -> None:
        model = _Model()
        config = TrainerConfig(
            optimizer="hybrid_muon_adamw",
            precision="fp32",
            learning_rate=2e-3,
            muon_lr_multiplier=0.25,
        )
        optimizer = build_hybrid_muon_adamw(model, config)
        TokenLRScheduler(optimizer, config)
        rates = {
            group["optimizer_role"]: group["lr"] for group in optimizer.param_groups
        }
        self.assertEqual(rates["muon"], 5e-4)
        self.assertEqual(rates["adamw_decay"], 2e-3)
        self.assertEqual(rates["adamw_no_decay"], 2e-3)

    def test_fp16_checkpoint_restore_keeps_fp32_optimizer_state(self) -> None:
        model = _Model().half()
        config = TrainerConfig(
            optimizer="hybrid_muon_adamw",
            precision="fp16",
        )
        optimizer = build_hybrid_muon_adamw(model, config)
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        state = optimizer.state_dict()

        restored_model = _Model().half()
        restored = build_hybrid_muon_adamw(restored_model, config)
        restored.load_state_dict(state)
        tensor_states = [
            value
            for parameter_state in restored.state.values()
            for value in parameter_state.values()
            if isinstance(value, torch.Tensor)
        ]
        self.assertTrue(tensor_states)
        self.assertTrue(all(value.dtype == torch.float32 for value in tensor_states))

    def test_checkpoint_state_binds_recipe_and_routing(self) -> None:
        model = _Model()
        config = TrainerConfig(optimizer="hybrid_muon_adamw", precision="fp32")
        optimizer = build_hybrid_muon_adamw(model, config)
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        state = optimizer.state_dict()
        self.assertEqual(
            state["small_llm_optimizer"]["recipe"],
            "deepseek_v4_whole_matrix_hybrid_ns10",
        )
        restored = build_hybrid_muon_adamw(_Model(), config)
        restored.load_state_dict(state)
        self.assertEqual(len(restored.state), len(optimizer.state))


if __name__ == "__main__":
    unittest.main()
