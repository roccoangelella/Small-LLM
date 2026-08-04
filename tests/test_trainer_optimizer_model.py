"""Routing qualification against the repository's assembled model variants."""

from __future__ import annotations

import unittest

from model.config import ModelConfig
from model.model import SmallLLM
from trainer.config import TrainerConfig
from trainer.optimizer import build_hybrid_muon_adamw, optimizer_parameter_routing


def _config(architecture: str) -> ModelConfig:
    return ModelConfig(
        semantic_vocab_size=257,
        padded_vocab_size=264,
        max_seq_len=512 if architecture == "swa_hybrid" else 8,
        d_model=64,
        n_layers=4,
        d_ff=128,
        n_heads=1,
        head_dim=64,
        gdn_num_key_heads=1,
        gdn_num_value_heads=1,
        gdn_key_dim=64,
        gdn_value_dim=64,
        gdn_conv_kernel_size=4,
        gdn_chunk_size=4,
        architecture=architecture,
    )


class AssembledModelOptimizerRoutingTests(unittest.TestCase):
    def test_all_runnable_architectures_route_every_parameter_once(self) -> None:
        trainer_config = TrainerConfig(
            optimizer="hybrid_muon_adamw",
            precision="fp32",
        )
        for architecture in ("gdn2_hybrid", "swa_hybrid", "all_mha"):
            with self.subTest(architecture=architecture):
                model = SmallLLM(_config(architecture))
                routing = optimizer_parameter_routing(model)
                expected = {
                    name
                    for name, parameter in model.named_parameters()
                    if parameter.requires_grad
                }
                self.assertEqual(set(routing.all_names), expected)
                self.assertEqual(len(routing.all_names), len(set(routing.all_names)))
                optimizer = build_hybrid_muon_adamw(model, trainer_config)
                self.assertEqual(
                    {group["optimizer_role"] for group in optimizer.param_groups},
                    {"muon", "adamw_decay", "adamw_no_decay"},
                )


if __name__ == "__main__":
    unittest.main()
