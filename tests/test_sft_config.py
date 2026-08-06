from __future__ import annotations

import unittest

import torch

from post_training.sft.config import SFTDataConfig, SFTSchedulePlan, build_s0_trainer_config
from post_training.sft.interpolation import interpolate_state_dicts


class ConfigAndInterpolationTests(unittest.TestCase):
    def test_default_complete_source_shares(self) -> None:
        shares = SFTDataConfig().complete_source_shares
        self.assertAlmostEqual(sum(shares.values()), 1.0)
        self.assertAlmostEqual(shares["smol-magpie-ultra-short"], 0.6375)
        self.assertAlmostEqual(shares["climbmix-replay"], 0.15)

    def test_s0_trainer_preserves_optimizer_policy(self) -> None:
        plan = SFTSchedulePlan.from_block_target_counts(tuple([32768] * 20))
        config = build_s0_trainer_config(plan, precision="fp32")
        self.assertEqual(config.optimizer, "hybrid_muon_adamw")
        self.assertEqual(config.learning_rate, 3e-5)
        self.assertEqual(config.weight_decay, 0.0)
        self.assertEqual(config.muon_weight_decay, 0.0)
        self.assertEqual(config.schedule, "wsd")
        self.assertEqual(
            config.warmup_tokens + config.stable_tokens + config.decay_tokens,
            20 * 32768,
        )

    def test_interpolation_defaults_to_full_sft_and_is_tunable(self) -> None:
        base = {"weight": torch.tensor([0.0, 2.0])}
        tuned = {"weight": torch.tensor([2.0, 4.0])}
        full = interpolate_state_dicts(base, tuned)
        half = interpolate_state_dicts(base, tuned, alpha=0.5)
        torch.testing.assert_close(full["weight"], tuned["weight"])
        torch.testing.assert_close(half["weight"], torch.tensor([1.0, 3.0]))

    def test_dataset_budget_is_not_hard_coded_to_s0(self) -> None:
        config = SFTDataConfig(target_loss_tokens=100_000_000)
        self.assertEqual(config.target_loss_tokens, 100_000_000)


if __name__ == "__main__":
    unittest.main()
