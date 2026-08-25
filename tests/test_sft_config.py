from __future__ import annotations

import math
import unittest

import torch

from post_training.sft.config import (
    S0_AGGRESSIVE_COOLDOWN_START_LR,
    S0_AGGRESSIVE_FINAL_LR,
    S0_AGGRESSIVE_PEAK_LR,
    S0_AGGRESSIVE_SETTLE_LR,
    SFTDataConfig,
    SFTSchedulePlan,
    build_s0_aggressive_trainer_config,
    build_s0_trainer_config,
)
from post_training.sft.interpolation import interpolate_state_dicts
from trainer import TokenLRScheduler


class ConfigAndInterpolationTests(unittest.TestCase):
    def test_default_complete_source_shares(self) -> None:
        shares = SFTDataConfig().complete_source_shares
        self.assertAlmostEqual(sum(shares.values()), 1.0)
        self.assertAlmostEqual(shares["smol-magpie-ultra-short"], 0.6375)
        self.assertAlmostEqual(shares["climbmix-replay"], 0.15)

    def test_historical_s0_trainer_preserves_optimizer_policy(self) -> None:
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

    def test_10pct_s0_aggressive_schedule_hits_frozen_lr_landmarks(self) -> None:
        total = 200_099_738
        full = total // 32_768
        remainder = total % 32_768
        counts = tuple([32_768] * full + ([remainder] if remainder else []))
        plan = SFTSchedulePlan.from_block_target_counts(counts)
        config = build_s0_aggressive_trainer_config(plan, precision="fp32")
        self.assertEqual(config.optimizer, "hybrid_muon_adamw")
        self.assertEqual(config.learning_rate, S0_AGGRESSIVE_PEAK_LR)
        self.assertEqual(config.schedule, "wsqd")
        self.assertEqual(config.stable_tokens, 0)
        self.assertEqual(config.schedule_anchor_tokens, config.warmup_tokens)
        self.assertEqual(config.cooldown_start_tokens + config.decay_tokens, total)

        parameter = torch.nn.Parameter(torch.ones(()))
        scheduler = TokenLRScheduler(torch.optim.SGD([parameter], lr=config.learning_rate), config)
        settle_end = config.schedule_anchor_tokens + config.settle_tokens
        self.assertTrue(
            math.isclose(
                scheduler.prepare_step(settle_end),
                S0_AGGRESSIVE_SETTLE_LR,
                rel_tol=1e-12,
            )
        )
        self.assertTrue(
            math.isclose(
                scheduler.prepare_step(config.cooldown_start_tokens),
                S0_AGGRESSIVE_COOLDOWN_START_LR,
                rel_tol=1e-12,
            )
        )
        self.assertTrue(
            math.isclose(
                scheduler.prepare_step(total),
                S0_AGGRESSIVE_FINAL_LR,
                rel_tol=1e-12,
            )
        )

    def test_10pct_s0_aggressive_peak_lr_is_fail_closed(self) -> None:
        plan = SFTSchedulePlan.from_block_target_counts(tuple([32_768] * 100))
        with self.assertRaisesRegex(ValueError, "peak LR is frozen at 3e-5"):
            build_s0_aggressive_trainer_config(
                plan,
                precision="fp32",
                learning_rate=2e-5,
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
