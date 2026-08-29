from __future__ import annotations

import math
import unittest

import torch

from post_training.sft.config import (
    S0_AGGRESSIVE_COOLDOWN_START_LR,
    S0_AGGRESSIVE_FINAL_LR,
    S0_AGGRESSIVE_PEAK_LR,
    S0_AGGRESSIVE_SETTLE_LR,
    SFTSchedulePlan,
)
from post_training.sft.s0_peak3000_schedule import (
    S0_PEAK3000_PEAK_END_STEP,
    S0_PEAK3000_SETTLE_STEPS,
    S0_PEAK3000_WARMUP_STEPS,
    build_s0_peak3000_trainer_config,
)
from trainer import TokenLRScheduler


class SFTPeak3000ScheduleTests(unittest.TestCase):
    @staticmethod
    def _counts() -> tuple[int, ...]:
        # Match the completed 10% run's exact observed horizon and 6,219 blocks
        # while keeping this unit fixture compact and deterministic.
        total = 200_099_738
        steps = 6_219
        base, remainder = divmod(total, steps)
        return tuple(
            base + (1 if index < remainder else 0)
            for index in range(steps)
        )

    def test_peak_is_reached_at_step_64_and_held_through_step_3000(self) -> None:
        counts = self._counts()
        schedule = SFTSchedulePlan.from_block_target_counts(counts)
        config = build_s0_peak3000_trainer_config(schedule, precision="fp32")

        warmup_end = sum(counts[:S0_PEAK3000_WARMUP_STEPS])
        peak_end = sum(counts[:S0_PEAK3000_PEAK_END_STEP])
        settle_end_step = S0_PEAK3000_PEAK_END_STEP + S0_PEAK3000_SETTLE_STEPS
        settle_end = sum(counts[:settle_end_step])

        self.assertEqual(config.warmup_tokens, warmup_end)
        self.assertEqual(config.schedule_anchor_tokens, peak_end)
        self.assertEqual(config.stable_tokens, peak_end - warmup_end)
        self.assertEqual(config.settle_tokens, settle_end - peak_end)

        parameter = torch.nn.Parameter(torch.ones(()))
        scheduler = TokenLRScheduler(
            torch.optim.SGD([parameter], lr=config.learning_rate),
            config,
        )

        self.assertTrue(
            math.isclose(
                scheduler.prepare_step(warmup_end),
                S0_AGGRESSIVE_PEAK_LR,
                rel_tol=1e-12,
            )
        )
        self.assertTrue(
            math.isclose(
                scheduler.prepare_step(peak_end),
                S0_AGGRESSIVE_PEAK_LR,
                rel_tol=1e-12,
            )
        )
        first_decay_step = sum(counts[: S0_PEAK3000_PEAK_END_STEP + 1])
        self.assertLess(
            scheduler.prepare_step(first_decay_step),
            S0_AGGRESSIVE_PEAK_LR,
        )
        self.assertTrue(
            math.isclose(
                scheduler.prepare_step(settle_end),
                S0_AGGRESSIVE_SETTLE_LR,
                rel_tol=1e-12,
            )
        )

    def test_aggressive_tail_keeps_low_lr_landmarks(self) -> None:
        counts = self._counts()
        total = sum(counts)
        schedule = SFTSchedulePlan.from_block_target_counts(counts)
        config = build_s0_peak3000_trainer_config(schedule, precision="fp32")
        parameter = torch.nn.Parameter(torch.ones(()))
        scheduler = TokenLRScheduler(
            torch.optim.SGD([parameter], lr=config.learning_rate),
            config,
        )

        self.assertEqual(config.cooldown_start_tokens + config.decay_tokens, total)
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

    def test_short_horizon_is_rejected(self) -> None:
        counts = tuple([32_768] * (S0_PEAK3000_PEAK_END_STEP + S0_PEAK3000_SETTLE_STEPS))
        schedule = SFTSchedulePlan.from_block_target_counts(counts)
        with self.assertRaisesRegex(ValueError, "requires more than"):
            build_s0_peak3000_trainer_config(schedule, precision="fp32")


if __name__ == "__main__":
    unittest.main()
