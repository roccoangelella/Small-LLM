import math
import unittest

import torch
from torch import nn

from trainer import TokenLRScheduler, TrainerConfig
from trainer.fresh_decay import fresh_aggressive_decay_plan


class ConfigAndScheduleTests(unittest.TestCase):
    def test_config_validation_and_wsd_schedule(self):
        with self.assertRaises(ValueError):
            TrainerConfig(microbatch_size=0)
        with self.assertRaises(ValueError):
            TrainerConfig(schedule="constant", warmup_tokens=10)
        config = TrainerConfig(
            precision="fp32",
            schedule="wsd",
            warmup_tokens=10,
            stable_tokens=10,
            decay_tokens=20,
            minimum_lr_ratio=0.1,
        )
        serialized = config.as_dict()
        self.assertNotIn("schedule_anchor_tokens", serialized)
        self.assertNotIn("cooldown_start_tokens", serialized)
        self.assertNotIn("settle_tokens", serialized)
        self.assertNotIn("settle_lr_ratio", serialized)
        self.assertNotIn("base_power", serialized)
        optimizer = torch.optim.SGD([nn.Parameter(torch.ones(()))], lr=config.learning_rate)
        schedule = TokenLRScheduler(optimizer, config)
        warm = schedule.prepare_step(5)
        schedule.commit(5)
        stable = schedule.prepare_step(15)
        schedule.commit(15)
        decayed = schedule.prepare_step(30)
        self.assertLess(warm, stable)
        self.assertLess(decayed, stable)
        restored = TokenLRScheduler(optimizer, config)
        restored.load_state_dict(schedule.state_dict())
        self.assertEqual(restored.committed_tokens, 15)

    def test_wsqd_continuation_anchor_base_and_linear_cooldown(self):
        anchor = 2_031_616_000
        cooldown_start = 9_599_975_424
        decay = 400_031_744
        config = TrainerConfig(
            precision="fp32",
            schedule="wsqd",
            learning_rate=3e-4,
            decay_tokens=decay,
            minimum_lr_ratio=0.1,
            schedule_anchor_tokens=anchor,
            cooldown_start_tokens=cooldown_start,
        )
        serialized = config.as_dict()
        self.assertEqual(serialized["schedule_anchor_tokens"], anchor)
        self.assertEqual(serialized["cooldown_start_tokens"], cooldown_start)
        self.assertNotIn("settle_tokens", serialized)
        self.assertNotIn("settle_lr_ratio", serialized)
        self.assertNotIn("base_power", serialized)
        optimizer = torch.optim.SGD([nn.Parameter(torch.ones(()))], lr=config.learning_rate)
        schedule = TokenLRScheduler(optimizer, config)

        self.assertEqual(schedule.prepare_step(anchor), 3e-4)
        middle = (anchor + cooldown_start) // 2
        expected_middle = 3e-4 * math.sqrt(anchor / middle)
        self.assertTrue(math.isclose(schedule.prepare_step(middle), expected_middle, rel_tol=1e-12))

        cooldown_lr = schedule.prepare_step(cooldown_start)
        expected_cooldown_lr = 3e-4 * math.sqrt(anchor / cooldown_start)
        self.assertTrue(math.isclose(schedule.prepare_step(cooldown_start), expected_cooldown_lr, rel_tol=1e-12))

        halfway = cooldown_start + decay // 2
        expected_halfway = 3e-5 + (expected_cooldown_lr - 3e-5) * 0.5
        self.assertTrue(math.isclose(schedule.prepare_step(halfway), expected_halfway, rel_tol=1e-12))
        self.assertTrue(math.isclose(schedule.prepare_step(cooldown_start + decay), 3e-5, rel_tol=1e-12))

        schedule.commit(middle)
        restored = TokenLRScheduler(optimizer, config)
        restored.load_state_dict(schedule.state_dict())
        self.assertEqual(restored.committed_tokens, middle)
        self.assertTrue(math.isclose(restored.last_lr, expected_middle, rel_tol=1e-12))

    def test_aggressive_wsqd_settle_invsqrt_and_terminal_floor(self):
        anchor = 2_031_616_000
        settle = 300_023_808
        settle_end = anchor + settle
        cooldown_start = 9_599_975_424
        decay = 400_031_744
        config = TrainerConfig(
            precision="fp32",
            schedule="wsqd",
            learning_rate=3e-4,
            decay_tokens=decay,
            minimum_lr_ratio=0.05,
            schedule_anchor_tokens=anchor,
            cooldown_start_tokens=cooldown_start,
            settle_tokens=settle,
            settle_lr_ratio=0.5,
        )
        optimizer = torch.optim.SGD([nn.Parameter(torch.ones(()))], lr=config.learning_rate)
        schedule = TokenLRScheduler(optimizer, config)

        self.assertEqual(schedule.prepare_step(anchor), 3e-4)
        halfway_settle = anchor + settle // 2
        self.assertTrue(math.isclose(schedule.prepare_step(halfway_settle), 2.25e-4, rel_tol=1e-12))
        self.assertTrue(math.isclose(schedule.prepare_step(settle_end), 1.5e-4, rel_tol=1e-12))

        middle = 5_000_000_000
        expected_middle = 1.5e-4 * math.sqrt(settle_end / middle)
        self.assertTrue(math.isclose(schedule.prepare_step(middle), expected_middle, rel_tol=1e-12))

        cooldown_lr = 1.5e-4 * math.sqrt(settle_end / cooldown_start)
        self.assertTrue(math.isclose(schedule.prepare_step(cooldown_start), cooldown_lr, rel_tol=1e-12))
        halfway_cooldown = cooldown_start + decay // 2
        expected_halfway = 1.5e-5 + (cooldown_lr - 1.5e-5) * 0.5
        self.assertTrue(
            math.isclose(schedule.prepare_step(halfway_cooldown), expected_halfway, rel_tol=1e-12)
        )
        self.assertTrue(
            math.isclose(schedule.prepare_step(cooldown_start + decay), 1.5e-5, rel_tol=1e-12)
        )

    def test_calibrated_power_decay_hits_1e5_then_5e6(self):
        anchor = 2_031_616_000
        settle = 300_023_808
        settle_end = anchor + settle
        cooldown_start = 9_599_975_424
        decay = 400_031_744
        settle_lr = 1e-4
        cooldown_lr = 1e-5
        terminal_lr = 5e-6
        power = math.log(settle_lr / cooldown_lr) / math.log(cooldown_start / settle_end)
        config = TrainerConfig(
            precision="fp32",
            schedule="wsqd",
            learning_rate=3e-4,
            decay_tokens=decay,
            minimum_lr_ratio=terminal_lr / 3e-4,
            schedule_anchor_tokens=anchor,
            cooldown_start_tokens=cooldown_start,
            settle_tokens=settle,
            settle_lr_ratio=settle_lr / 3e-4,
            base_power=power,
        )
        serialized = config.as_dict()
        self.assertTrue(math.isclose(serialized["base_power"], power, rel_tol=1e-12))
        optimizer = torch.optim.SGD([nn.Parameter(torch.ones(()))], lr=config.learning_rate)
        schedule = TokenLRScheduler(optimizer, config)

        self.assertTrue(math.isclose(schedule.prepare_step(settle_end), settle_lr, rel_tol=1e-12))
        self.assertTrue(math.isclose(schedule.prepare_step(cooldown_start), cooldown_lr, rel_tol=1e-12))
        halfway = cooldown_start + decay // 2
        self.assertTrue(math.isclose(schedule.prepare_step(halfway), 7.5e-6, rel_tol=1e-12))
        self.assertTrue(math.isclose(schedule.prepare_step(cooldown_start + decay), terminal_lr, rel_tol=1e-12))

    def test_fresh_aggressive_wsqd_starts_at_zero_and_hits_all_landmarks(self):
        total = 200_099_738
        peak = 3e-5
        plan = fresh_aggressive_decay_plan(total)
        config = TrainerConfig(
            precision="fp32",
            learning_rate=peak,
            **plan.trainer_kwargs(),
        )
        optimizer = torch.optim.SGD([nn.Parameter(torch.ones(()))], lr=peak)
        schedule = TokenLRScheduler(optimizer, config)

        self.assertEqual(plan.warmup_tokens, 10_004_986)
        self.assertEqual(plan.settle_tokens, 6_002_992)
        self.assertEqual(plan.cooldown_start_tokens, 192_095_749)
        self.assertEqual(plan.decay_tokens, 8_003_989)
        self.assertTrue(math.isclose(schedule.prepare_step(plan.warmup_tokens), peak, rel_tol=1e-12))
        self.assertLess(schedule.prepare_step(plan.warmup_tokens // 2), peak)
        self.assertTrue(
            math.isclose(schedule.prepare_step(plan.settle_end_tokens), 1e-5, rel_tol=1e-12)
        )
        self.assertTrue(
            math.isclose(schedule.prepare_step(plan.cooldown_start_tokens), 1e-6, rel_tol=1e-12)
        )
        self.assertTrue(
            math.isclose(schedule.prepare_step(total), 5e-7, rel_tol=1e-12)
        )

    def test_fresh_pretraining_uses_same_geometry_at_pretraining_lr_scale(self):
        total = 10_000_007_168
        peak = 3e-4
        plan = fresh_aggressive_decay_plan(total)
        landmarks = plan.lr_landmarks(peak)
        self.assertTrue(math.isclose(landmarks["peak_lr"], 3e-4, rel_tol=1e-12))
        self.assertTrue(math.isclose(landmarks["settle_lr"], 1e-4, rel_tol=1e-12))
        self.assertTrue(math.isclose(landmarks["cooldown_start_lr"], 1e-5, rel_tol=1e-12))
        self.assertTrue(math.isclose(landmarks["final_lr"], 5e-6, rel_tol=1e-12))
        self.assertEqual(plan.schedule_anchor_tokens, plan.warmup_tokens)
        self.assertEqual(plan.cooldown_start_tokens + plan.decay_tokens, total)

    def test_fresh_wsqd_rejects_anchor_that_is_not_warmup_endpoint(self):
        with self.assertRaisesRegex(ValueError, "schedule_anchor_tokens == warmup_tokens"):
            TrainerConfig(
                schedule="wsqd",
                warmup_tokens=100,
                schedule_anchor_tokens=101,
                cooldown_start_tokens=900,
                decay_tokens=100,
            )

    def test_wsqd_validation_rejects_rising_terminal_floor(self):
        with self.assertRaises(ValueError):
            TrainerConfig(
                schedule="wsqd",
                decay_tokens=100,
                schedule_anchor_tokens=100,
                cooldown_start_tokens=10_000,
                minimum_lr_ratio=0.5,
            )

    def test_wsqd_settle_requires_room_before_cooldown(self):
        with self.assertRaises(ValueError):
            TrainerConfig(
                schedule="wsqd",
                decay_tokens=100,
                schedule_anchor_tokens=100,
                cooldown_start_tokens=200,
                settle_tokens=100,
                settle_lr_ratio=0.5,
            )
