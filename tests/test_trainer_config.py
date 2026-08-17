import math
import unittest

import torch
from torch import nn

from trainer import TokenLRScheduler, TrainerConfig


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
        optimizer = torch.optim.SGD([nn.Parameter(torch.ones(()))], lr=config.learning_rate)
        schedule = TokenLRScheduler(optimizer, config)

        self.assertEqual(schedule.prepare_step(anchor), 3e-4)
        middle = (anchor + cooldown_start) // 2
        expected_middle = 3e-4 * math.sqrt(anchor / middle)
        self.assertTrue(math.isclose(schedule.prepare_step(middle), expected_middle, rel_tol=1e-12))

        cooldown_lr = schedule.prepare_step(cooldown_start)
        expected_cooldown_lr = 3e-4 * math.sqrt(anchor / cooldown_start)
        self.assertTrue(math.isclose(cooldown_lr, expected_cooldown_lr, rel_tol=1e-12))

        halfway = cooldown_start + decay // 2
        expected_halfway = 3e-5 + (expected_cooldown_lr - 3e-5) * 0.5
        self.assertTrue(math.isclose(schedule.prepare_step(halfway), expected_halfway, rel_tol=1e-12))
        self.assertTrue(math.isclose(schedule.prepare_step(cooldown_start + decay), 3e-5, rel_tol=1e-12))

        schedule.commit(middle)
        restored = TokenLRScheduler(optimizer, config)
        restored.load_state_dict(schedule.state_dict())
        self.assertEqual(restored.committed_tokens, middle)
        self.assertTrue(math.isclose(restored.last_lr, expected_middle, rel_tol=1e-12))

    def test_wsqd_validation_rejects_rising_terminal_floor(self):
        with self.assertRaises(ValueError):
            TrainerConfig(
                schedule="wsqd",
                decay_tokens=100,
                schedule_anchor_tokens=100,
                cooldown_start_tokens=10_000,
                minimum_lr_ratio=0.5,
            )
