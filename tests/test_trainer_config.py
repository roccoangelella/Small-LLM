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
        config = TrainerConfig(precision="fp32", schedule="wsd", warmup_tokens=10,
            stable_tokens=10, decay_tokens=20, minimum_lr_ratio=0.1)
        optimizer = torch.optim.SGD([nn.Parameter(torch.ones(()))], lr=config.learning_rate)
        schedule = TokenLRScheduler(optimizer, config)
        warm = schedule.prepare_step(5); schedule.commit(5)
        stable = schedule.prepare_step(15); schedule.commit(15)
        decayed = schedule.prepare_step(30)
        self.assertLess(warm, stable)
        self.assertLess(decayed, stable)
        restored = TokenLRScheduler(optimizer, config)
        restored.load_state_dict(schedule.state_dict())
        self.assertEqual(restored.committed_tokens, 15)
