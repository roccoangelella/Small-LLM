"""CPU tests for Muon and AdamW effective-update telemetry."""

from __future__ import annotations

import math
import unittest

import torch

from tests.test_trainer_optimizer import _Model
from trainer.config import TrainerConfig
from trainer.engine import TrainerEngine
from trainer.optimizer_telemetry import InstrumentedHybridMuonAdamW


class OptimizerTelemetryTests(unittest.TestCase):
    def _optimizer(self) -> InstrumentedHybridMuonAdamW:
        model = _Model()
        engine = TrainerEngine(
            model,
            TrainerConfig(
                optimizer="hybrid_muon_adamw",
                precision="fp32",
                learning_rate=1e-3,
                weight_decay=0.1,
                muon_weight_decay=0.1,
                muon_update_rms=0.18,
            ),
            device="cpu",
        )
        self.assertIsInstance(engine.optimizer, InstrumentedHybridMuonAdamW)
        return engine.optimizer

    def test_step_reports_both_optimizer_branches(self) -> None:
        optimizer = self._optimizer()
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        statistics = optimizer.step_statistics()

        self.assertIn("muon", statistics)
        self.assertIn("adamw_decay", statistics)
        self.assertIn("adamw_no_decay", statistics)
        muon = statistics["muon"]
        self.assertTrue(
            math.isclose(muon["optimizer_direction_rms"], 0.18, rel_tol=1e-5)
        )
        self.assertGreater(muon["effective_update_rms"], 0.0)
        self.assertGreater(muon["effective_update_to_weight_ratio"], 0.0)
        self.assertIn(
            "blocks.0.mixer.q_proj.weight",
            muon["matrix_optimizer_direction_rms"],
        )
        self.assertIn(
            "blocks.0.ffn.down.weight",
            muon["matrix_effective_update_to_weight_ratio"],
        )
        for role in statistics.values():
            self.assertTrue(math.isfinite(role["pre_update_weight_rms"]))
            self.assertTrue(math.isfinite(role["optimizer_direction_rms"]))
            self.assertTrue(math.isfinite(role["effective_update_rms"]))
            self.assertTrue(math.isfinite(role["effective_update_to_weight_ratio"]))

    def test_statistics_are_not_checkpoint_state(self) -> None:
        optimizer = self._optimizer()
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        self.assertTrue(optimizer.step_statistics())
        state = optimizer.state_dict()
        self.assertNotIn("step_statistics", state)
        optimizer.clear_step_statistics()
        self.assertEqual(optimizer.step_statistics(), {})


if __name__ == "__main__":
    unittest.main()
