"""MoE evaluation-loader identity tests."""

from __future__ import annotations

import unittest

from MOE_model.config import MoEModelConfig
from MOE_model.evaluation import normalize_moe_model_config


class TestMoEEvaluation(unittest.TestCase):
    def test_checkpoint_model_config_round_trips_exactly(self) -> None:
        original = MoEModelConfig.smoke()
        payload = original.as_dict()
        # JSON checkpoint metadata may materialize tuples as lists.
        payload["moe_layer_indices"] = list(payload["moe_layer_indices"])
        dense = dict(payload["dense"])
        dense["layer_pattern"] = list(dense["layer_pattern"])
        payload["dense"] = dense

        restored = normalize_moe_model_config(payload)
        self.assertEqual(restored.as_dict(), original.as_dict())

    def test_eval_loader_rejects_dense_only_config(self) -> None:
        with self.assertRaises(RuntimeError):
            normalize_moe_model_config({"d_model": 512, "n_layers": 20})

    def test_eval_loader_fails_closed_on_topk_mismatch(self) -> None:
        payload = MoEModelConfig.smoke().as_dict()
        payload["top_k"] = 2
        with self.assertRaises(ValueError):
            normalize_moe_model_config(payload)


if __name__ == "__main__":
    unittest.main()
