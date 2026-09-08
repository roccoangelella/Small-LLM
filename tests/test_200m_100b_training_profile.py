"""Scientific contract tests for the active 200M/100B trajectory."""

from __future__ import annotations

import math
import unittest

from model.config import ModelConfig
from trainer.config import TrainerConfig
from trainer.fresh_decay import (
    PRETRAIN_200M_100B_ANCHOR_TOKENS,
    PRETRAIN_200M_100B_COOLDOWN_START_TOKENS,
    PRETRAIN_200M_100B_PEAK_LR,
    PRETRAIN_200M_100B_SETTLE_END_TOKENS,
    PRETRAIN_200M_100B_TOTAL_TOKENS,
    PRETRAIN_200M_100B_WARMUP_TOKENS,
    pretrain_200m_100b_decay_plan,
)


class Model200M100BContractTests(unittest.TestCase):
    def test_expanded_geometry_is_width_scaled_not_deeper(self) -> None:
        base = ModelConfig.substantive()
        expanded = ModelConfig.expanded()

        self.assertEqual(expanded.n_layers, base.n_layers)
        self.assertEqual(expanded.n_layers, 20)
        self.assertEqual(expanded.d_model, 768)
        self.assertEqual(expanded.d_ff, 2_048)
        self.assertEqual(expanded.n_heads, 12)
        self.assertEqual(expanded.head_dim, 64)
        self.assertEqual(expanded.gdn_num_key_heads, 12)
        self.assertEqual(expanded.gdn_num_value_heads, 12)
        self.assertEqual(expanded.gdn_key_dim, 64)
        self.assertEqual(expanded.gdn_value_dim, 64)
        self.assertEqual(
            expanded.layer_kinds,
            ("gdn", "gdn", "gdn", "mha") * 5,
        )
        self.assertEqual(expanded.max_seq_len, 2_048)
        self.assertEqual(expanded.semantic_vocab_size, 50_257)
        self.assertEqual(expanded.padded_vocab_size, 50_304)
        self.assertEqual(expanded.dropout, 0.0)

    def test_exact_wsqd_schedule_matches_frozen_anchors(self) -> None:
        plan = pretrain_200m_100b_decay_plan()

        self.assertEqual(plan.total_tokens, PRETRAIN_200M_100B_TOTAL_TOKENS)
        self.assertEqual(plan.warmup_tokens, PRETRAIN_200M_100B_WARMUP_TOKENS)
        self.assertEqual(plan.schedule_anchor_tokens, PRETRAIN_200M_100B_ANCHOR_TOKENS)
        self.assertEqual(plan.settle_end_tokens, PRETRAIN_200M_100B_SETTLE_END_TOKENS)
        self.assertEqual(plan.cooldown_start_tokens, PRETRAIN_200M_100B_COOLDOWN_START_TOKENS)
        self.assertEqual(plan.decay_tokens, 4_000_317_440)
        self.assertEqual(plan.peak_tokens, 15_316_156_416)
        self.assertEqual(plan.settle_tokens, 3_000_238_080)
        self.assertAlmostEqual(plan.base_power, 1.6270515945225403, places=12)

        landmarks = plan.lr_landmarks(PRETRAIN_200M_100B_PEAK_LR)
        self.assertTrue(math.isclose(landmarks["peak_lr"], 3.5e-4, rel_tol=0, abs_tol=1e-16))
        self.assertTrue(math.isclose(landmarks["settle_lr"], 8e-5, rel_tol=0, abs_tol=1e-16))
        self.assertTrue(math.isclose(landmarks["cooldown_start_lr"], 8e-6, rel_tol=0, abs_tol=1e-16))
        self.assertTrue(math.isclose(landmarks["final_lr"], 4e-6, rel_tol=0, abs_tol=1e-16))

    def test_wsqd_plan_validates_as_trainer_config(self) -> None:
        plan = pretrain_200m_100b_decay_plan()
        config = TrainerConfig(
            optimizer="hybrid_muon_adamw",
            microbatch_size=1,
            learning_rate=PRETRAIN_200M_100B_PEAK_LR,
            weight_decay=0.1,
            muon_momentum=0.95,
            muon_lr_multiplier=1.0,
            muon_update_rms=0.18,
            muon_weight_decay=0.1,
            max_grad_norm=1.0,
            precision="fp16",
            **plan.trainer_kwargs(),
        )

        self.assertEqual(config.schedule, "wsqd")
        self.assertEqual(config.warmup_tokens, 5_000_003_584)
        self.assertEqual(config.stable_tokens, 15_316_156_416)
        self.assertEqual(config.schedule_anchor_tokens, 20_316_160_000)
        self.assertEqual(config.settle_tokens, 3_000_238_080)
        self.assertEqual(config.cooldown_start_tokens, 95_999_754_240)
        self.assertEqual(config.decay_tokens, 4_000_317_440)
        self.assertAlmostEqual(config.minimum_lr_ratio, 2.0 / 175.0)


if __name__ == "__main__":
    unittest.main()
