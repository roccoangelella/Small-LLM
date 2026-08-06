from __future__ import annotations

import unittest

import torch
from torch.nn import functional as F

from trainer.config import TrainerConfig
from trainer.engine import TrainerEngine
from trainer.types import TokenBatch
from tests.trainer_fixtures import TinyLM


class SFTTrainerIntegrationTests(unittest.TestCase):
    def test_engine_normalizes_by_active_targets_across_microbatches(self) -> None:
        torch.manual_seed(11)
        model = TinyLM()
        config = TrainerConfig(
            optimizer="adamw",
            microbatch_size=1,
            learning_rate=1e-3,
            weight_decay=0.0,
            precision="fp32",
            schedule="constant",
        )
        batch = TokenBatch(
            block_id=0,
            split="train",
            input_ids=torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long),
            labels=torch.tensor([[-100, 3, 4], [5, -100, -100]], dtype=torch.long),
            sequence_count=2,
            target_token_count=3,
        )
        with torch.no_grad():
            logits = model(batch.input_ids)
            expected = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                batch.labels.reshape(-1),
                reduction="sum",
            ) / 3
        engine = TrainerEngine(model, config, device="cpu")
        metrics = engine.train_batch(batch)
        self.assertAlmostEqual(metrics.loss, float(expected), places=6)
        self.assertEqual(engine.consumed_tokens, 3)


if __name__ == "__main__":
    unittest.main()
