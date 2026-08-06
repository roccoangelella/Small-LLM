from __future__ import annotations

import unittest

import torch

from post_training.sft.schema import SFTBlock, TokenizedSFTRecord
from trainer.types import TokenBatch


class SFTSchemaTests(unittest.TestCase):
    def test_right_padding_is_masked_and_active_count_is_exact(self) -> None:
        first = TokenizedSFTRecord(
            "first", "a", "train", (1, 2, 3, 4), (False, True, True)
        )
        second = TokenizedSFTRecord(
            "second", "b", "train", (5, 6, 7), (True, False)
        )
        batch = SFTBlock(0, "train", (first, second)).to_token_batch(
            pad_token_id=99
        )
        self.assertEqual(batch.input_ids.tolist(), [[1, 2, 3], [5, 6, 99]])
        self.assertEqual(batch.labels.tolist(), [[-100, 3, 4], [6, -100, -100]])
        self.assertEqual(batch.target_token_count, 3)

    def test_token_batch_counts_only_non_masked_labels(self) -> None:
        batch = TokenBatch(
            block_id=0,
            split="train",
            input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long),
            labels=torch.tensor([[-100, 2, 3]], dtype=torch.long),
            sequence_count=1,
            target_token_count=2,
        )
        self.assertEqual(batch.target_token_count, 2)
        with self.assertRaises(ValueError):
            TokenBatch(
                block_id=0,
                split="train",
                input_ids=batch.input_ids,
                labels=batch.labels,
                sequence_count=1,
                target_token_count=3,
            )


if __name__ == "__main__":
    unittest.main()
