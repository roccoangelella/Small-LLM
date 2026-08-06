from __future__ import annotations

import unittest

from post_training.sft.mixture import BufferedShuffle, TargetTokenMixer, build_atomic_blocks
from post_training.sft.schema import TokenizedSFTRecord


def records(source: str, count: int, targets: int = 10):
    for index in range(count):
        yield TokenizedSFTRecord(
            f"{source}-{index}",
            source,
            "train",
            tuple(range(targets + 1)),
            tuple(True for _ in range(targets)),
        )


class MixtureTests(unittest.TestCase):
    def test_buffered_shuffle_is_repeatable(self) -> None:
        first = [
            item.record_id
            for item in BufferedShuffle(records("a", 20), seed=7, buffer_size=5)
        ]
        second = [
            item.record_id
            for item in BufferedShuffle(records("a", 20), seed=7, buffer_size=5)
        ]
        self.assertEqual(first, second)
        self.assertEqual(len(set(first)), 20)

    def test_mixer_tracks_target_token_shares(self) -> None:
        mixed = list(
            TargetTokenMixer(
                {"a": records("a", 100), "b": records("b", 100)},
                {"a": 0.75, "b": 0.25},
                seed=17,
                target_loss_tokens=400,
            )
        )
        counts = {
            source: sum(item.target_token_count for item in mixed if item.source == source)
            for source in ("a", "b")
        }
        self.assertEqual(counts, {"a": 300, "b": 100})

    def test_blocks_do_not_cross_target_ceiling(self) -> None:
        blocks = list(build_atomic_blocks(records("a", 10, 7), target_tokens_per_block=20))
        self.assertTrue(all(block.target_token_count <= 20 for block in blocks))
        self.assertEqual(sum(block.target_token_count for block in blocks), 70)


if __name__ == "__main__":
    unittest.main()
