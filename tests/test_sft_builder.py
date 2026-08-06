from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from post_training.sft.builder import SFTDatasetBuilder
from post_training.sft.config import SFTDataConfig
from post_training.sft.schema import ChatMessage, ConversationRecord, TokenizedSFTRecord
from post_training.sft.storage import SFTShardReader


class CharacterEncoder:
    def encode(self, text: str) -> list[int]:
        return [ord(character) + 1 for character in text]


def conversations(source: str, count: int):
    for index in range(count):
        yield ConversationRecord(
            f"{source}-{index}",
            source,
            (
                ChatMessage("user", f"Question {index}"),
                ChatMessage("assistant", "Yes"),
            ),
        )


def replay(count: int):
    for index in range(count):
        yield TokenizedSFTRecord(
            f"replay-{index}",
            "climbmix-replay",
            "train",
            (1, 2, 3, 4, 5),
            (True, True, True, True),
        )


class BuilderTests(unittest.TestCase):
    def test_end_to_end_build_is_budget_configured_not_hard_coded(self) -> None:
        shares = {
            "smol-magpie-ultra-short": 0.75,
            "smol-contraints": 0.10,
            "smollm-rewrite-30k": 0.075,
            "smol-summarize-20k": 0.075,
        }
        config = SFTDataConfig(
            target_loss_tokens=160,
            optimizer_target_tokens=32,
            context_length=128,
            maximum_assistant_tokens=16,
            instruction_source_shares=shares,
            shuffle_buffer_records=8,
            shard_target_bytes=1024,
        )
        sources = {name: conversations(name, 50) for name in shares}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "sft"
            result = SFTDatasetBuilder(
                config,
                encoder=CharacterEncoder(),
            ).build(
                instruction_sources=sources,
                replay_source=replay(50),
                output_dir=output,
            )
            total = result["manifest"]["totals"]["loss_bearing_target_tokens"]
            self.assertLessEqual(total, 160)
            self.assertGreater(total, 0)
            reader = SFTShardReader(output)
            self.assertEqual(sum(reader.block_target_counts), total)
            self.assertTrue((output / "build-report.json").is_file())


if __name__ == "__main__":
    unittest.main()
