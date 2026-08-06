from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from post_training.sft.config import SFTDataConfig
from post_training.sft.schema import SFTBlock, TokenizedSFTRecord
from post_training.sft.storage import (
    SFTDatasetWriter,
    SFTShardReader,
    decode_sft_block,
    encode_sft_block,
)


def record(identity: str, source: str, targets: tuple[bool, ...]):
    return TokenizedSFTRecord(
        identity,
        source,
        "train",
        tuple(range(len(targets) + 1)),
        targets,
    )


class StorageTests(unittest.TestCase):
    def test_block_binary_roundtrip(self) -> None:
        block = SFTBlock(
            0,
            "train",
            (
                record("a", "source-a", (False, True, True)),
                record("b", "source-b", (True, False)),
            ),
        )
        restored = decode_sft_block(encode_sft_block(block))
        self.assertEqual(restored, block)

    def test_dataset_verification_and_resume_cursor(self) -> None:
        config = SFTDataConfig(
            target_loss_tokens=6,
            optimizer_target_tokens=4,
            context_length=16,
            maximum_assistant_tokens=8,
            instruction_share=0.5,
            replay_share=0.5,
            instruction_source_shares={"source-a": 1.0},
            shuffle_buffer_records=2,
            shard_target_bytes=128,
        )
        blocks = (
            SFTBlock(0, "train", (record("a", "source-a", (True, True)),)),
            SFTBlock(1, "train", (record("b", "climbmix-replay", (True, True)),)),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "sft"
            SFTDatasetWriter(root, config).write(blocks)
            reader = SFTShardReader(root)
            first = reader.next_batch()
            self.assertEqual(first.block_id, 0)
            reader.acknowledge(0)
            state = reader.pipeline_state()

            resumed = SFTShardReader(root)
            resumed.load_pipeline_state(state)
            second = resumed.next_batch()
            self.assertEqual(second.block_id, 1)

            shard = next(root.glob("*.sft"))
            data = bytearray(shard.read_bytes())
            data[-1] ^= 1
            shard.write_bytes(data)
            with self.assertRaises(RuntimeError):
                list(SFTShardReader(root).iter_from_start())


if __name__ == "__main__":
    unittest.main()
