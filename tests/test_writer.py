"""Buffered binary-writer behavior independent of the build loop."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from dataset.src.bitio import decode_uint16_le
from dataset.src.writer import BinaryCorpusWriter


class BinaryWriterTest(unittest.TestCase):
    def test_buffers_soft_flushes_and_checkpoints_both_outputs(self) -> None:
        with tempfile.TemporaryDirectory(prefix="climbmix-writer-") as tmp:
            root = Path(tmp)
            train = root / "train.bin"
            validation = root / "validation.bin"
            writer = BinaryCorpusWriter(
                train, validation, buffer_bytes=8, resume_sizes=(0, 0)
            )
            try:
                writer.append(validation=False, tokens=[1, 2])
                self.assertEqual(train.stat().st_size, 0)
                writer.append(validation=False, tokens=[3])
                # 10 buffered bytes crossed the 8-byte soft-flush threshold.
                self.assertEqual(train.stat().st_size, 10)

                writer.append(validation=True, tokens=[4])
                self.assertEqual(validation.stat().st_size, 0)
                uncommitted = writer.flush_uncommitted()
                self.assertEqual(uncommitted, (10, 4))
                self.assertEqual(writer.written_since_checkpoint, 14)

                confirmed = writer.checkpoint()
                self.assertEqual(confirmed, (10, 4))
                self.assertEqual(writer.written_since_checkpoint, 0)
            finally:
                writer.close()

            self.assertEqual(decode_uint16_le(train.read_bytes()), [1, 2, 50256, 3, 50256])
            self.assertEqual(decode_uint16_le(validation.read_bytes()), [4, 50256])

    def test_constructor_refuses_sizes_that_do_not_match_disk(self) -> None:
        with tempfile.TemporaryDirectory(prefix="climbmix-writer-size-") as tmp:
            root = Path(tmp)
            train = root / "train.bin"
            validation = root / "validation.bin"
            train.write_bytes(b"\x01\x00")
            validation.write_bytes(b"")
            with self.assertRaises(RuntimeError):
                BinaryCorpusWriter(
                    train, validation, buffer_bytes=8, resume_sizes=(0, 0)
                )


if __name__ == "__main__":
    unittest.main()
