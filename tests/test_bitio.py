"""Little-endian uint16 serialization and end-of-document handling."""

from __future__ import annotations

import unittest

from dataset import config
from dataset.src.bitio import decode_uint16_le, insert_eod, tokens_to_uint16_le_bytes


class BitIoTest(unittest.TestCase):
    def test_explicit_little_endian_round_trip(self) -> None:
        tokens = [0, 1, 256, 65535, 50256, 50255]
        encoded = tokens_to_uint16_le_bytes(tokens)
        self.assertEqual(len(encoded), len(tokens) * 2)
        # Explicit little-endian: 0 -> 00 00, 1 -> 01 00, 256 -> 00 01,
        # 65535 -> ff ff, 50256 (0xC450) -> 50 c4, 50255 (0xC44F) -> 4f c4.
        self.assertEqual(encoded[0:2], b"\x00\x00")
        self.assertEqual(encoded[2:4], b"\x01\x00")
        self.assertEqual(encoded[4:6], b"\x00\x01")
        self.assertEqual(encoded[6:8], b"\xff\xff")
        self.assertEqual(encoded[8:10], b"\x50\xc4")
        self.assertEqual(encoded[10:12], b"\x4f\xc4")
        self.assertEqual(decode_uint16_le(encoded), tokens)

    def test_no_duplicate_eod_when_document_already_ends_with_eod(self) -> None:
        self.assertEqual(insert_eod([1, 2, config.EOD_TOKEN_ID]), [1, 2, config.EOD_TOKEN_ID])

    def test_inserts_eod_otherwise(self) -> None:
        self.assertEqual(insert_eod([1, 2, 3]), [1, 2, 3, config.EOD_TOKEN_ID])

    def test_empty_inserts_single_eod(self) -> None:
        self.assertEqual(insert_eod([]), [config.EOD_TOKEN_ID])

    def test_byte_length_not_divisible_by_two_raises(self) -> None:
        with self.assertRaises(ValueError):
            decode_uint16_le(b"\x01")


if __name__ == "__main__":
    unittest.main()