"""Contract tests for the opt-in live incremental Modal smoke."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "modal"))
import incremental_smoke_support as smoke  # noqa: E402


class ModalIncrementalSmokeTests(unittest.TestCase):
    def test_geometry(self) -> None:
        self.assertEqual(smoke.SMOKE_CONTEXT_LENGTH, 2048)
        self.assertEqual(smoke.SMOKE_SEQUENCES_PER_BLOCK, 64)
        self.assertEqual(smoke.SMOKE_BLOCK_BYTES, (2048 + 1) * 64 * 2)
        self.assertEqual(smoke.SMOKE_TARGET_SHARD_BYTES, smoke.SMOKE_BLOCK_BYTES * 16)
        self.assertEqual(smoke.SMOKE_FIRST_SEGMENT_STEPS, 16)
        self.assertGreater(smoke.SMOKE_TRAIN_BLOCKS, smoke.SMOKE_TOTAL_EXERCISED_STEPS)


if __name__ == "__main__":
    unittest.main()
