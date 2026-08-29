from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
KAGGLE = ROOT / "kaggle"
if str(KAGGLE) not in sys.path:
    sys.path.insert(0, str(KAGGLE))

import launch_sft  # noqa: E402


class SFTHuggingFaceUploadHardeningTests(unittest.TestCase):
    def test_canonical_launcher_disables_xet_and_progress_bars(self) -> None:
        self.assertEqual(os.environ.get("HF_HUB_DISABLE_XET"), "1")
        self.assertEqual(os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS"), "1")

    def test_hardening_overrides_unsafe_operator_values(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "HF_HUB_DISABLE_XET": "0",
                "HF_HUB_DISABLE_PROGRESS_BARS": "0",
            },
            clear=False,
        ):
            launch_sft._apply_hf_upload_hardening()
            self.assertEqual(os.environ["HF_HUB_DISABLE_XET"], "1")
            self.assertEqual(os.environ["HF_HUB_DISABLE_PROGRESS_BARS"], "1")


if __name__ == "__main__":
    unittest.main()
