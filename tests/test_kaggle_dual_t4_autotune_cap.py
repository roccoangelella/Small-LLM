"""CPU-only tests for the qualification-only Triton autotune cap."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KAGGLE = ROOT / "kaggle"
if str(KAGGLE) not in sys.path:
    sys.path.insert(0, str(KAGGLE))

SPEC = importlib.util.spec_from_file_location(
    "dual_t4_watchdog_autotune_test",
    KAGGLE / "qualify_dual_t4_watchdog.py",
)
assert SPEC is not None and SPEC.loader is not None
watchdog = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(watchdog)


class DualT4AutotuneCapTests(unittest.TestCase):
    def test_default_cap_is_six(self) -> None:
        self.assertEqual(watchdog.DEFAULT_AUTOTUNE_CONFIG_CAP, 6)

    def test_representative_indices_span_full_candidate_list(self) -> None:
        self.assertEqual(
            watchdog._representative_config_indices(36, 6),
            [0, 7, 14, 21, 28, 35],
        )

    def test_small_candidate_lists_are_not_pruned(self) -> None:
        self.assertEqual(watchdog._representative_config_indices(4, 6), [0, 1, 2, 3])

    def test_single_candidate_cap_uses_middle_candidate(self) -> None:
        self.assertEqual(watchdog._representative_config_indices(9, 1), [4])


if __name__ == "__main__":
    unittest.main()
