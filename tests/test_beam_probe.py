"""Regression guards for the Beam startup microbatch qualification."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BEAM = ROOT / "beam"


def _load_profiles():
    spec = importlib.util.spec_from_file_location("small_llm_beam_probe_profiles", BEAM / "profiles.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class BeamProbeTest(unittest.TestCase):
    def test_probe_candidates_are_8_12_16(self) -> None:
        profiles = _load_profiles()
        self.assertEqual(profiles.MICROBATCH_CANDIDATES, (8, 12, 16))
        self.assertEqual(profiles.SEQUENCES_PER_BLOCK, 64)

    def test_probe_measures_throughput_and_memory(self) -> None:
        source = (BEAM / "runtime.py").read_text(encoding="utf-8")
        self.assertIn('"tokens_per_second"', source)
        self.assertIn("median_tokens_per_second", source)
        self.assertIn("peak_reserved_memory_bytes", source)
        self.assertIn("fastest_safe_measured_candidate", source)


if __name__ == "__main__":
    unittest.main()
