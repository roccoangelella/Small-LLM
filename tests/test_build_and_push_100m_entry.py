"""Regression test for kagglehub round-trip transport artifacts."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "kaggle" / "build_and_push_100m_entry.py"
SPEC = importlib.util.spec_from_file_location("small_llm_build_push_100m_entry", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
entry = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = entry
SPEC.loader.exec_module(entry)


class BuildAndPush100MEntryTests(unittest.TestCase):
    def test_top_level_kaggle_transport_archive_is_not_dataset_content(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "manifest.json").write_bytes(b"manifest")
            baseline = entry.dataset_tree_identity(root)
            (root / "1.archive").write_bytes(b"kagglehub transport")
            self.assertEqual(entry.dataset_tree_identity(root), baseline)

    def test_nested_archive_named_payload_is_still_hashed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "train").mkdir()
            (root / "train" / "1.archive").write_bytes(b"dataset payload")
            first = entry.dataset_tree_identity(root)
            (root / "train" / "1.archive").write_bytes(b"changed payload")
            self.assertNotEqual(entry.dataset_tree_identity(root), first)


if __name__ == "__main__":
    unittest.main()
