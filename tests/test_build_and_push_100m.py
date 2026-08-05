"""Offline tests for the one-command 100M Kaggle publication suite."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "kaggle" / "build_and_push_100m.py"
SPEC = importlib.util.spec_from_file_location("small_llm_build_push_100m", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
suite = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = suite
SPEC.loader.exec_module(suite)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


class BuildAndPush100MTests(unittest.TestCase):
    def test_handle_can_be_derived_from_username(self) -> None:
        self.assertEqual(
            suite.resolve_handle(None, {"KAGGLE_USERNAME": "owner"}),
            "owner/small-llm-20m-100m-dataset-001",
        )

    def test_handle_must_be_resolvable(self) -> None:
        with self.assertRaises(suite.SuiteFailure):
            suite.resolve_handle(None, {})

    def test_resume_only_adds_resume_flag(self) -> None:
        config = suite.Config(
            weights=Path("/weights.json"),
            dataset=Path("/dataset"),
            ops=Path("/ops"),
            handle="owner/data",
            force_upload=False,
            timeout=900,
        )
        fresh = suite.producer_command(config, False)
        resumed = suite.producer_command(config, True)
        self.assertEqual(resumed[:-1], fresh)
        self.assertEqual(resumed[-1], "--resume")

    def test_training_shape_and_tree_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "train").mkdir()
            (root / "validation").mkdir()
            train = root / "train" / "train-000000.bin"
            validation = root / "validation" / "validation-000000.bin"
            train.write_bytes(b"train")
            validation.write_bytes(b"valid")
            manifest = {
                "schema_version": 2,
                "sequence_format": "context_plus_one",
                "context_length": 2048,
                "stored_tokens_per_sequence": 2049,
                "sequences_per_block": 16,
                "target_shard_bytes": 8 * 1024 * 1024,
                "accepted_source_tokens": 100_000_001,
                "production": suite.production_identity(),
                "shards": [],
            }
            drive = {
                "version": 1,
                "run_id": suite.RUN_ID,
                "shards": [{"remote_durable": True}],
            }
            write_json(root / "manifest.json", manifest)
            write_json(root / "drive_manifest.json", drive)
            write_json(
                root / "qualification_plan.json",
                {
                    "qualification_profile": suite.PROFILE,
                    "identity": {
                        "manifest_sha256": suite.sha256(root / "manifest.json"),
                        "drive_manifest_sha256": suite.sha256(root / "drive_manifest.json"),
                    },
                },
            )
            shape = suite.validate_shape(root)
            self.assertEqual(shape["run_id"], suite.RUN_ID)
            identity = suite.tree_identity(root)
            self.assertEqual(identity["file_count"], 5)
            self.assertRegex(identity["tree_sha256"], r"^[0-9a-f]{64}$")

    def test_publish_state_is_bound_to_handle_and_tree(self) -> None:
        config = suite.Config(
            weights=Path("/weights.json"),
            dataset=Path("/dataset"),
            ops=Path("/ops"),
            handle="owner/data",
            force_upload=False,
            timeout=900,
        )
        identity = {"tree_sha256": "a" * 64}
        state = {
            "handle": "owner/data",
            "tree_sha256": "a" * 64,
            "profile": suite.PROFILE,
            "run_id": suite.RUN_ID,
        }
        self.assertTrue(suite.state_matches(state, config, identity))
        state["handle"] = "other/data"
        self.assertFalse(suite.state_matches(state, config, identity))


if __name__ == "__main__":
    unittest.main()
