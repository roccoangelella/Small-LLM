"""Tests for the fixed finite dataset used by the first 20M qualification."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from dataset.qualification_20m import (
    CHECKPOINT_SOURCE_TOKENS,
    CONTEXT_LENGTH,
    MAXIMUM_SOURCE_TOKENS,
    MINIMUM_SOURCE_TOKENS,
    SEQUENCES_PER_BLOCK,
    TARGET_SHARD_BYTES,
    TARGET_SOURCE_TOKENS,
    main,
    qualification_arguments,
)


class Qualification20MDatasetTests(unittest.TestCase):
    def test_profile_appends_the_frozen_geometry(self) -> None:
        args = qualification_arguments(
            [
                "--weights-file",
                "/tmp/weights.json",
                "--output-dir",
                "/tmp/data",
                "--run-id",
                "qualification-001",
            ]
        )
        expected = {
            "--target-tokens": str(TARGET_SOURCE_TOKENS),
            "--minimum-tokens": str(MINIMUM_SOURCE_TOKENS),
            "--maximum-tokens": str(MAXIMUM_SOURCE_TOKENS),
            "--checkpoint-source-tokens": str(CHECKPOINT_SOURCE_TOKENS),
            "--context-length": str(CONTEXT_LENGTH),
            "--sequences-per-block": str(SEQUENCES_PER_BLOCK),
            "--target-shard-bytes": str(TARGET_SHARD_BYTES),
        }
        for flag, value in expected.items():
            index = args.index(flag)
            self.assertEqual(args[index + 1], value)
        self.assertEqual(TARGET_SHARD_BYTES, 8_388_608)

    def test_profile_rejects_geometry_override(self) -> None:
        with self.assertRaisesRegex(SystemExit, "fixes these arguments"):
            qualification_arguments(["--sequences-per-block=32"])

    def test_profile_rejects_local_only_escape_hatch(self) -> None:
        with self.assertRaisesRegex(SystemExit, "allow-local-only"):
            qualification_arguments(["--allow-local-only"])

    def test_main_delegates_to_production_cli(self) -> None:
        with patch("dataset.qualification_20m.production_main", return_value=7) as production:
            result = main(
                [
                    "--weights-file",
                    "/tmp/weights.json",
                    "--output-dir",
                    "/tmp/data",
                    "--run-id",
                    "qualification-001",
                ]
            )
        self.assertEqual(result, 7)
        delegated = production.call_args.args[0]
        self.assertIn("--target-shard-bytes", delegated)
        self.assertIn("8388608", delegated)


if __name__ == "__main__":
    unittest.main()
