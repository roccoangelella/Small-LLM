"""Offline unit tests for dataset.acceptance harness."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from dataset.acceptance import (
    check_temporary_artifacts,
    run_environment_preflight,
    validate_weights_file,
    write_acceptance_reports,
)


class AcceptanceHarnessTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp_dir.name)

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_validate_weights_file_valid(self) -> None:
        valid_weights = {str(i): 100 for i in range(1, 11)}
        valid_weights.update({str(i): 100 for i in range(12, 21)})
        weights_path = self.root / "valid_weights.json"
        weights_path.write_text(json.dumps(valid_weights), encoding="utf-8")

        data, sha256_hex = validate_weights_file(weights_path)
        self.assertEqual(len(data), 19)
        self.assertNotIn("11", data)
        self.assertEqual(len(sha256_hex), 64)

    def test_validate_weights_file_rejects_cluster_11(self) -> None:
        invalid_weights = {str(i): 100 for i in range(1, 21)}
        weights_path = self.root / "invalid_weights.json"
        weights_path.write_text(json.dumps(invalid_weights), encoding="utf-8")

        with self.assertRaises(ValueError) as ctx:
            validate_weights_file(weights_path)
        self.assertIn("Excluded cluster 11", str(ctx.exception))

    def test_validate_weights_file_rejects_non_positive_weights(self) -> None:
        invalid_weights = {str(i): 100 for i in range(1, 11)}
        invalid_weights.update({str(i): 100 for i in range(12, 21)})
        invalid_weights["1"] = 0
        weights_path = self.root / "zero_weight.json"
        weights_path.write_text(json.dumps(invalid_weights), encoding="utf-8")

        with self.assertRaises(ValueError) as ctx:
            validate_weights_file(weights_path)
        self.assertIn("positive integer", str(ctx.exception))

    def test_environment_preflight(self) -> None:
        info = run_environment_preflight()
        self.assertIn("git_commit", info)
        self.assertIn("python_version", info)
        self.assertIn("platform", info)
        self.assertIn("secrets_untracked", info)

    def test_check_temporary_artifacts(self) -> None:
        (self.root / "clean.bin").write_bytes(b"clean")
        (self.root / "stale.tmp").write_bytes(b"tmp")
        (self.root / "progress.production.safe.json").write_bytes(b"{}")

        artifacts = check_temporary_artifacts(self.root)
        self.assertEqual(len(artifacts), 2)

    def test_write_acceptance_reports_atomic(self) -> None:
        json_path = self.root / "ops" / "report.json"
        md_path = self.root / "ops" / "report.md"
        report_data = {
            "passed": True,
            "git_commit": "abc1234",
            "python_version": "3.13.0",
            "platform": "Linux",
            "secrets_untracked": True,
            "preflight_passed": True,
            "offline_tests_passed": True,
            "calibration_complete": True,
            "scanned_source_bytes": 1000,
            "calibration_report_sha256": "abc",
            "approved_weights_sha256": "def",
            "drive_smoke_passed": True,
            "pilot_passed": True,
            "resume_passed": True,
            "verification_passed": True,
            "idempotence_passed": True,
            "artifacts_clean": True,
            "final_accepted_tokens": 10000000,
            "local_shard_count": 5,
            "local_shard_bytes": 50000,
            "failures": [],
        }

        write_acceptance_reports(report_data, json_path=json_path, md_path=md_path)
        self.assertTrue(json_path.exists())
        self.assertTrue(md_path.exists())

        saved_json = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual(saved_json["git_commit"], "abc1234")
        self.assertIn("# Dataset Operational Qualification Acceptance Report", md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
