"""Offline tests for the fail-closed dataset acceptance evidence verifier."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dataset.acceptance import (
    check_temporary_artifacts,
    compare_idempotence,
    evaluate_acceptance,
    validate_command_evidence,
    validate_interruption_resume,
    validate_weights_file,
    write_acceptance_reports,
)


class AcceptanceHarnessTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _weights() -> dict[str, int]:
        return {
            str(cluster): cluster * 100
            for cluster in (*range(1, 11), *range(12, 21))
        }

    def test_validate_weights_file_requires_exact_cluster_set(self) -> None:
        path = self.root / "weights.json"
        path.write_text(json.dumps(self._weights()), encoding="utf-8")
        weights, digest = validate_weights_file(path, expected_sha256=None)
        self.assertEqual(set(weights), set(self._weights()))
        self.assertEqual(len(digest), 64)

        extra = self._weights()
        extra["11"] = 1
        path.write_text(json.dumps(extra), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "exactly clusters"):
            validate_weights_file(path, expected_sha256=None)

    def test_validate_weights_file_rejects_non_positive_bool_and_wrong_hash(self) -> None:
        path = self.root / "weights.json"
        values = self._weights()
        for invalid in (0, -1, True):
            with self.subTest(invalid=invalid):
                changed = dict(values)
                changed["1"] = invalid
                path.write_text(json.dumps(changed), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    validate_weights_file(path, expected_sha256=None)
        path.write_text(json.dumps(values), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            validate_weights_file(path, expected_sha256="0" * 64)

    def test_temporary_artifact_detection_is_recursive(self) -> None:
        nested = self.root / "nested"
        nested.mkdir()
        (nested / "clean.bin").write_bytes(b"clean")
        (nested / "stale.tmp").write_bytes(b"tmp")
        (nested / "upload.part").write_bytes(b"part")
        (nested / "report.json.tmp.deadbeef").write_bytes(b"tmp")
        artifacts = check_temporary_artifacts(self.root)
        self.assertEqual(len(artifacts), 3)
        self.assertTrue(all("clean.bin" not in path for path in artifacts))

    def test_command_evidence_fails_closed(self) -> None:
        result = validate_command_evidence(
            exit_code_path=self.root / "missing.exit-code",
            log_path=self.root / "missing.log",
            required_markers=("RESULT=PASS",),
        )
        self.assertFalse(result["passed"])
        self.assertGreaterEqual(len(result["problems"]), 2)

        exit_code = self.root / "command.exit-code"
        log = self.root / "command.log"
        exit_code.write_text("0\n", encoding="utf-8")
        log.write_text("OK\nRESULT=PASS\n", encoding="utf-8")
        result = validate_command_evidence(
            exit_code_path=exit_code,
            log_path=log,
            required_markers=("OK", "RESULT=PASS"),
        )
        self.assertTrue(result["passed"])

    @staticmethod
    def _completed_snapshot() -> dict[str, object]:
        return {
            "version": 1,
            "output_dir": "/data/pilot",
            "complete": True,
            "accepted_source_tokens": 10_000_000,
            "production": {"configuration_hash": "cfg", "policy": {"run_id": "run"}},
            "source_reader": {
                "documents_consumed": 20,
                "last_incorporated_record_start": {"1": 200},
            },
            "local_shards": [
                {
                    "filename": "train/train-000000.bin",
                    "split": "train",
                    "byte_size": 20,
                    "checksum": "a" * 64,
                    "first_block_id": 0,
                    "last_block_id": 0,
                }
            ],
            "drive_manifest_identity": {
                "run_id": "run",
                "configuration_hash": "cfg",
                "schema_hash": "schema",
            },
            "drive_shards": [
                {
                    "filename": "train/train-000000.bin",
                    "drive_file_id": "file-1",
                    "byte_size": 20,
                    "local_sha256": "a" * 64,
                    "remote_durable": True,
                    "configuration_hash": "cfg",
                    "schema_hash": "schema",
                }
            ],
        }

    def test_completed_resume_comparison_requires_exact_semantic_identity(self) -> None:
        baseline = self._completed_snapshot()
        self.assertTrue(compare_idempotence(baseline, dict(baseline))["passed"])
        changed = json.loads(json.dumps(baseline))
        changed["drive_shards"][0]["drive_file_id"] = "new-id"
        result = compare_idempotence(baseline, changed)
        self.assertFalse(result["passed"])
        self.assertIn("drive_shards", " ".join(result["problems"]))

    def test_interruption_resume_requires_progress_and_preserves_durable_shards(self) -> None:
        completed = self._completed_snapshot()
        interrupted = json.loads(json.dumps(completed))
        interrupted["complete"] = False
        interrupted["accepted_source_tokens"] = 2_000_000
        interrupted["source_reader"]["documents_consumed"] = 5
        interrupted["source_reader"]["last_incorporated_record_start"] = {"1": 50}
        result = validate_interruption_resume(interrupted, completed)
        self.assertTrue(result["passed"], result["problems"])

        interrupted["complete"] = True
        result = validate_interruption_resume(interrupted, completed)
        self.assertFalse(result["passed"])

    def test_missing_live_evidence_can_never_report_pass(self) -> None:
        weights = self.root / "weights.json"
        weights.write_text(json.dumps(self._weights()), encoding="utf-8")
        result = evaluate_acceptance(
            repo_root=Path.cwd(),
            weights_path=weights,
            calibration_dir=self.root / "missing-calibration",
            pilot_output_dir=self.root / "missing-pilot",
            run_id="pilot",
            interrupted_snapshot_path=self.root / "missing-interrupted.json",
            idempotence_baseline_path=self.root / "missing-idempotence.json",
            ops_dir=self.root / "missing-ops",
            drive_smoke_log=self.root / "missing-drive.log",
            expected_weights_sha256="0" * 64,
        )
        self.assertFalse(result["passed"])
        self.assertFalse(result["gates"]["calibration"]["passed"])
        self.assertFalse(result["gates"]["drive_smoke"]["passed"])
        self.assertFalse(result["gates"]["pilot"]["passed"])
        self.assertFalse(result["gates"]["interruption_resume"]["passed"])
        self.assertFalse(result["gates"]["completed_resume_idempotence"]["passed"])

    def test_reports_are_written_atomically(self) -> None:
        json_path = self.root / "ops" / "report.json"
        md_path = self.root / "ops" / "report.md"
        report = {
            "passed": False,
            "git_commit": "abc",
            "run_id": "pilot",
            "gates": {
                "pilot": {"passed": False, "problems": ["missing evidence"]}
            },
            "failures": ["pilot: missing evidence"],
        }
        write_acceptance_reports(report, json_path=json_path, markdown_path=md_path)
        self.assertEqual(json.loads(json_path.read_text()), report)
        markdown = md_path.read_text(encoding="utf-8")
        self.assertIn("Overall Status:** FAIL", markdown)
        self.assertIn("missing evidence", markdown)
        self.assertFalse(any(self.root.rglob("*.tmp.*")))


if __name__ == "__main__":
    unittest.main()
