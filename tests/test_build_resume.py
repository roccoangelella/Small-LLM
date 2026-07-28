"""End-to-end build, interruption+resume, exact-once, and verifier tests."""

from __future__ import annotations

import csv
import logging
import mmap
import tempfile
import unittest
from pathlib import Path

from dataset import config
from dataset.src import verify as verify_module
from dataset.src.bitio import decode_uint16_le
from dataset.src.build import build
from dataset.src.exceptions import IntentionalCrash
from dataset.src.progress_report import status_report
from dataset.src.storage import read_json

from tests.synthetic import (
    FULL_ACCEPTED_SOURCE_TOKENS,
    SyntheticSource,
    build_default_synthetic_source,
    doc_line,
    make_effective,
)


def _run(effective, source, *, reader_factory_on_disk=None):
    factory = source.reader_factory()
    return build(effective, reader_factory=factory,
                 source_files_provider=lambda: source.source_files)


class BuildResumeTest(unittest.TestCase):
    def setUp(self) -> None:
        logging.disable(logging.CRITICAL)

    def tearDown(self) -> None:
        logging.disable(logging.NOTSET)

    # --- core end-to-end ----------------------------------------------------

    def test_build_to_completion_and_verify(self) -> None:
        source = build_default_synthetic_source()
        with tempfile.TemporaryDirectory(prefix="climbmix-test-a-") as tmp:
            out = Path(tmp) / "out"
            effective = make_effective(
                out, target_accepted_source_tokens=FULL_ACCEPTED_SOURCE_TOKENS
            )
            summary = _run(effective, source)
            self.assertTrue(summary["complete"])
            self._assert_common_counts(summary, out)

            report = verify_module.verify(out, full_scan=True)
            self.assertTrue(report.passed, report.problems)
            self._assert_memory_mappable(out)

    def test_cluster_11_is_excluded_using_only_cluster_id(self) -> None:
        source = build_default_synthetic_source()
        with tempfile.TemporaryDirectory(prefix="climbmix-test-cl-") as tmp:
            out = Path(tmp) / "out"
            effective = make_effective(
                out, target_accepted_source_tokens=FULL_ACCEPTED_SOURCE_TOKENS
            )
            summary = _run(effective, source)
            per_cluster = summary["per_cluster"]
            self.assertEqual(per_cluster["11"]["documents"], 0)
            self.assertEqual(per_cluster["11"]["source_tokens"], 0)
            self.assertEqual(summary["cluster_exclusions"].get("11"), 2)

    def test_no_document_level_semantic_filter_runs(self) -> None:
        # Structural validation only: the out-of-range token record is rejected,
        # but nothing is decoded/classified.  Accepted records are written verbatim.
        source = build_default_synthetic_source()
        with tempfile.TemporaryDirectory(prefix="climbmix-test-nf-") as tmp:
            out = Path(tmp) / "out"
            effective = make_effective(
                out, target_accepted_source_tokens=FULL_ACCEPTED_SOURCE_TOKENS
            )
            summary = _run(effective, source)
            self.assertEqual(summary["structural_rejections"].get("token_out_of_range"), 1)
            # Only one structural reason exists; no quality/code/topic filters.
            self.assertEqual(set(summary["structural_rejections"]), {"token_out_of_range"})

    def test_interrupted_and_resumed_output_is_byte_identical_to_uninterrupted(self) -> None:
        source = build_default_synthetic_source()
        with tempfile.TemporaryDirectory(prefix="climbmix-test-a-") as ta, \
             tempfile.TemporaryDirectory(prefix="climbmix-test-c-") as tc:
            out_a = Path(ta) / "out"
            out_c = Path(tc) / "out"

            # Uninterrupted reference run.
            summary_a = _run(
                make_effective(
                    out_a,
                    target_accepted_source_tokens=FULL_ACCEPTED_SOURCE_TOKENS,
                ),
                source,
            )
            self.assertTrue(summary_a["complete"])

            # Interrupted run: crash mid-stream, then resume to completion.
            crash_effective = make_effective(
                out_c,
                target_accepted_source_tokens=FULL_ACCEPTED_SOURCE_TOKENS,
                crash_after_written_bytes=550,
            )
            with self.assertRaises(IntentionalCrash):
                _run(crash_effective, source)

            progress = read_json(out_c / config.PROGRESS_FILENAME)
            train = (out_c / config.TRAIN_FILENAME).stat().st_size
            val = (out_c / config.VALIDATION_FILENAME).stat().st_size
            confirmed = (progress["confirmed_train_byte_size"]
                         + progress["confirmed_validation_byte_size"])
            # Truncation target: checking that on-disk bytes are at least the
            # confirmed checkpoint (crash may align exactly with a checkpoint).
            self.assertGreaterEqual(train + val, confirmed)
            self.assertGreater(confirmed, 0)

            resume_effective = make_effective(
                out_c,
                target_accepted_source_tokens=FULL_ACCEPTED_SOURCE_TOKENS,
                resume=True,
            )
            summary_c = _run(resume_effective, source)
            self.assertTrue(summary_c["complete"])

            train_a = (out_a / config.TRAIN_FILENAME).read_bytes()
            train_c = (out_c / config.TRAIN_FILENAME).read_bytes()
            val_a = (out_a / config.VALIDATION_FILENAME).read_bytes()
            val_c = (out_c / config.VALIDATION_FILENAME).read_bytes()
            self.assertEqual(train_a, train_c, "train.bin differs between uninterrupted and resumed runs")
            self.assertEqual(val_a, val_c, "validation.bin differs between uninterrupted and resumed runs")

            report_c = verify_module.verify(out_c, full_scan=True)
            self.assertTrue(report_c.passed, report_c.problems)

    def test_uncheckpointed_tail_bytes_are_truncated_on_resume(self) -> None:
        """A crash with no committed checkpoint must truncate everything written."""
        source = build_default_synthetic_source()
        with tempfile.TemporaryDirectory(prefix="climbmix-test-tr-") as tt, \
             tempfile.TemporaryDirectory(prefix="climbmix-test-ref-") as tr:
            out = Path(tt) / "out"
            out_ref = Path(tr) / "out"

            # Fresh reference run with default thresholds.
            summary_ref = _run(
                make_effective(
                    out_ref,
                    target_accepted_source_tokens=FULL_ACCEPTED_SOURCE_TOKENS,
                ),
                source,
            )
            self.assertTrue(summary_ref["complete"])

            # Huge checkpoint threshold: no mid-run checkpoint, so the only durable
            # state is the initial empty checkpoint (confirmed sizes = 0). Every byte
            # written before the crash is therefore an uncommitted tail.
            crash_eff = make_effective(
                out,
                target_accepted_source_tokens=FULL_ACCEPTED_SOURCE_TOKENS,
                checkpoint_bytes_threshold=10_000_000_000,
                crash_after_written_bytes=550,
            )
            with self.assertRaises(IntentionalCrash):
                _run(crash_eff, source)

            progress = read_json(out / config.PROGRESS_FILENAME)
            confirmed = (progress["confirmed_train_byte_size"]
                         + progress["confirmed_validation_byte_size"])
            train = (out / config.TRAIN_FILENAME).stat().st_size
            val = (out / config.VALIDATION_FILENAME).stat().st_size
            # Nothing is committed, but bytes exist on disk: they must be truncated.
            self.assertEqual(confirmed, 0)
            self.assertGreater(train + val, 0)

            summary = _run(
                make_effective(
                    out,
                    target_accepted_source_tokens=FULL_ACCEPTED_SOURCE_TOKENS,
                    resume=True,
                ),
                source,
            )
            self.assertTrue(summary["complete"])

            # Truncating to 0 and reprocessing must reproduce the reference bytes.
            self.assertEqual((out_ref / config.TRAIN_FILENAME).read_bytes(),
                             (out / config.TRAIN_FILENAME).read_bytes())
            self.assertEqual((out_ref / config.VALIDATION_FILENAME).read_bytes(),
                             (out / config.VALIDATION_FILENAME).read_bytes())

    def test_resume_refuses_changed_settings(self) -> None:
        source = build_default_synthetic_source()
        with tempfile.TemporaryDirectory(prefix="climbmix-test-rs-") as tmp:
            out = Path(tmp) / "out"
            # Leave a partial (crashed) corpus on disk.
            crash = make_effective(
                out,
                target_accepted_source_tokens=FULL_ACCEPTED_SOURCE_TOKENS,
                crash_after_written_bytes=550,
            )
            with self.assertRaises(IntentionalCrash):
                _run(crash, source)
            # Resuming with a different target must be refused: the output identity
            # would change, so the run signature no longer matches the checkpoint.
            changed = make_effective(
                out,
                target_accepted_source_tokens=FULL_ACCEPTED_SOURCE_TOKENS + 1,
                resume=True,
            )
            with self.assertRaises(ValueError):
                _run(changed, source)
            # Resuming with identical settings is allowed and completes.
            same = make_effective(
                out,
                target_accepted_source_tokens=FULL_ACCEPTED_SOURCE_TOKENS,
                resume=True,
            )
            self.assertTrue(_run(same, source)["complete"])

    def test_progress_csv_records_resume_and_status_reads_it(self) -> None:
        source = build_default_synthetic_source()
        with tempfile.TemporaryDirectory(prefix="climbmix-progress-") as tmp:
            out = Path(tmp) / "out"
            with self.assertRaises(IntentionalCrash):
                _run(
                    make_effective(
                        out,
                        target_accepted_source_tokens=FULL_ACCEPTED_SOURCE_TOKENS,
                        crash_after_written_bytes=550,
                    ),
                    source,
                )
            self.assertTrue(
                _run(
                    make_effective(
                        out,
                        target_accepted_source_tokens=FULL_ACCEPTED_SOURCE_TOKENS,
                        resume=True,
                    ),
                    source,
                )["complete"]
            )

            with (out / config.PROGRESS_CSV_FILENAME).open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertIn("start", [row["event"] for row in rows])
            self.assertIn("resume", [row["event"] for row in rows])
            self.assertEqual(rows[-1]["event"], "complete")

            report = status_report(out)
            self.assertTrue(report["complete"])
            self.assertIsNotNone(report["latest_snapshot"])
            self.assertEqual(report["latest_snapshot"]["event"], "complete")

    def test_refuse_overwrite_uncheckpointed_corpus_without_reset(self) -> None:
        source = build_default_synthetic_source()
        with tempfile.TemporaryDirectory(prefix="climbmix-test-ow-") as tmp:
            out = Path(tmp) / "out"
            _run(
                make_effective(
                    out,
                    target_accepted_source_tokens=FULL_ACCEPTED_SOURCE_TOKENS,
                ),
                source,
            )
            with self.assertRaises(RuntimeError):
                _run(
                    make_effective(
                        out,
                        target_accepted_source_tokens=FULL_ACCEPTED_SOURCE_TOKENS,
                    ),
                    source,
                )
            # With reset=True, a fresh build succeeds.
            summary = _run(
                make_effective(
                    out,
                    target_accepted_source_tokens=FULL_ACCEPTED_SOURCE_TOKENS,
                    reset=True,
                ),
                source,
            )
            self.assertTrue(summary["complete"])

    def test_strict_mode_aborts_on_structural_rejection(self) -> None:
        source = build_default_synthetic_source()
        with tempfile.TemporaryDirectory(prefix="climbmix-test-st-") as tmp:
            out = Path(tmp) / "out"
            effective = make_effective(
                out,
                target_accepted_source_tokens=FULL_ACCEPTED_SOURCE_TOKENS,
                strict=True,
            )
            with self.assertRaises(RuntimeError):
                _run(effective, source)

    def test_bounded_interruption_resume_honors_absolute_work_item_cap(self) -> None:
        source = SyntheticSource()
        source.add_file(
            "part_many.tokenized.jsonl",
            [doc_line(1 + (index % 10), [index % 100] * 4) for index in range(200)],
        )
        with tempfile.TemporaryDirectory(prefix="climbmix-bound-ref-") as ref_tmp, \
             tempfile.TemporaryDirectory(prefix="climbmix-bound-resume-") as resume_tmp:
            out_ref = Path(ref_tmp) / "out"
            out_resume = Path(resume_tmp) / "out"
            settings = {
                "target_accepted_source_tokens": 100_000,
                "region_bytes": 256,
                "max_work_items": 8,
                "checkpoint_bytes_threshold": 40,
            }

            reference = _run(make_effective(out_ref, **settings), source)
            self.assertFalse(reference["complete"])
            self.assertEqual(reference["next_work_item_index"], 8)

            with self.assertRaises(IntentionalCrash):
                _run(
                    make_effective(
                        out_resume,
                        crash_after_written_bytes=90,
                        **settings,
                    ),
                    source,
                )
            resumed = _run(
                make_effective(out_resume, resume=True, **settings),
                source,
            )
            self.assertFalse(resumed["complete"])
            self.assertEqual(resumed["next_work_item_index"], 8)
            self.assertEqual(
                (out_ref / config.TRAIN_FILENAME).read_bytes(),
                (out_resume / config.TRAIN_FILENAME).read_bytes(),
            )
            self.assertEqual(
                (out_ref / config.VALIDATION_FILENAME).read_bytes(),
                (out_resume / config.VALIDATION_FILENAME).read_bytes(),
            )
            report = verify_module.verify(out_resume, full_scan=True)
            self.assertTrue(report.passed, report.problems)

    def test_resume_refuses_a_missing_confirmed_binary(self) -> None:
        source = build_default_synthetic_source()
        with tempfile.TemporaryDirectory(prefix="climbmix-missing-") as tmp:
            out = Path(tmp) / "out"
            with self.assertRaises(IntentionalCrash):
                _run(
                    make_effective(
                        out,
                        target_accepted_source_tokens=FULL_ACCEPTED_SOURCE_TOKENS,
                        crash_after_written_bytes=550,
                    ),
                    source,
                )
            progress = read_json(out / config.PROGRESS_FILENAME)
            self.assertGreater(progress["confirmed_train_byte_size"], 0)
            (out / config.TRAIN_FILENAME).unlink()
            with self.assertRaises(RuntimeError):
                _run(
                    make_effective(
                        out,
                        target_accepted_source_tokens=FULL_ACCEPTED_SOURCE_TOKENS,
                        resume=True,
                    ),
                    source,
                )

    def test_verifier_detects_binary_and_work_plan_tampering(self) -> None:
        source = build_default_synthetic_source()
        with tempfile.TemporaryDirectory(prefix="climbmix-tamper-") as tmp:
            out = Path(tmp) / "out"
            _run(
                make_effective(
                    out,
                    target_accepted_source_tokens=FULL_ACCEPTED_SOURCE_TOKENS,
                ),
                source,
            )
            with (out / config.TRAIN_FILENAME).open("r+b") as handle:
                handle.write(b"\xff\xff")
            with (out / config.WORK_PLAN_FILENAME).open("ab") as handle:
                handle.write(b"\n")
            report = verify_module.verify(out, full_scan=True)
            self.assertFalse(report.passed)
            self.assertTrue(
                any("train_sha256 mismatch" in problem for problem in report.problems)
            )
            self.assertTrue(
                any("work_plan_sha256 mismatch" in problem for problem in report.problems)
            )
            self.assertTrue(
                any("outside" in problem for problem in report.problems)
            )

    def test_default_split_produces_readable_train_and_validation_files(self) -> None:
        source = SyntheticSource()
        source.add_file(
            "part_many.tokenized.jsonl",
            [doc_line(1, [index % 100]) for index in range(5_000)],
        )
        with tempfile.TemporaryDirectory(prefix="climbmix-splits-") as tmp:
            out = Path(tmp) / "out"
            summary = _run(
                make_effective(
                    out,
                    target_accepted_source_tokens=5_000,
                    region_bytes=1_000_000,
                ),
                source,
            )
            self.assertTrue(summary["complete"])
            self.assertGreater((out / config.TRAIN_FILENAME).stat().st_size, 0)
            self.assertGreater((out / config.VALIDATION_FILENAME).stat().st_size, 0)
            self._assert_memory_mappable(out)
            report = verify_module.verify(out, full_scan=True)
            self.assertTrue(report.passed, report.problems)

    # --- helpers -------------------------------------------------------------

    def _assert_common_counts(self, summary: dict, out: Path) -> None:
        accepted_docs = summary["accepted_document_count"]
        self.assertGreater(accepted_docs, 0)
        self.assertEqual(
            summary["accepted_source_tokens"], FULL_ACCEPTED_SOURCE_TOKENS
        )
        self.assertEqual(summary["inspected_document_count"], 17)
        # Exactly one accepted document already ends in EOD (cluster 10), so the
        # inserted EOD count is one fewer than the accepted document count.
        self.assertEqual(summary["inserted_eod_count"], accepted_docs - 1)
        # Total written tokens == accepted source tokens + inserted EODs.
        self.assertEqual(
            summary["train_written_tokens"] + summary["validation_written_tokens"],
            summary["accepted_source_tokens"] + summary["inserted_eod_count"],
        )
        # One EOD separator per accepted document across both streams.
        train = (out / config.TRAIN_FILENAME).read_bytes()
        val = (out / config.VALIDATION_FILENAME).read_bytes()
        all_tokens = decode_uint16_le(train + val)
        self.assertEqual(all_tokens.count(config.EOD_TOKEN_ID), accepted_docs)

    def _assert_memory_mappable(self, out: Path) -> None:
        for name in (config.TRAIN_FILENAME, config.VALIDATION_FILENAME):
            path = out / name
            size = path.stat().st_size
            if size == 0:
                continue
            with path.open("rb") as handle:
                with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapping:
                    for offset in range(0, size, 2):
                        value = int.from_bytes(mapping[offset:offset + 2], "little")
                        self.assertGreaterEqual(value, config.TOKEN_MIN)
                        self.assertLessEqual(value, config.TOKEN_MAX)


if __name__ == "__main__":
    unittest.main()
