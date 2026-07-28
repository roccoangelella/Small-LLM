"""Token-corpus build orchestration.

This is the single production entry point.  It resolves the deterministic work
plan (or loads it on resume), streams records by HTTP byte ranges, applies
structural validation and the numeric cluster policy, assigns documents to
train/validation deterministically, appends little-endian uint16 token IDs to
``train.bin`` / ``validation.bin``, and takes crash-safe checkpoints.

Production never decodes accepted documents to text, never runs a code/quality
filter, and never contacts an LLM.  The only semantic signal is the source
``cluster_id`` (cluster 11 excluded).
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dataset import config

from .bytesource import (
    RangeReader,
    SourceFile,
    list_source_files,
    make_http_reader,
)
from .checkpoint import (
    Progress,
    remove_uncheckpointed_corpus,
    stamp_progress,
    truncate_to_confirmed,
    validate_resume,
)
from .exceptions import IntentionalCrash
from .manifest import build_manifest
from .progress_report import ProgressReporter
from .records import ParsedRecord, iter_owned_records, record_identity_str, validate_record
from .split import is_validation
from .storage import write_json_atomic
from .workplan import WorkItem, WorkPlan, build_work_plan, load_work_plan, save_work_plan
from .writer import BinaryCorpusWriter


LOGGER = logging.getLogger(__name__)

ReaderFactory = Callable[[SourceFile], RangeReader]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def build(
    effective: "config.EffectiveConfig",
    *,
    reader_factory: ReaderFactory | None = None,
    source_files_provider: "Callable[[], list[SourceFile]] | None" = None,
) -> dict[str, Any]:
    """Build (or resume) the corpus.  Returns a summary dict for the CLI.

    ``reader_factory`` and ``source_files_provider`` are test hooks: production
    leaves them ``None`` so the real HTTP reader and Hugging Face tree-API
    resolver are used.  Tests inject a local range reader and synthetic source
    files so no network access is required.
    """

    output_dir = effective.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    progress_path = output_dir / config.PROGRESS_FILENAME
    plan_path = output_dir / config.WORK_PLAN_FILENAME
    train_path = output_dir / config.TRAIN_FILENAME
    validation_path = output_dir / config.VALIDATION_FILENAME
    manifest_path = output_dir / config.MANIFEST_FILENAME
    progress_csv_path = output_dir / config.PROGRESS_CSV_FILENAME

    is_production = reader_factory is None
    factory = reader_factory or _make_default_reader_factory()

    if effective.resume:
        plan, progress = _resume(effective, plan_path, progress_path)
        _preflight_local(
            effective,
            output_dir,
            existing_confirmed_bytes=(
                progress.confirmed_train_byte_size
                + progress.confirmed_validation_byte_size
            ),
            check_capacity=not progress.complete,
        )
    else:
        # Local preflight (writable dir, disk space, large-file support) runs
        # before the fresh build contacts the remote source.
        _preflight_local(effective, output_dir)
        plan, progress = _fresh(
            effective, output_dir, plan_path, progress_path, source_files_provider
        )

    if is_production:
        _preflight_reachability(plan, verify_listing=effective.resume)

    truncate_to_confirmed(progress, train_path, validation_path)

    # A completed checkpoint is immutable.  Do not even reopen the binary files
    # in append mode; require the final manifest to be present and return.
    if effective.resume and progress.complete:
        if not manifest_path.exists():
            raise RuntimeError(
                "Checkpoint is marked complete but manifest.json is missing. "
                "The corpus is inconsistent; restore the manifest or verify and rebuild."
            )
        LOGGER.info("Checkpoint is already complete; nothing to do.")
        return _summary(True, output_dir, plan, progress, manifest_path)

    writer = BinaryCorpusWriter(
        train_path, validation_path,
        buffer_bytes=effective.writer_buffer_bytes,
        resume_sizes=(progress.confirmed_train_byte_size, progress.confirmed_validation_byte_size),
    )
    reporter = ProgressReporter.create(
        progress_csv_path, plan, progress, effective.target_accepted_source_tokens
    )
    reporter.snapshot("resume" if effective.resume else "start", progress)

    reached_target = False
    try:
        reached_target = _process_plan(
            effective, plan, progress, writer, factory, progress_path, reporter
        )
        _finalize(
            effective=effective,
            reached_target=reached_target,
            plan=plan,
            progress=progress,
            writer=writer,
            output_dir=output_dir,
            progress_path=progress_path,
            manifest_path=manifest_path,
            reporter=reporter,
        )
    finally:
        writer.close()

    summary = _summary(reached_target, output_dir, plan, progress, manifest_path)
    LOGGER.info("Build %s: %s", "complete" if reached_target else "stopped",
                ", ".join(f"{k}={v}" for k, v in summary.items()
                          if k in ("accepted_source_tokens", "confirmed_train_bytes",
                                   "confirmed_validation_bytes", "train_written_tokens",
                                   "validation_written_tokens")))
    return summary


# ---------------------------------------------------------------------------
# Durable finalization
# ---------------------------------------------------------------------------


def _finalize(
    *,
    effective: "config.EffectiveConfig",
    reached_target: bool,
    plan: WorkPlan,
    progress: Progress,
    writer: BinaryCorpusWriter,
    output_dir: Path,
    progress_path: Path,
    manifest_path: Path,
    reporter: ProgressReporter,
) -> None:
    """Flush, validate, hash, and atomically publish final state.

    The durable progress checkpoint remains ``complete=false`` while potentially
    long SHA-256 calculations run.  Only after the manifest has been written
    successfully is the checkpoint marked complete.  A crash anywhere before
    that point can be resumed safely from the confirmed binary sizes.
    """

    confirmed = writer.final_flush()
    progress.confirmed_train_byte_size, progress.confirmed_validation_byte_size = confirmed
    progress.last_checkpoint_written_bytes = confirmed[0] + confirmed[1]
    progress.complete = False
    progress.save(progress_path)
    reporter.snapshot("final_checkpoint", progress, writer=writer)

    _validate_final_state(effective, reached_target, progress, output_dir)
    manifest = build_manifest(
        progress=progress,
        plan=plan,
        output_dir=output_dir,
        repo_commit=_software_commit(),
        accepted_cluster_ids=config.ACCEPTED_CLUSTER_IDS,
        excluded_cluster_ids=config.EXCLUDED_CLUSTER_IDS,
        start_time=progress.build_start_time,
        complete=reached_target,
    )
    write_json_atomic(manifest_path, manifest)

    if reached_target:
        progress.complete = True
        # The preceding save already recorded the actual durable checkpoint
        # time.  Publishing the completion bit should not pretend that another
        # data checkpoint occurred.
        write_json_atomic(progress_path, progress.to_dict())
        reporter.snapshot("complete", progress, writer=writer)
    else:
        reporter.snapshot("stopped", progress, writer=writer)


def _validate_final_state(
    effective: "config.EffectiveConfig",
    reached_target: bool,
    progress: Progress,
    output_dir: Path,
) -> None:
    """Check counter and byte-size invariants before publishing a manifest."""

    train_path = output_dir / config.TRAIN_FILENAME
    validation_path = output_dir / config.VALIDATION_FILENAME
    train_size = train_path.stat().st_size
    validation_size = validation_path.stat().st_size
    problems: list[str] = []

    if train_size != progress.confirmed_train_byte_size:
        problems.append("train.bin size does not match the confirmed checkpoint")
    if validation_size != progress.confirmed_validation_byte_size:
        problems.append("validation.bin size does not match the confirmed checkpoint")
    if train_size != progress.train_written_tokens * 2:
        problems.append("train.bin size does not match train written-token count")
    if validation_size != progress.validation_written_tokens * 2:
        problems.append("validation.bin size does not match validation written-token count")

    if (
        progress.train_written_tokens + progress.validation_written_tokens
        != progress.accepted_source_tokens + progress.inserted_eod_count
    ):
        problems.append("total written tokens != accepted source tokens + inserted EODs")
    if (
        progress.train_source_tokens + progress.validation_source_tokens
        != progress.accepted_source_tokens
    ):
        problems.append("train/validation source-token counts do not sum to accepted tokens")
    if (
        progress.train_inserted_eod_count + progress.validation_inserted_eod_count
        != progress.inserted_eod_count
    ):
        problems.append("train/validation EOD counts do not sum to inserted EODs")
    if (
        progress.train_document_count + progress.validation_document_count
        != progress.accepted_document_count
    ):
        problems.append("train/validation document counts do not sum to accepted documents")

    cluster_source_tokens = sum(
        int(counters.get("source_tokens", 0))
        for counters in progress.per_cluster.values()
    )
    cluster_documents = sum(
        int(counters.get("documents", 0))
        for counters in progress.per_cluster.values()
    )
    if cluster_source_tokens != progress.accepted_source_tokens:
        problems.append("per-cluster source tokens do not sum to accepted source tokens")
    if cluster_documents != progress.accepted_document_count:
        problems.append("per-cluster documents do not sum to accepted documents")
    excluded = progress.per_cluster.get("11", {})
    if int(excluded.get("documents", 0)) or int(excluded.get("source_tokens", 0)):
        problems.append("excluded cluster 11 appears in accepted counters")

    should_be_complete = (
        effective.target_accepted_source_tokens
        <= progress.accepted_source_tokens
        <= effective.maximum_accepted_source_tokens
    )
    if reached_target != should_be_complete:
        problems.append(
            "completion state disagrees with the accepted-source-token target/maximum"
        )
    if reached_target and (
        progress.accepted_source_tokens < effective.minimum_accepted_source_tokens
    ):
        problems.append("completed corpus is below the minimum acceptable size")

    if problems:
        raise RuntimeError(
            "Refusing to finalize an inconsistent corpus: " + "; ".join(problems)
        )


# ---------------------------------------------------------------------------
# Fresh build vs resume setup
# ---------------------------------------------------------------------------


def _fresh(
    effective: "config.EffectiveConfig",
    output_dir: Path,
    plan_path: Path,
    progress_path: Path,
    source_files_provider: "Callable[[], list[SourceFile]] | None",
) -> tuple[WorkPlan, Progress]:
    if not effective.reset and _has_existing_corpus(output_dir):
        raise RuntimeError(
            f"{output_dir} already contains generated corpus files. Move them, pass "
            "--reset to delete them, or use --resume to continue."
        )
    if effective.reset and _has_existing_corpus(output_dir):
        remove_uncheckpointed_corpus(output_dir)

    if source_files_provider is not None:
        source_files = source_files_provider()
    else:
        source_files = list_source_files(config.DATASET_REPOSITORY, config.DATASET_REVISION)
    if effective.region_bytes != config.REGION_BYTES:
        LOGGER.warning("Using non-default region_bytes=%d (smoke/test override)",
                       effective.region_bytes)
    plan = build_work_plan(
        source_files,
        region_bytes=effective.region_bytes,
        seed=config.SELECTION_SEED,
        repository=config.DATASET_REPOSITORY,
        revision=config.DATASET_REVISION,
    )
    save_work_plan(plan_path, plan)

    progress = Progress.new(
        effective,
        dataset=config.DATASET_REPOSITORY,
        revision=config.DATASET_REVISION,
        work_plan_hash=plan.hash,
        build_start_time=_iso_now(),
    )
    stamp_progress(progress, effective)
    # Persist immediately so even an early interruption leaves a valid checkpoint
    # and the directory is never mistaken for a clean, empty one.
    progress.save(progress_path)
    return plan, progress


def _resume(
    effective: "config.EffectiveConfig",
    plan_path: Path,
    progress_path: Path,
) -> tuple[WorkPlan, Progress]:
    if not (plan_path.exists() and progress_path.exists()):
        raise RuntimeError(
            f"Cannot resume: {config.WORK_PLAN_FILENAME} / {config.PROGRESS_FILENAME} are missing."
        )
    plan = load_work_plan(plan_path)
    progress = Progress.load(progress_path)
    validate_resume(progress, effective, plan.hash)
    if progress.complete:
        LOGGER.warning("Checkpoint is already complete; resume is a no-op.")
    return plan, progress


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------


@dataclass
class _LoopState:
    cumulative_written_bytes: int
    starting_written_bytes: int
    starting_source_bytes: int
    processed_items: int
    started: float


def _process_plan(
    effective: "config.EffectiveConfig",
    plan: WorkPlan,
    progress: Progress,
    writer: BinaryCorpusWriter,
    factory: ReaderFactory,
    progress_path: Path,
    reporter: ProgressReporter,
) -> bool:
    """Process plan items from the resume cursor.  Return True iff target met."""

    if progress.accepted_source_tokens >= effective.target_accepted_source_tokens:
        return True

    file_by_name = {f.path: f for f in plan.source_files}
    state = _LoopState(
        cumulative_written_bytes=progress.last_checkpoint_written_bytes,
        starting_written_bytes=progress.last_checkpoint_written_bytes,
        starting_source_bytes=progress.source_bytes_processed,
        processed_items=0,
        started=time.monotonic(),
    )

    # The loop reads the cursor from ``progress`` directly: ``_process_item``
    # advances ``work_item_index`` itself when an item completes, so the saved
    # checkpoint always holds a consistent (work_item_index, item_resume_record_start).
    while progress.work_item_index < len(plan.work_items):
        # This is an absolute cap in the saved plan, not "N more items per
        # invocation".  An interrupted+resumed bounded run therefore produces
        # exactly the same bytes as one uninterrupted bounded run.
        if (
            effective.max_work_items is not None
            and progress.work_item_index >= effective.max_work_items
        ):
            LOGGER.info(
                "Stopping at absolute max_work_items=%d before completing the target.",
                effective.max_work_items,
            )
            return False
        item = plan.work_items[progress.work_item_index]
        reader = factory(file_by_name[item.filename])

        outcome = _process_item(
            item=item, reader=reader, effective=effective, progress=progress,
            writer=writer, progress_path=progress_path, state=state, reporter=reporter,
        )
        state.cumulative_written_bytes = outcome.cumulative_written_bytes

        state.processed_items += 1
        _log_progress(effective, plan, progress, item, state.processed_items, state)
        reporter.snapshot("work_item_complete", progress, writer=writer, item=item)

        if outcome.target_reached:
            return True

    if progress.accepted_source_tokens >= effective.target_accepted_source_tokens:
        return True
    if effective.max_work_items is None:
        raise RuntimeError(
            "Source work plan exhausted before the accepted-source-token target was reached. "
            f"Have {progress.accepted_source_tokens:,}; need {effective.target_accepted_source_tokens:,}."
        )
    return False


@dataclass
class _ItemOutcome:
    target_reached: bool
    cumulative_written_bytes: int


def _process_item(
    *,
    item: WorkItem,
    reader: RangeReader,
    effective: "config.EffectiveConfig",
    progress: Progress,
    writer: BinaryCorpusWriter,
    progress_path: Path,
    state: _LoopState,
    reporter: ProgressReporter,
) -> _ItemOutcome:
    target = effective.target_accepted_source_tokens
    maximum = effective.maximum_accepted_source_tokens
    threshold = effective.checkpoint_bytes_threshold
    crash_after = effective.crash_after_written_bytes
    seed = config.SELECTION_SEED
    revision = config.DATASET_REVISION

    cum = state.cumulative_written_bytes
    it = iter_owned_records(item, reader)
    target_reached = False
    record = _next_owned(it, progress.item_resume_record_start)

    while record is not None:
        nxt = _next_owned(it, None)

        progress.inspect_record()
        result = validate_record(record)
        if not result.valid:
            progress.reject_structural(result.rejection_reason)
            identity = record_identity_str(revision, item.filename, record.record_start)
            if effective.strict:
                raise RuntimeError(
                    f"strict mode: aborting on structurally invalid record "
                    f"(reason={result.rejection_reason}) at {identity}"
                )
            LOGGER.warning(
                "Skipping structurally invalid record reason=%s source=%s",
                result.rejection_reason,
                identity,
            )
            _advance_cursor(progress, item, nxt)
            reporter.maybe_snapshot(progress, writer=writer, item=item)
            record = nxt
            continue

        assert result.cluster_id is not None and result.tokens is not None
        cluster_id = result.cluster_id
        if cluster_id not in config.ACCEPTED_CLUSTER_IDS:
            progress.exclude_cluster(cluster_id)
            _advance_cursor(progress, item, nxt)
            reporter.maybe_snapshot(progress, writer=writer, item=item)
            record = nxt
            continue

        tokens = list(result.tokens)
        # Refuse a document that would breach the hard maximum. Normal
        # production reaches the 90B target well before the 100B safeguard.
        if progress.accepted_source_tokens + len(tokens) > maximum:
            raise RuntimeError(
                "The next accepted document would exceed the configured maximum "
                f"accepted source tokens ({maximum:,}) before the target was reached. "
                f"Current={progress.accepted_source_tokens:,}, document={len(tokens):,}."
            )

        validation = is_validation(
            seed=seed, revision=revision, filename=item.filename,
            record_start=record.record_start,
            probability=config.VALIDATION_PROBABILITY,
        )
        already_ends_eod = bool(tokens) and tokens[-1] == config.EOD_TOKEN_ID
        inserted_eods = 0 if already_ends_eod else 1
        written_tokens = len(tokens) + inserted_eods
        cum += writer.append(validation=validation, tokens=tokens)
        progress.accept(
            cluster_id=cluster_id, source_tokens=len(tokens),
            written_tokens=written_tokens, inserted_eods=inserted_eods,
            validation=validation,
        )
        # Advance the cursor to the next unprocessed record BEFORE any checkpoint
        # is taken.  When this was the item's last record the cursor moves to the
        # next item, keeping the saved checkpoint self-consistent.
        _advance_cursor(progress, item, nxt)

        # Periodic durable checkpoint (byte threshold).
        if writer.written_since_checkpoint >= threshold:
            _do_checkpoint(progress, writer, progress_path)
            state.cumulative_written_bytes = cum
            reporter.snapshot("checkpoint", progress, writer=writer, item=item)

        reporter.maybe_snapshot(progress, writer=writer, item=item)

        # Optional bounded crash for smoke/resume tests.  Fires after writing but
        # before any further checkpoint, leaving an uncommitted tail to truncate.
        if crash_after is not None and cum >= crash_after:
            writer.flush_uncommitted()
            writer.close()
            raise IntentionalCrash(
                f"intentional test crash after {cum} written bytes "
                f"(item {item.index}, record_start {record.record_start})"
            )

        if progress.accepted_source_tokens >= target:
            target_reached = True
            break
        record = nxt

    if record is None:
        # Item exhausted: mark it complete so a checkpoint fired here (or a
        # later resume) starts at the NEXT item, never re-reading this one.
        _finish_item(progress, item)

    state.cumulative_written_bytes = cum
    return _ItemOutcome(target_reached=target_reached, cumulative_written_bytes=cum)


def _advance_cursor(
    progress: Progress, item: WorkItem, nxt: ParsedRecord | None
) -> None:
    """Point the resume cursor at the next unprocessed record.

    When ``nxt`` is None the current item is finished, so the cursor moves to the
    following item (``work_item_index`` advanced, ``item_resume_record_start``
    reset).  This keeps a checkpoint that fires exactly on an item boundary
    consistent: it never leaves ``item_resume_record_start=None`` while the
    current item still has committed records, which previously caused a resume
    to reprocess the whole item and inflate ``accepted_source_tokens``.
    """

    if nxt is None:
        _finish_item(progress, item)
    else:
        progress.item_resume_record_start = nxt.record_start


def _finish_item(progress: Progress, item: WorkItem) -> None:
    """Mark one logical byte region complete exactly once."""

    if progress.work_item_index <= item.index:
        progress.complete_work_item(item.range_end - item.range_start)
    progress.work_item_index = item.index + 1
    progress.item_resume_record_start = None


def _next_owned(
    iterator: Iterator[ParsedRecord], skip_before: int | None
) -> ParsedRecord | None:
    """Consume the iterator to the next owned record, skipping committed ones."""

    while True:
        try:
            rec = next(iterator)
        except StopIteration:
            return None
        if skip_before is not None and rec.record_start < skip_before:
            continue
        return rec


def _do_checkpoint(progress: Progress, writer: BinaryCorpusWriter, progress_path: Path) -> None:
    confirmed = writer.checkpoint()
    progress.confirmed_train_byte_size, progress.confirmed_validation_byte_size = confirmed
    progress.last_checkpoint_written_bytes = confirmed[0] + confirmed[1]
    progress.save(progress_path)


def _log_progress(
    effective: "config.EffectiveConfig",
    plan: WorkPlan,
    progress: Progress,
    item: WorkItem,
    processed_items: int,
    state: _LoopState,
) -> None:
    elapsed = time.monotonic() - state.started
    written_mib_per_s = (
        (
            (state.cumulative_written_bytes - state.starting_written_bytes)
            / (1024 * 1024)
        )
        / elapsed
        if elapsed
        else 0.0
    )
    source_mib_per_s = (
        (
            (progress.source_bytes_processed - state.starting_source_bytes)
            / (1024 * 1024)
        )
        / elapsed
        if elapsed
        else 0.0
    )
    LOGGER.info(
        "work_items_completed=%d/%d invocation_items=%d item_idx=%d "
        "source_bytes_processed=%s documents_inspected=%d accepted_docs=%d "
        "accepted_source_tokens=%s train_written=%s "
        "val_written=%s eod=%s clusters=[%s] rejections=%d exclusions=%d "
        "elapsed=%.1fs source=%.1fMiB/s written=%.1fMiB/s",
        progress.work_item_index, len(plan.work_items), processed_items, item.index,
        f"{progress.source_bytes_processed:,}", progress.inspected_document_count,
        progress.accepted_document_count, f"{progress.accepted_source_tokens:,}",
        f"{progress.train_written_tokens:,}", f"{progress.validation_written_tokens:,}",
        progress.inserted_eod_count, _cluster_digest(progress),
        _dict_total(progress.structural_rejections),
        _dict_total(progress.cluster_exclusions),
        elapsed, source_mib_per_s, written_mib_per_s,
    )


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------


def _preflight_local(
    effective: "config.EffectiveConfig",
    output_dir: Path,
    *,
    existing_confirmed_bytes: int = 0,
    check_capacity: bool = True,
) -> None:
    probe = output_dir / ".write-probe"
    try:
        with probe.open("xb") as handle:
            handle.write(b"ok")
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as error:
        raise RuntimeError(
            f"Output directory is not writable: {output_dir} "
            f"({type(error).__name__}: {error})"
        ) from error
    finally:
        if probe.exists():
            probe.unlink()
    if check_capacity and not effective.allow_unsafe_low_disk:
        required = _required_disk_bytes(
            effective, existing_confirmed_bytes=existing_confirmed_bytes
        )
        _check_disk_space(output_dir, required)
        _check_large_file_support(output_dir, _projected_corpus_bytes(effective))


def _preflight_reachability(plan: WorkPlan, *, verify_listing: bool) -> None:
    if verify_listing:
        remote_files = list_source_files(
            config.DATASET_REPOSITORY, config.DATASET_REVISION
        )
        if remote_files != list(plan.source_files):
            raise RuntimeError(
                "Pinned remote source file list/sizes no longer match work_plan.json. "
                "Refusing to resume."
            )
    # Probe several points in the immutable file list.  The tree listing above
    # establishes every expected path and exact size; these tiny range reads
    # additionally verify that the content endpoint honors byte ranges.
    probe_indexes = sorted({0, len(plan.source_files) // 2, len(plan.source_files) - 1})
    for index in probe_indexes:
        source = plan.source_files[index]
        try:
            probe = make_http_reader(
                source, config.DATASET_REPOSITORY, config.DATASET_REVISION
            )
            probe.read_range(0, 1)
        except Exception as error:  # noqa: BLE001
            raise RuntimeError(
                f"Could not reach pinned source {source.path} at "
                f"{config.DATASET_REVISION}: {type(error).__name__}: {error}"
            ) from error


def _projected_corpus_bytes(effective: "config.EffectiveConfig") -> int:
    return int(effective.target_accepted_source_tokens * 2 * (1 + config.DISK_EOD_OVERHEAD_FRACTION))


def _required_disk_bytes(
    effective: "config.EffectiveConfig", *, existing_confirmed_bytes: int = 0
) -> int:
    conservative_total = int(
        _projected_corpus_bytes(effective) * config.DISK_SAFETY_MULTIPLIER
    )
    return max(0, conservative_total - existing_confirmed_bytes)


def _check_disk_space(output_dir: Path, required: int) -> None:
    stat = os.statvfs(output_dir)
    free = stat.f_bavail * stat.f_frsize
    if free < required:
        raise RuntimeError(
            f"Insufficient free disk space on {output_dir}: {free / (1024**3):.1f} GiB free, "
            f"need {required / (1024**3):.1f} GiB. Pass --allow-unsafe-low-disk only if you accept "
            "running out of space mid-build."
        )
    LOGGER.info("Disk preflight: %.1f GiB free, %.1f GiB required",
                free / (1024**3), required / (1024**3))


def _check_large_file_support(output_dir: Path, projected: int) -> None:
    probe = output_dir / ".large-file-probe"
    try:
        with probe.open("wb") as handle:
            handle.truncate(projected)
        actual = probe.stat().st_size
        if actual != projected:
            raise OSError(f"truncate to {projected} yielded size {actual}")
    except OSError as error:
        raise RuntimeError(
            f"Output filesystem does not appear to support files of "
            f"~{projected / (1024**3):.1f} GiB ({type(error).__name__}: {error}). "
            "Use a filesystem such as ext4 or xfs."
        ) from error
    finally:
        if probe.exists():
            probe.unlink()


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _make_default_reader_factory() -> ReaderFactory:
    def factory(source_file: SourceFile) -> RangeReader:
        return make_http_reader(source_file, config.DATASET_REPOSITORY, config.DATASET_REVISION)
    return factory


def _software_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=str(config.DATASET_DIR),
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, FileNotFoundError):
        return None
    return None


def _has_existing_corpus(output_dir: Path) -> bool:
    return any(
        (output_dir / name).exists()
        for name in (
            config.TRAIN_FILENAME, config.VALIDATION_FILENAME,
            config.PROGRESS_FILENAME, config.WORK_PLAN_FILENAME, config.MANIFEST_FILENAME,
        )
    )


def _summary(reached: bool, output_dir: Path, plan: WorkPlan, progress: Progress,
             manifest_path: Path) -> dict[str, Any]:
    return {
        "complete": reached,
        "output_dir": str(output_dir),
        "work_plan_hash": plan.hash,
        "work_items_total": len(plan.work_items),
        "next_work_item_index": progress.work_item_index,
        "accepted_source_tokens": progress.accepted_source_tokens,
        "accepted_document_count": progress.accepted_document_count,
        "inspected_document_count": progress.inspected_document_count,
        "source_bytes_processed": progress.source_bytes_processed,
        "train_written_tokens": progress.train_written_tokens,
        "validation_written_tokens": progress.validation_written_tokens,
        "train_source_tokens": progress.train_source_tokens,
        "validation_source_tokens": progress.validation_source_tokens,
        "train_document_count": progress.train_document_count,
        "validation_document_count": progress.validation_document_count,
        "inserted_eod_count": progress.inserted_eod_count,
        "confirmed_train_bytes": progress.confirmed_train_byte_size,
        "confirmed_validation_bytes": progress.confirmed_validation_byte_size,
        "per_cluster": progress.per_cluster,
        "structural_rejections": progress.structural_rejections,
        "cluster_exclusions": progress.cluster_exclusions,
        "manifest_path": str(manifest_path),
    }


def _cluster_digest(progress: Progress) -> str:
    present = {
        cid: progress.per_cluster[cid]["documents"]
        for cid in sorted(progress.per_cluster, key=lambda x: int(x))
        if progress.per_cluster[cid]["documents"]
    }
    return ",".join(f"{k}:{v}" for k, v in present.items()) or "none"


def _dict_total(d: dict[str, int]) -> int:
    return sum(d.values())


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
