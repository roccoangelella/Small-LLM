"""Human-facing, append-only progress snapshots for long corpus builds.

``progress.json`` is the crash-safe source of truth.  This module deliberately
keeps ``progress.csv`` separate: it is an operational log with live, possibly
uncommitted counters, useful while a multi-day build is still running.
"""

from __future__ import annotations

import csv
import os
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dataset import config

from .checkpoint import Progress
from .workplan import WorkItem, WorkPlan

if TYPE_CHECKING:
    from .writer import BinaryCorpusWriter


CSV_FIELDS = (
    "timestamp_utc",
    "event",
    "complete",
    "work_items_completed",
    "work_items_total",
    "work_items_percent",
    "current_work_item_index",
    "current_filename",
    "current_range_start",
    "current_range_end",
    "source_bytes_completed",
    "source_bytes_total",
    "source_percent",
    "inspected_documents",
    "accepted_documents",
    "accepted_source_tokens",
    "target_source_tokens",
    "target_percent",
    "train_written_tokens",
    "validation_written_tokens",
    "total_written_tokens",
    "inserted_eod_tokens",
    "confirmed_output_bytes",
    "pending_output_bytes",
    "observed_output_bytes",
    "invocation_elapsed_seconds",
    "source_mib_per_second",
    "written_mib_per_second",
    "last_checkpoint_time",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class ProgressReporter:
    """Write periodic live snapshots without changing resume semantics."""

    path: Path
    plan: WorkPlan
    target_source_tokens: int
    started: float
    starting_source_bytes: int
    starting_confirmed_bytes: int
    last_snapshot: float

    @classmethod
    def create(
        cls,
        path: Path,
        plan: WorkPlan,
        progress: Progress,
        target_source_tokens: int,
    ) -> "ProgressReporter":
        now = time.monotonic()
        return cls(
            path=path,
            plan=plan,
            target_source_tokens=target_source_tokens,
            started=now,
            starting_source_bytes=progress.source_bytes_processed,
            starting_confirmed_bytes=(
                progress.confirmed_train_byte_size
                + progress.confirmed_validation_byte_size
            ),
            last_snapshot=now,
        )

    def maybe_snapshot(
        self,
        progress: Progress,
        *,
        writer: "BinaryCorpusWriter | None",
        item: WorkItem | None,
    ) -> None:
        if time.monotonic() - self.last_snapshot >= config.PROGRESS_CSV_HEARTBEAT_SECONDS:
            self.snapshot("heartbeat", progress, writer=writer, item=item)

    def snapshot(
        self,
        event: str,
        progress: Progress,
        *,
        writer: "BinaryCorpusWriter | None" = None,
        item: WorkItem | None = None,
    ) -> None:
        """Append and fsync one CSV row.

        Counters from ``progress`` can be newer than ``progress.json`` until
        the next checkpoint.  ``pending_output_bytes`` makes that distinction
        explicit rather than hiding it.
        """

        now = time.monotonic()
        elapsed = max(0.0, now - self.started)
        confirmed = (
            progress.confirmed_train_byte_size + progress.confirmed_validation_byte_size
        )
        pending = writer.written_since_checkpoint if writer is not None else 0
        total_source_bytes = sum(source.size for source in self.plan.source_files)
        total_written = progress.train_written_tokens + progress.validation_written_tokens
        source_delta = progress.source_bytes_processed - self.starting_source_bytes
        written_delta = confirmed + pending - self.starting_confirmed_bytes
        row = {
            "timestamp_utc": _utc_now(),
            "event": event,
            "complete": str(progress.complete).lower(),
            "work_items_completed": progress.work_item_index,
            "work_items_total": len(self.plan.work_items),
            "work_items_percent": _percent(progress.work_item_index, len(self.plan.work_items)),
            "current_work_item_index": item.index if item is not None else "",
            "current_filename": item.filename if item is not None else "",
            "current_range_start": item.range_start if item is not None else "",
            "current_range_end": item.range_end if item is not None else "",
            "source_bytes_completed": progress.source_bytes_processed,
            "source_bytes_total": total_source_bytes,
            "source_percent": _percent(progress.source_bytes_processed, total_source_bytes),
            "inspected_documents": progress.inspected_document_count,
            "accepted_documents": progress.accepted_document_count,
            "accepted_source_tokens": progress.accepted_source_tokens,
            "target_source_tokens": self.target_source_tokens,
            "target_percent": _percent(progress.accepted_source_tokens, self.target_source_tokens),
            "train_written_tokens": progress.train_written_tokens,
            "validation_written_tokens": progress.validation_written_tokens,
            "total_written_tokens": total_written,
            "inserted_eod_tokens": progress.inserted_eod_count,
            "confirmed_output_bytes": confirmed,
            "pending_output_bytes": pending,
            "observed_output_bytes": confirmed + pending,
            "invocation_elapsed_seconds": f"{elapsed:.3f}",
            "source_mib_per_second": _rate_mib(source_delta, elapsed),
            "written_mib_per_second": _rate_mib(written_delta, elapsed),
            "last_checkpoint_time": progress.last_checkpoint_time,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not self.path.exists() or self.path.stat().st_size == 0
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer_csv = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            if new_file:
                writer_csv.writeheader()
            writer_csv.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())
        self.last_snapshot = now


def read_latest_snapshot(path: Path) -> dict[str, str] | None:
    """Read only the final CSV row without retaining the whole log in memory."""

    if not path.exists() or path.stat().st_size == 0:
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        final = deque(rows, maxlen=1)
    return dict(final[0]) if final else None


def status_report(output_dir: Path) -> dict[str, Any]:
    """Return durable state plus the most recent live CSV observation."""

    progress = Progress.load(output_dir / config.PROGRESS_FILENAME)
    plan_path = output_dir / config.WORK_PLAN_FILENAME
    total_work_items: int | None = None
    total_source_bytes: int | None = None
    if plan_path.exists():
        from .workplan import load_work_plan

        plan = load_work_plan(plan_path)
        total_work_items = len(plan.work_items)
        total_source_bytes = sum(source.size for source in plan.source_files)
    train_path = output_dir / config.TRAIN_FILENAME
    validation_path = output_dir / config.VALIDATION_FILENAME
    actual_train = train_path.stat().st_size if train_path.exists() else 0
    actual_validation = validation_path.stat().st_size if validation_path.exists() else 0
    confirmed = progress.confirmed_train_byte_size + progress.confirmed_validation_byte_size
    return {
        "complete": progress.complete,
        "accepted_source_tokens": progress.accepted_source_tokens,
        "target_source_tokens": int(progress.run_config.get("target_accepted_source_tokens", 0)),
        "target_percent": _percent(
            progress.accepted_source_tokens,
            int(progress.run_config.get("target_accepted_source_tokens", 0)),
        ),
        "accepted_documents": progress.accepted_document_count,
        "inspected_documents": progress.inspected_document_count,
        "work_items_completed": progress.work_item_index,
        "work_items_total": total_work_items,
        "source_bytes_completed": progress.source_bytes_processed,
        "source_bytes_total": total_source_bytes,
        "confirmed_output_bytes": confirmed,
        "actual_output_bytes": actual_train + actual_validation,
        "uncheckpointed_output_bytes_on_disk": actual_train + actual_validation - confirmed,
        "last_checkpoint_time": progress.last_checkpoint_time,
        "progress_csv": str(output_dir / config.PROGRESS_CSV_FILENAME),
        "latest_snapshot": read_latest_snapshot(output_dir / config.PROGRESS_CSV_FILENAME),
    }


def format_status(report: dict[str, Any]) -> str:
    """Format a small terminal readout; detailed history stays in the CSV."""

    target = report["target_source_tokens"]
    total_items = report["work_items_total"]
    item_text = (
        f"{report['work_items_completed']:,}/{total_items:,}"
        if total_items is not None
        else f"{report['work_items_completed']:,}/?"
    )
    target_text = f"{target:,}" if target else "unknown"
    lines = [
        f"state: {'complete' if report['complete'] else 'in progress'}",
        f"accepted source tokens: {report['accepted_source_tokens']:,}/{target_text} ({report['target_percent']:.6f}%)",
        f"documents: {report['accepted_documents']:,} accepted / {report['inspected_documents']:,} inspected",
        f"work items: {item_text}",
        f"durable output: {report['confirmed_output_bytes']:,} bytes",
        f"on-disk uncheckpointed tail: {report['uncheckpointed_output_bytes_on_disk']:,} bytes",
        f"last checkpoint: {report['last_checkpoint_time'] or 'none yet'}",
        f"live log: {report['progress_csv']}",
    ]
    snapshot = report["latest_snapshot"]
    if snapshot is not None:
        lines.append(
            "latest live snapshot: "
            f"{snapshot['event']} at {snapshot['timestamp_utc']} "
            f"({snapshot['target_percent']}% target, "
            f"{snapshot['observed_output_bytes']} observed output bytes)"
        )
    return "\n".join(lines)


def _percent(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return 100.0 * numerator / denominator


def _rate_mib(byte_count: int, elapsed: float) -> float:
    if elapsed <= 0:
        return 0.0
    return byte_count / (1024 * 1024) / elapsed
