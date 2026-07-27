"""Crash-safe checkpoints and exact resume for the token-only build.

A checkpoint is durable only after the writer has flushed and ``fsync`` ed both
binary files and the resulting confirmed byte sizes and progress counters have
been atomically saved to ``progress.json``.  On ``--resume`` the persisted state
is validated against the current effective configuration (frozen policy, work
plan, seed, format, cluster policy, target, split, etc.); incompatible settings
are refused.  Both binary files are truncated to their confirmed checkpoint
sizes so uncommitted tail bytes are discarded and reprocessed without
duplication.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from dataset import config

from .storage import canonical_json_bytes, read_json, write_json_atomic


LOGGER = logging.getLogger(__name__)


# Per-cluster counter keys.  Kept explicit so the manifest and progress never
# drift in shape.
_CLUSTER_KEYS = ("documents", "source_tokens", "written_tokens", "inserted_eods")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Progress:
    """All durable build state.  Atomically persisted as ``progress.json``."""

    schema_version: int
    complete: bool
    dataset: str
    revision: str
    work_plan_hash: str
    work_item_index: int
    item_resume_record_start: int | None
    confirmed_train_byte_size: int
    confirmed_validation_byte_size: int
    train_written_tokens: int
    validation_written_tokens: int
    train_source_tokens: int
    validation_source_tokens: int
    train_inserted_eod_count: int
    validation_inserted_eod_count: int
    train_document_count: int
    validation_document_count: int
    accepted_source_tokens: int
    accepted_document_count: int
    inserted_eod_count: int
    inspected_document_count: int
    source_bytes_processed: int
    per_cluster: dict[str, dict[str, int]]
    structural_rejections: dict[str, int]
    cluster_exclusions: dict[str, int]
    run_config_hash: str
    run_config: dict[str, object]
    build_start_time: str
    last_checkpoint_time: str
    last_checkpoint_written_bytes: int

    # ----- construction -------------------------------------------------

    @classmethod
    def new(
        cls,
        effective: "config.EffectiveConfig",
        *,
        dataset: str,
        revision: str,
        work_plan_hash: str,
        build_start_time: str,
    ) -> "Progress":
        per_cluster = {str(cid): {k: 0 for k in _CLUSTER_KEYS}
                       for cid in sorted(config.ALL_CLUSTER_IDS)}
        return cls(
            schema_version=config.PROGRESS_SCHEMA_VERSION,
            complete=False,
            dataset=dataset,
            revision=revision,
            work_plan_hash=work_plan_hash,
            work_item_index=0,
            item_resume_record_start=None,
            confirmed_train_byte_size=0,
            confirmed_validation_byte_size=0,
            train_written_tokens=0,
            validation_written_tokens=0,
            train_source_tokens=0,
            validation_source_tokens=0,
            train_inserted_eod_count=0,
            validation_inserted_eod_count=0,
            train_document_count=0,
            validation_document_count=0,
            accepted_source_tokens=0,
            accepted_document_count=0,
            inserted_eod_count=0,
            inspected_document_count=0,
            source_bytes_processed=0,
            per_cluster=per_cluster,
            structural_rejections={},
            cluster_exclusions={},
            run_config_hash="",
            run_config={},
            build_start_time=build_start_time,
            last_checkpoint_time="",
            last_checkpoint_written_bytes=0,
        )

    # ----- accounting helpers ------------------------------------------

    def _cluster(self, cluster_id: int) -> dict[str, int]:
        key = str(cluster_id)
        if key not in self.per_cluster:
            self.per_cluster[key] = {k: 0 for k in _CLUSTER_KEYS}
        return self.per_cluster[key]

    def accept(
        self,
        *,
        cluster_id: int,
        source_tokens: int,
        written_tokens: int,
        inserted_eods: int,
        validation: bool,
    ) -> None:
        self.accepted_document_count += 1
        self.accepted_source_tokens += source_tokens
        self.inserted_eod_count += inserted_eods
        if validation:
            self.validation_written_tokens += written_tokens
            self.validation_source_tokens += source_tokens
            self.validation_inserted_eod_count += inserted_eods
            self.validation_document_count += 1
        else:
            self.train_written_tokens += written_tokens
            self.train_source_tokens += source_tokens
            self.train_inserted_eod_count += inserted_eods
            self.train_document_count += 1
        counters = self._cluster(cluster_id)
        counters["documents"] += 1
        counters["source_tokens"] += source_tokens
        counters["written_tokens"] += written_tokens
        counters["inserted_eods"] += inserted_eods

    def reject_structural(self, reason: str) -> None:
        self.structural_rejections[reason] = self.structural_rejections.get(reason, 0) + 1

    def inspect_record(self) -> None:
        self.inspected_document_count += 1

    def complete_work_item(self, source_bytes: int) -> None:
        if source_bytes < 0:
            raise ValueError("completed source byte count cannot be negative")
        self.source_bytes_processed += source_bytes

    def exclude_cluster(self, cluster_id: int) -> None:
        key = str(cluster_id)
        self.cluster_exclusions[key] = self.cluster_exclusions.get(key, 0) + 1

    # ----- persistence --------------------------------------------------

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "complete": self.complete,
            "dataset": self.dataset,
            "revision": self.revision,
            "work_plan_hash": self.work_plan_hash,
            "work_item_index": self.work_item_index,
            "item_resume_record_start": self.item_resume_record_start,
            "confirmed_train_byte_size": self.confirmed_train_byte_size,
            "confirmed_validation_byte_size": self.confirmed_validation_byte_size,
            "train_written_tokens": self.train_written_tokens,
            "validation_written_tokens": self.validation_written_tokens,
            "train_source_tokens": self.train_source_tokens,
            "validation_source_tokens": self.validation_source_tokens,
            "train_inserted_eod_count": self.train_inserted_eod_count,
            "validation_inserted_eod_count": self.validation_inserted_eod_count,
            "train_document_count": self.train_document_count,
            "validation_document_count": self.validation_document_count,
            "accepted_source_tokens": self.accepted_source_tokens,
            "accepted_document_count": self.accepted_document_count,
            "inserted_eod_count": self.inserted_eod_count,
            "inspected_document_count": self.inspected_document_count,
            "source_bytes_processed": self.source_bytes_processed,
            "per_cluster": self.per_cluster,
            "structural_rejections": self.structural_rejections,
            "cluster_exclusions": self.cluster_exclusions,
            "run_config_hash": self.run_config_hash,
            "run_config": self.run_config,
            "build_start_time": self.build_start_time,
            "last_checkpoint_time": self.last_checkpoint_time,
            "last_checkpoint_written_bytes": self.last_checkpoint_written_bytes,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "Progress":
        if payload.get("schema_version") != config.PROGRESS_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported progress schema version {payload.get('schema_version')!r}; "
                f"expected {config.PROGRESS_SCHEMA_VERSION}"
            )
        return cls(
            schema_version=int(payload["schema_version"]),
            complete=bool(payload.get("complete", False)),
            dataset=str(payload["dataset"]),
            revision=str(payload["revision"]),
            work_plan_hash=str(payload["work_plan_hash"]),
            work_item_index=int(payload["work_item_index"]),
            item_resume_record_start=(
                int(payload["item_resume_record_start"])
                if payload.get("item_resume_record_start") is not None
                else None
            ),
            confirmed_train_byte_size=int(payload["confirmed_train_byte_size"]),
            confirmed_validation_byte_size=int(payload["confirmed_validation_byte_size"]),
            train_written_tokens=int(payload["train_written_tokens"]),
            validation_written_tokens=int(payload["validation_written_tokens"]),
            train_source_tokens=int(payload["train_source_tokens"]),
            validation_source_tokens=int(payload["validation_source_tokens"]),
            train_inserted_eod_count=int(payload["train_inserted_eod_count"]),
            validation_inserted_eod_count=int(payload["validation_inserted_eod_count"]),
            train_document_count=int(payload.get("train_document_count", 0)),
            validation_document_count=int(payload.get("validation_document_count", 0)),
            accepted_source_tokens=int(payload["accepted_source_tokens"]),
            accepted_document_count=int(payload["accepted_document_count"]),
            inserted_eod_count=int(payload["inserted_eod_count"]),
            inspected_document_count=int(payload["inspected_document_count"]),
            source_bytes_processed=int(payload["source_bytes_processed"]),
            per_cluster=dict(payload.get("per_cluster", {})),
            structural_rejections=dict(payload.get("structural_rejections", {})),
            cluster_exclusions=dict(payload.get("cluster_exclusions", {})),
            run_config_hash=str(payload.get("run_config_hash", "")),
            run_config=dict(payload.get("run_config", {})),
            build_start_time=str(payload.get("build_start_time", "")),
            last_checkpoint_time=str(payload.get("last_checkpoint_time", "")),
            last_checkpoint_written_bytes=int(payload.get("last_checkpoint_written_bytes", 0)),
        )

    def save(self, path: Path) -> None:
        self.last_checkpoint_time = _utc_now_iso()
        write_json_atomic(path, self.to_dict())

    @classmethod
    def load(cls, path: Path) -> "Progress":
        return cls.from_dict(read_json(path))


# ---------------------------------------------------------------------------
# Resume compatibility
# ---------------------------------------------------------------------------


def run_signature(
    effective: "config.EffectiveConfig",
    *,
    work_plan_hash: str,
) -> dict[str, object]:
    """Return the configuration fingerprint that must match for a safe resume.

    Only settings that affect the *output identity* are included.  Buffer sizes
    and checkpoint thresholds change throughput/recency, not the bytes produced,
    so they are intentionally excluded.
    """

    return {
        "dataset": config.DATASET_REPOSITORY,
        "revision": config.DATASET_REVISION,
        "selection_seed": config.SELECTION_SEED,
        "source_glob": config.SOURCE_DATA_GLOB,
        "eod_token_id": config.EOD_TOKEN_ID,
        "token_min": config.TOKEN_MIN,
        "token_max": config.TOKEN_MAX,
        "int_type": config.INT_TYPE,
        "byte_order": config.BYTE_ORDER,
        "validation_probability": config.VALIDATION_PROBABILITY,
        "split_hash_version": config.SPLIT_HASH_VERSION,
        "accepted_cluster_ids": sorted(config.ACCEPTED_CLUSTER_IDS),
        "excluded_cluster_ids": sorted(config.EXCLUDED_CLUSTER_IDS),
        "target_accepted_source_tokens": effective.target_accepted_source_tokens,
        "minimum_accepted_source_tokens": effective.minimum_accepted_source_tokens,
        "maximum_accepted_source_tokens": effective.maximum_accepted_source_tokens,
        "region_bytes": effective.region_bytes,
        "max_work_items": effective.max_work_items,
        "strict": effective.strict,
        "work_plan_hash": work_plan_hash,
    }


def run_signature_hash(signature: dict[str, object]) -> str:
    return hashlib.sha256(canonical_json_bytes(signature)).hexdigest()


def stamp_progress(progress: Progress, effective: "config.EffectiveConfig") -> None:
    """Attach the resume compatibility signature to a fresh progress object."""

    signature = run_signature(effective, work_plan_hash=progress.work_plan_hash)
    progress.run_config = signature
    progress.run_config_hash = run_signature_hash(signature)


def validate_resume(progress: Progress, effective: "config.EffectiveConfig", work_plan_hash: str) -> None:
    """Refuse to resume when a setting that affects output identity changed."""

    _validate_progress_invariants(progress)
    if progress.dataset != config.DATASET_REPOSITORY:
        _fail("dataset", progress.dataset, config.DATASET_REPOSITORY)
    if progress.revision != config.DATASET_REVISION:
        _fail("revision", progress.revision, config.DATASET_REVISION)
    if progress.work_plan_hash != work_plan_hash:
        _fail("work_plan_hash", progress.work_plan_hash, work_plan_hash)
    current = run_signature(effective, work_plan_hash=work_plan_hash)
    current_hash = run_signature_hash(current)
    if not progress.run_config_hash:
        raise ValueError(
            "Refusing to resume: checkpoint has no configuration hash. "
            "Delete the output directory and rebuild."
        )
    if progress.run_config_hash != run_signature_hash(progress.run_config):
        raise ValueError(
            "Refusing to resume: checkpoint configuration hash is corrupt. "
            "Delete the output directory and rebuild."
        )
    if progress.run_config_hash != current_hash:
        differ = _diff_keys(progress.run_config, current)
        raise ValueError(
            "Refusing to resume: settings that affect the output identity have changed "
            f"({', '.join(differ) or 'signature'}). Delete the output directory or revert "
            "the overrides and resume again."
        )


def _validate_progress_invariants(progress: Progress) -> None:
    """Reject a corrupt checkpoint before touching binary output files."""

    problems: list[str] = []
    integer_fields = {
        "work_item_index": progress.work_item_index,
        "confirmed_train_byte_size": progress.confirmed_train_byte_size,
        "confirmed_validation_byte_size": progress.confirmed_validation_byte_size,
        "train_written_tokens": progress.train_written_tokens,
        "validation_written_tokens": progress.validation_written_tokens,
        "train_source_tokens": progress.train_source_tokens,
        "validation_source_tokens": progress.validation_source_tokens,
        "accepted_source_tokens": progress.accepted_source_tokens,
        "accepted_document_count": progress.accepted_document_count,
        "inserted_eod_count": progress.inserted_eod_count,
        "inspected_document_count": progress.inspected_document_count,
        "source_bytes_processed": progress.source_bytes_processed,
    }
    for name, value in integer_fields.items():
        if value < 0:
            problems.append(f"{name} is negative")
    if (
        progress.item_resume_record_start is not None
        and progress.item_resume_record_start < 0
    ):
        problems.append("item_resume_record_start is negative")
    if progress.confirmed_train_byte_size != progress.train_written_tokens * 2:
        problems.append("confirmed train bytes do not match train written tokens")
    if (
        progress.confirmed_validation_byte_size
        != progress.validation_written_tokens * 2
    ):
        problems.append("confirmed validation bytes do not match validation written tokens")
    if progress.last_checkpoint_written_bytes != (
        progress.confirmed_train_byte_size + progress.confirmed_validation_byte_size
    ):
        problems.append("last checkpoint byte total does not match confirmed file sizes")
    if progress.train_source_tokens + progress.validation_source_tokens != (
        progress.accepted_source_tokens
    ):
        problems.append("split source-token counts do not sum to accepted source tokens")
    if progress.train_document_count + progress.validation_document_count != (
        progress.accepted_document_count
    ):
        problems.append("split document counts do not sum to accepted documents")
    if progress.train_inserted_eod_count + progress.validation_inserted_eod_count != (
        progress.inserted_eod_count
    ):
        problems.append("split EOD counts do not sum to inserted EODs")
    if progress.train_written_tokens + progress.validation_written_tokens != (
        progress.accepted_source_tokens + progress.inserted_eod_count
    ):
        problems.append("written tokens do not equal source tokens plus inserted EODs")
    if progress.accepted_document_count > progress.inspected_document_count:
        problems.append("accepted documents exceed inspected documents")
    cluster_11 = progress.per_cluster.get("11", {})
    if int(cluster_11.get("documents", 0)) or int(
        cluster_11.get("source_tokens", 0)
    ):
        problems.append("excluded cluster 11 appears in accepted counters")
    if problems:
        raise ValueError(
            "Refusing to resume from an inconsistent checkpoint: "
            + "; ".join(problems)
        )


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------


def truncate_to_confirmed(
    progress: Progress,
    train_path: Path,
    validation_path: Path,
) -> None:
    """Truncate both binaries to their confirmed checkpoint byte sizes."""

    targets = (
        (train_path, progress.confirmed_train_byte_size),
        (validation_path, progress.confirmed_validation_byte_size),
    )
    for path, confirmed in targets:
        _truncate(path, confirmed)


def _truncate(path: Path, confirmed: int) -> None:
    if not path.exists():
        if confirmed == 0:
            # Nothing to truncate; the writer will create it.
            return
        raise RuntimeError(
            f"Output file {path} is missing but the checkpoint confirms "
            f"{confirmed} bytes. Restore the file or rebuild the corpus."
        )
    actual = path.stat().st_size
    if actual > confirmed:
        LOGGER.warning(
            "Truncating uncheckpointed binary tail: %s from %d to %d bytes",
            path,
            actual,
            confirmed,
        )
        with path.open("r+b") as handle:
            handle.truncate(confirmed)
            handle.flush()
            os.fsync(handle.fileno())
    elif actual < confirmed:
        raise RuntimeError(
            f"Output file {path} is smaller ({actual}) than the confirmed checkpoint "
            f"size ({confirmed}); the corpus is inconsistent. Delete the output "
            "directory and rebuild."
        )


def remove_uncheckpointed_corpus(output_dir: Path) -> None:
    """Delete generated corpus files so a fresh build can start clean."""

    for name in (
        config.TRAIN_FILENAME,
        config.VALIDATION_FILENAME,
        config.PROGRESS_FILENAME,
        config.WORK_PLAN_FILENAME,
        config.MANIFEST_FILENAME,
    ):
        path = output_dir / name
        if path.exists():
            path.unlink()


def _fail(name: str, old: object, new: object) -> None:
    raise ValueError(
        f"Refusing to resume: {name} changed (checkpoint has {old!r}, current is {new!r}). "
        "Delete the output directory and rebuild."
    )


def _diff_keys(a: dict[str, object], b: dict[str, object]) -> list[str]:
    keys = set(a) | set(b)
    return sorted(k for k in keys if a.get(k) != b.get(k))
