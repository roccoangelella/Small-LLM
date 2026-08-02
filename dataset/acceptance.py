"""Fail-closed evidence verifier for dataset operational qualification.

This module does not claim that a live pilot, interruption/resume, Google Drive
roundtrip, or completed-resume idempotence passed unless concrete artifacts for
those gates are supplied and verified.  It also provides a deterministic pilot
snapshot command used before interruption and before the final idempotence run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from dataset import config
from dataset.src.storage import canonical_json_bytes, read_json
from dataset.src.verify import verify
from dataset.src.workplan import load_work_plan

APPROVED_WEIGHTS_SHA256 = (
    "76e82e22760adcac59c7294fe9bac11358f5a8b7a26035aae64c3f2e6fa1acb7"
)
DEFAULT_CALIBRATION_DIR = Path("/data/climbmix-mixture-calibration")
DEFAULT_OPS_DIR = Path("/data/climbmix-ops")
DEFAULT_REPORT_JSON_NAME = "dataset_acceptance_report.json"
DEFAULT_REPORT_MD_NAME = "dataset_acceptance_report.md"


def sha256_file(path: Path, *, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            hasher.update(block)
    return hasher.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _atomic_write_text(target_path: Path, content: str) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = target_path.with_name(f"{target_path.name}.tmp.{uuid.uuid4().hex}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(target_path)
        directory_fd = os.open(target_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(target_path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write_text(
        target_path,
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def validate_weights_file(
    weights_path: Path,
    *,
    expected_sha256: str | None = APPROVED_WEIGHTS_SHA256,
) -> tuple[dict[str, int], str]:
    if not weights_path.is_file():
        raise FileNotFoundError(f"weights file not found: {weights_path}")
    raw = _read_object(weights_path)
    expected_keys = {
        *(str(cluster) for cluster in range(1, 11)),
        *(str(cluster) for cluster in range(12, 21)),
    }
    if set(raw) != expected_keys:
        missing = sorted(expected_keys - set(raw), key=int)
        extra = sorted(set(raw) - expected_keys)
        raise ValueError(
            f"weights must contain exactly clusters 1-10 and 12-20; "
            f"missing={missing}, extra={extra}"
        )
    weights: dict[str, int] = {}
    for cluster in sorted(expected_keys, key=int):
        value = raw[cluster]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(
                f"cluster {cluster} weight must be a positive integer, got {value!r}"
            )
        weights[cluster] = value
    digest = sha256_file(weights_path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(
            f"weights SHA-256 mismatch: expected {expected_sha256}, got {digest}"
        )
    return weights, digest


def run_environment_preflight(repo_root: Path | None = None) -> dict[str, Any]:
    root = (repo_root or Path.cwd()).resolve()

    def git(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *arguments],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )

    commit = git("rev-parse", "HEAD")
    tracked_secrets = git("ls-files", "--", ".env", ".secrets")
    status = git("status", "--porcelain=v1", "--untracked-files=all")
    return {
        "git_commit": commit.stdout.strip() if commit.returncode == 0 else "unknown",
        "python_version": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "secrets_untracked": tracked_secrets.returncode == 0
        and not tracked_secrets.stdout.strip(),
        "worktree_clean": status.returncode == 0 and not status.stdout.strip(),
    }


def check_temporary_artifacts(directory: Path) -> list[str]:
    if not directory.exists():
        return []
    artifacts: list[str] = []
    for path in directory.rglob("*"):
        if not path.is_file():
            continue
        name = path.name
        if (
            name.endswith((".tmp", ".part", ".safe.json"))
            or ".tmp." in name
            or name.startswith(".smoke_test_")
        ):
            artifacts.append(str(path))
    return sorted(artifacts)


def validate_calibration(
    calibration_dir: Path,
    weights_path: Path,
    *,
    expected_weights_sha256: str = APPROVED_WEIGHTS_SHA256,
) -> dict[str, Any]:
    report_path = calibration_dir / "mixture_report.json"
    progress_path = calibration_dir / "mixture_progress.json"
    work_plan_path = calibration_dir / config.WORK_PLAN_FILENAME
    report = _read_object(report_path)
    progress = _read_object(progress_path)
    plan = load_work_plan(work_plan_path)
    weights, weights_sha256 = validate_weights_file(
        weights_path, expected_sha256=expected_weights_sha256
    )
    problems: list[str] = []

    expected_source_files = [
        {"path": source.path, "size": source.size} for source in plan.source_files
    ]
    expected_source_bytes = sum(source.size for source in plan.source_files)
    expected_cluster_keys = {str(cluster) for cluster in range(1, 21)}
    accepted_keys = {str(cluster) for cluster in config.ACCEPTED_CLUSTER_IDS}

    if report.get("complete") is not True or progress.get("complete") is not True:
        problems.append("calibration is not marked complete")
    if report.get("dataset") != config.DATASET_REPOSITORY:
        problems.append("calibration dataset identity mismatch")
    if report.get("revision") != config.DATASET_REVISION:
        problems.append("calibration source revision mismatch")
    if report.get("source_glob") != config.SOURCE_DATA_GLOB:
        problems.append("calibration source glob mismatch")
    if report.get("source_files") != expected_source_files:
        problems.append("calibration source-file list differs from work_plan.json")
    if report.get("source_bytes_scanned") != expected_source_bytes:
        problems.append("report source-byte total differs from work_plan.json")
    if progress.get("source_bytes_covered") != expected_source_bytes:
        problems.append("progress source-byte coverage differs from work_plan.json")
    if report.get("work_plan_hash") != plan.hash:
        problems.append("report work-plan hash mismatch")
    if progress.get("work_plan_hash") != plan.hash:
        problems.append("progress work-plan hash mismatch")
    if progress.get("completed_work_items") != len(plan.work_items):
        problems.append("progress completed-work-item count mismatch")
    if progress.get("next_work_item_index") != len(plan.work_items):
        problems.append("progress next-work-item index mismatch")

    embedded_report_hash = hashlib.sha256(
        canonical_json_bytes(report, exclude_keys=("report_sha256",))
    ).hexdigest()
    if report.get("report_sha256") != embedded_report_hash:
        problems.append("report canonical self-hash mismatch")
    if progress.get("report_sha256") != embedded_report_hash:
        problems.append("progress report hash mismatch")
    if report.get("weights_sha256") != weights_sha256:
        problems.append("report weights hash mismatch")
    if progress.get("weights_sha256") != weights_sha256:
        problems.append("progress weights hash mismatch")

    token_counts = report.get("all_cluster_source_tokens")
    document_counts = report.get("all_cluster_document_counts")
    if not isinstance(token_counts, dict) or set(token_counts) != expected_cluster_keys:
        problems.append("report has an invalid all-cluster token map")
        token_counts = {}
    if not isinstance(document_counts, dict) or set(document_counts) != expected_cluster_keys:
        problems.append("report has an invalid all-cluster document map")
        document_counts = {}
    if token_counts and any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in token_counts.values()
    ):
        problems.append("report contains a non-positive cluster token total")
    if document_counts and any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in document_counts.values()
    ):
        problems.append("report contains a non-positive cluster document total")
    if token_counts:
        expected_weights = {cluster: int(token_counts[cluster]) for cluster in accepted_keys}
        if weights != dict(sorted(expected_weights.items(), key=lambda item: int(item[0]))):
            problems.append("weights do not exactly match accepted report token totals")
        if report.get("all_source_tokens") != sum(int(v) for v in token_counts.values()):
            problems.append("all-source-token total mismatch")
        accepted_total = sum(int(token_counts[key]) for key in accepted_keys)
        if report.get("accepted_source_tokens") != accepted_total:
            problems.append("accepted-source-token total mismatch")
    if report.get("record_count") != progress.get("record_count"):
        problems.append("report/progress record counts differ")
    if report.get("accepted_cluster_ids") != sorted(config.ACCEPTED_CLUSTER_IDS):
        problems.append("accepted-cluster policy mismatch")
    if report.get("excluded_cluster_ids") != sorted(config.EXCLUDED_CLUSTER_IDS):
        problems.append("excluded-cluster policy mismatch")

    return {
        "passed": not problems,
        "problems": problems,
        "source_files": len(plan.source_files),
        "work_items": len(plan.work_items),
        "source_bytes": expected_source_bytes,
        "record_count": report.get("record_count"),
        "all_source_tokens": report.get("all_source_tokens"),
        "accepted_source_tokens": report.get("accepted_source_tokens"),
        "weights_sha256": weights_sha256,
        "report_sha256": embedded_report_hash,
        "work_plan_hash": plan.hash,
        "raw_report_sha256": sha256_file(report_path),
        "raw_progress_sha256": sha256_file(progress_path),
        "raw_work_plan_sha256": sha256_file(work_plan_path),
    }


def validate_command_evidence(
    *,
    exit_code_path: Path,
    log_path: Path,
    required_markers: Sequence[str],
) -> dict[str, Any]:
    problems: list[str] = []
    if not exit_code_path.is_file():
        problems.append(f"missing exit-code file: {exit_code_path}")
        exit_code: int | None = None
    else:
        try:
            exit_code = int(exit_code_path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            exit_code = None
            problems.append(f"invalid exit-code file: {exit_code_path}")
        if exit_code not in (None, 0):
            problems.append(f"command exit code was {exit_code}")
    if not log_path.is_file():
        problems.append(f"missing log file: {log_path}")
        log_text = ""
    else:
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        for marker in required_markers:
            if marker not in log_text:
                problems.append(f"log is missing required marker {marker!r}")
        if "RESULT=FAIL" in log_text or "Traceback (most recent call last)" in log_text:
            problems.append("log contains a failure marker or traceback")
    return {
        "passed": not problems,
        "problems": problems,
        "exit_code": exit_code,
        "log_sha256": sha256_file(log_path) if log_path.is_file() else None,
    }


def _normalized_local_shards(raw: object) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    fields = (
        "filename",
        "split",
        "byte_size",
        "checksum",
        "first_block_id",
        "last_block_id",
    )
    normalized = [
        {field: entry.get(field) for field in fields}
        for entry in raw
        if isinstance(entry, Mapping)
    ]
    return sorted(normalized, key=lambda entry: str(entry.get("filename")))


def _normalized_drive_shards(raw: object) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    fields = (
        "filename",
        "drive_file_id",
        "byte_size",
        "local_sha256",
        "remote_durable",
        "configuration_hash",
        "schema_hash",
    )
    normalized = [
        {field: entry.get(field) for field in fields}
        for entry in raw
        if isinstance(entry, Mapping)
    ]
    return sorted(normalized, key=lambda entry: str(entry.get("filename")))


def pilot_snapshot(output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    progress = _read_object(output_dir / config.PROGRESS_FILENAME)
    manifest_path = output_dir / config.MANIFEST_FILENAME
    drive_path = output_dir / "drive_manifest.json"
    manifest = _read_object(manifest_path) if manifest_path.is_file() else {}
    drive = _read_object(drive_path) if drive_path.is_file() else {}
    local_shards = _normalized_local_shards(
        manifest.get("shards", progress.get("finalized_shards", []))
    )
    return {
        "version": 1,
        "output_dir": str(output_dir),
        "complete": progress.get("complete") is True,
        "accepted_source_tokens": progress.get("accepted_source_tokens_incorporated"),
        "production": progress.get("production"),
        "source_reader": progress.get("source_reader"),
        "local_shards": local_shards,
        "drive_manifest_identity": {
            "run_id": drive.get("run_id"),
            "configuration_hash": drive.get("configuration_hash"),
            "schema_hash": drive.get("schema_hash"),
        },
        "drive_shards": _normalized_drive_shards(drive.get("shards", [])),
    }


def write_pilot_snapshot(output_dir: Path, snapshot_path: Path) -> dict[str, Any]:
    snapshot = pilot_snapshot(output_dir)
    _atomic_write_json(snapshot_path, snapshot)
    return snapshot


def compare_idempotence(
    baseline: Mapping[str, Any], current: Mapping[str, Any]
) -> dict[str, Any]:
    problems: list[str] = []
    if dict(baseline) != dict(current):
        for key in sorted(set(baseline) | set(current)):
            if baseline.get(key) != current.get(key):
                problems.append(f"completed-resume state changed at {key}")
    return {"passed": not problems, "problems": problems}


def validate_interruption_resume(
    interrupted: Mapping[str, Any], completed: Mapping[str, Any]
) -> dict[str, Any]:
    problems: list[str] = []
    if interrupted.get("complete") is not False:
        problems.append("interrupted snapshot is not an incomplete durable state")
    if completed.get("complete") is not True:
        problems.append("completed snapshot is not complete")
    if interrupted.get("production") != completed.get("production"):
        problems.append("production identity changed across resume")
    before_tokens = interrupted.get("accepted_source_tokens")
    after_tokens = completed.get("accepted_source_tokens")
    if (
        isinstance(before_tokens, bool)
        or not isinstance(before_tokens, int)
        or before_tokens <= 0
    ):
        problems.append("interrupted snapshot has no positive durable token count")
    if (
        isinstance(after_tokens, bool)
        or not isinstance(after_tokens, int)
        or not isinstance(before_tokens, int)
        or after_tokens < before_tokens
    ):
        problems.append("accepted token count regressed across resume")
    before_reader = interrupted.get("source_reader")
    after_reader = completed.get("source_reader")
    before_documents = (
        before_reader.get("documents_consumed") if isinstance(before_reader, Mapping) else None
    )
    after_documents = (
        after_reader.get("documents_consumed") if isinstance(after_reader, Mapping) else None
    )
    if (
        isinstance(before_documents, bool)
        or not isinstance(before_documents, int)
        or isinstance(after_documents, bool)
        or not isinstance(after_documents, int)
        or after_documents <= before_documents
    ):
        problems.append("durable source cursor did not advance after resume")

    completed_local = {
        str(entry.get("filename")): entry
        for entry in completed.get("local_shards", [])
        if isinstance(entry, Mapping)
    }
    for entry in interrupted.get("local_shards", []):
        if not isinstance(entry, Mapping):
            problems.append("interrupted snapshot contains an invalid local shard")
            continue
        filename = str(entry.get("filename"))
        if completed_local.get(filename) != entry:
            problems.append(f"durable local shard changed or disappeared: {filename}")

    completed_drive = {
        str(entry.get("filename")): entry
        for entry in completed.get("drive_shards", [])
        if isinstance(entry, Mapping)
    }
    for entry in interrupted.get("drive_shards", []):
        if not isinstance(entry, Mapping):
            problems.append("interrupted snapshot contains an invalid Drive shard")
            continue
        filename = str(entry.get("filename"))
        if completed_drive.get(filename) != entry:
            problems.append(f"durable Drive shard changed or disappeared: {filename}")
    return {"passed": not problems, "problems": problems}


def validate_pilot(
    output_dir: Path,
    *,
    run_id: str,
    minimum_tokens: int,
    maximum_tokens: int,
    full_scan: bool,
) -> dict[str, Any]:
    problems: list[str] = []
    report = verify(output_dir, full_scan=full_scan)
    if not report.passed:
        problems.extend(report.problems)
    progress = _read_object(output_dir / config.PROGRESS_FILENAME)
    manifest = _read_object(output_dir / config.MANIFEST_FILENAME)
    drive = _read_object(output_dir / "drive_manifest.json")
    production = manifest.get("production")
    if progress.get("complete") is not True:
        problems.append("pilot progress is not complete")
    if not isinstance(production, Mapping):
        problems.append("pilot manifest has no production identity")
        production = {}
    if production.get("run_id") != run_id:
        problems.append("pilot run_id mismatch")
    if production.get("remote_required") is not True:
        problems.append("pilot did not require remote durability")
    if production.get("target_reached") is not True:
        problems.append("pilot did not reach its target")
    accepted = manifest.get("accepted_source_tokens")
    if (
        isinstance(accepted, bool)
        or not isinstance(accepted, int)
        or not minimum_tokens <= accepted <= maximum_tokens
    ):
        problems.append("pilot accepted-source-token total is outside the approved range")
    if progress.get("accepted_source_tokens_incorporated") != accepted:
        problems.append("pilot manifest/progress token totals differ")

    drive_shards = _normalized_drive_shards(drive.get("shards", []))
    local_shards = _normalized_local_shards(manifest.get("shards", []))
    if drive.get("run_id") != run_id:
        problems.append("Drive manifest run_id mismatch")
    if drive.get("configuration_hash") != production.get("configuration_hash"):
        problems.append("Drive manifest configuration hash mismatch")
    if drive.get("schema_hash") != production.get("schema_hash"):
        problems.append("Drive manifest schema hash mismatch")
    if len(drive_shards) != len(local_shards):
        problems.append("Drive/local shard counts differ")
    local_by_name = {str(entry.get("filename")): entry for entry in local_shards}
    drive_ids: set[str] = set()
    for entry in drive_shards:
        filename = str(entry.get("filename"))
        local = local_by_name.get(filename)
        if local is None:
            problems.append(f"Drive manifest contains unknown shard {filename}")
            continue
        if entry.get("remote_durable") is not True:
            problems.append(f"Drive shard is not verified durable: {filename}")
        if entry.get("byte_size") != local.get("byte_size"):
            problems.append(f"Drive/local size mismatch: {filename}")
        if entry.get("local_sha256") != local.get("checksum"):
            problems.append(f"Drive/local hash mismatch: {filename}")
        file_id = entry.get("drive_file_id")
        if not isinstance(file_id, str) or not file_id or file_id in drive_ids:
            problems.append(f"invalid or duplicate Drive file ID: {filename}")
        else:
            drive_ids.add(file_id)

    temporary = check_temporary_artifacts(output_dir)
    if temporary:
        problems.append(f"temporary artifacts remain: {temporary}")
    snapshot = pilot_snapshot(output_dir)
    return {
        "passed": not problems,
        "problems": problems,
        "verify_report": report.as_dict(),
        "accepted_source_tokens": accepted,
        "local_shard_count": len(local_shards),
        "drive_shard_count": len(drive_shards),
        "snapshot": snapshot,
    }


def _gate(action: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        result = action()
        if "passed" not in result:
            result["passed"] = True
        result.setdefault("problems", [])
        return result
    except Exception as error:  # noqa: BLE001 - report every failed evidence boundary
        return {
            "passed": False,
            "problems": [f"{type(error).__name__}: {error}"],
        }


def evaluate_acceptance(
    *,
    repo_root: Path,
    weights_path: Path,
    calibration_dir: Path,
    pilot_output_dir: Path,
    run_id: str,
    interrupted_snapshot_path: Path,
    idempotence_baseline_path: Path,
    ops_dir: Path,
    drive_smoke_log: Path,
    expected_weights_sha256: str = APPROVED_WEIGHTS_SHA256,
    minimum_tokens: int = 9_000_000,
    maximum_tokens: int = 11_000_000,
    full_scan: bool = True,
) -> dict[str, Any]:
    preflight = run_environment_preflight(repo_root)
    gates: dict[str, dict[str, Any]] = {}
    gates["environment"] = {
        "passed": bool(
            preflight["git_commit"] != "unknown"
            and preflight["secrets_untracked"]
            and preflight["worktree_clean"]
        ),
        "problems": [
            message
            for condition, message in (
                (preflight["git_commit"] == "unknown", "Git commit could not be resolved"),
                (not preflight["secrets_untracked"], "secret paths are tracked by Git"),
                (not preflight["worktree_clean"], "Git worktree is not clean"),
            )
            if condition
        ],
        **preflight,
    }
    gates["calibration"] = _gate(
        lambda: validate_calibration(
            calibration_dir,
            weights_path,
            expected_weights_sha256=expected_weights_sha256,
        )
    )
    gates["offline_tests"] = _gate(
        lambda: validate_command_evidence(
            exit_code_path=ops_dir / "offline-tests.exit-code",
            log_path=ops_dir / "logs" / "offline-tests.log",
            required_markers=("RESULT=PASS", "OK"),
        )
    )
    gates["calibration_run"] = _gate(
        lambda: validate_command_evidence(
            exit_code_path=ops_dir / "mixture-calibration.exit-code",
            log_path=ops_dir / "logs" / "mixture-calibration.log",
            required_markers=("RESULT=PASS",),
        )
    )
    gates["drive_smoke"] = _gate(
        lambda: validate_command_evidence(
            exit_code_path=ops_dir / "drive-oauth-smoke.exit-code",
            log_path=drive_smoke_log,
            required_markers=("Smoke test: PASSED",),
        )
    )
    gates["pilot"] = _gate(
        lambda: validate_pilot(
            pilot_output_dir,
            run_id=run_id,
            minimum_tokens=minimum_tokens,
            maximum_tokens=maximum_tokens,
            full_scan=full_scan,
        )
    )

    current_snapshot = gates["pilot"].get("snapshot")
    gates["interruption_resume"] = _gate(
        lambda: validate_interruption_resume(
            _read_object(interrupted_snapshot_path),
            current_snapshot
            if isinstance(current_snapshot, Mapping)
            else pilot_snapshot(pilot_output_dir),
        )
    )
    gates["completed_resume_idempotence"] = _gate(
        lambda: compare_idempotence(
            _read_object(idempotence_baseline_path),
            current_snapshot
            if isinstance(current_snapshot, Mapping)
            else pilot_snapshot(pilot_output_dir),
        )
    )

    passed = all(gate.get("passed") is True for gate in gates.values())
    failures = [
        f"{name}: {problem}"
        for name, gate in gates.items()
        for problem in gate.get("problems", [])
    ]
    return {
        "schema_version": 1,
        "passed": passed,
        "git_commit": preflight["git_commit"],
        "weights_file": str(weights_path.resolve()),
        "calibration_dir": str(calibration_dir.resolve()),
        "pilot_output_dir": str(pilot_output_dir.resolve()),
        "run_id": run_id,
        "expected_weights_sha256": expected_weights_sha256,
        "gates": gates,
        "failures": failures,
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Dataset Operational Qualification Acceptance Report",
        "",
        f"**Overall Status:** {'PASS' if report.get('passed') else 'FAIL'}",
        f"**Git Commit:** `{report.get('git_commit', 'unknown')}`",
        f"**Run ID:** `{report.get('run_id', 'unknown')}`",
        "",
        "## Gates",
        "",
        "| Gate | Status | Problems |",
        "| --- | --- | --- |",
    ]
    gates = report.get("gates", {})
    if isinstance(gates, Mapping):
        for name, raw_gate in gates.items():
            gate = raw_gate if isinstance(raw_gate, Mapping) else {}
            problems = "; ".join(str(value) for value in gate.get("problems", [])) or "—"
            lines.append(
                f"| {name} | {'PASS' if gate.get('passed') else 'FAIL'} | {problems} |"
            )
    failures = report.get("failures", [])
    if failures:
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- {failure}" for failure in failures)
    lines.append("")
    return "\n".join(lines)


def write_acceptance_reports(
    report: Mapping[str, Any], *, json_path: Path, markdown_path: Path
) -> None:
    _atomic_write_json(json_path, report)
    _atomic_write_text(markdown_path, build_markdown_report(report))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m dataset.acceptance",
        description="Fail-closed dataset operational qualification evidence verifier.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    snapshot = commands.add_parser(
        "snapshot", help="Capture deterministic durable pilot state for resume checks."
    )
    snapshot.add_argument("--output-dir", type=Path, required=True)
    snapshot.add_argument("--snapshot-file", type=Path, required=True)

    verify_parser = commands.add_parser(
        "verify", help="Verify all calibration, pilot, Drive, resume, and idempotence evidence."
    )
    verify_parser.add_argument("--weights-file", type=Path, required=True)
    verify_parser.add_argument(
        "--calibration-dir", type=Path, default=DEFAULT_CALIBRATION_DIR
    )
    verify_parser.add_argument("--output-dir", type=Path, required=True)
    verify_parser.add_argument("--run-id", required=True)
    verify_parser.add_argument("--interrupted-snapshot", type=Path, required=True)
    verify_parser.add_argument("--idempotence-baseline", type=Path, required=True)
    verify_parser.add_argument("--ops-dir", type=Path, default=DEFAULT_OPS_DIR)
    verify_parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    verify_parser.add_argument("--drive-smoke-log", type=Path, default=None)
    verify_parser.add_argument(
        "--expected-weights-sha256", default=APPROVED_WEIGHTS_SHA256
    )
    verify_parser.add_argument("--minimum-tokens", type=int, default=9_000_000)
    verify_parser.add_argument("--maximum-tokens", type=int, default=11_000_000)
    verify_parser.add_argument("--full-scan", action="store_true")
    verify_parser.add_argument("--report-json", type=Path, default=None)
    verify_parser.add_argument("--report-md", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "snapshot":
        snapshot = write_pilot_snapshot(args.output_dir, args.snapshot_file)
        print(json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    if args.minimum_tokens <= 0 or args.maximum_tokens < args.minimum_tokens:
        raise SystemExit("token range must satisfy 0 < minimum <= maximum")
    drive_smoke_log = args.drive_smoke_log or args.ops_dir / "logs" / "drive-oauth-smoke.log"
    report = evaluate_acceptance(
        repo_root=args.repo_root,
        weights_path=args.weights_file,
        calibration_dir=args.calibration_dir,
        pilot_output_dir=args.output_dir,
        run_id=args.run_id,
        interrupted_snapshot_path=args.interrupted_snapshot,
        idempotence_baseline_path=args.idempotence_baseline,
        ops_dir=args.ops_dir,
        drive_smoke_log=drive_smoke_log,
        expected_weights_sha256=args.expected_weights_sha256,
        minimum_tokens=args.minimum_tokens,
        maximum_tokens=args.maximum_tokens,
        full_scan=bool(args.full_scan),
    )
    json_path = args.report_json or args.ops_dir / DEFAULT_REPORT_JSON_NAME
    markdown_path = args.report_md or args.ops_dir / DEFAULT_REPORT_MD_NAME
    write_acceptance_reports(report, json_path=json_path, markdown_path=markdown_path)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
