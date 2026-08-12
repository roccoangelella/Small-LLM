#!/usr/bin/env python3
"""Prepare the canonical 2B block-64 Modal corpus entirely from the VPS.

The workflow intentionally keeps Kaggle and Modal interaction on the operator VPS:
1. resolve the already-published Kaggle 2B dataset by its frozen slug/recorded handle;
2. download/unzip it into a fixed VPS cache only when a verified source is absent;
3. verify the exact schema-v2/run identity;
4. byte-preservingly reblock the corpus to 64 sequences per optimizer block;
5. verify the derived directory in the active Modal workspace/environment, uploading only when needed.

No Kaggle notebook is involved.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from dataset import config  # noqa: E402
from dataset.qualification import derive_plan, get_profile  # noqa: E402
from dataset.reblock import reblock_dataset  # noqa: E402
from dataset.src.verify import verify  # noqa: E402

SOURCE_PROFILE = "20m-2b"
TARGET_PROFILE = "modal-2b-b64"
SOURCE_DATASET_SLUG = "small-llm-20m-2b-dataset-001"
SOURCE_RUN_ID = "20m-2b-dataset-001"
TARGET_RUN_ID = "modal-2b-b64-dataset-001"

VPS_DATA_ROOT = Path.home() / "small-llm-data"
KAGGLE_DOWNLOAD_DIR = VPS_DATA_ROOT / "kaggle" / SOURCE_DATASET_SLUG
OUTPUT_DIR = VPS_DATA_ROOT / TARGET_RUN_ID
UPLOAD_MARKER = VPS_DATA_ROOT / ".modal-2b-b64-upload.json"
KAGGLE_PUBLISH_STATE = Path("/data/small-llm/20m-2b-ops/kaggle-publish-state.json")
MAX_KAGGLE_LIST_PAGES = 100

MODAL_VOLUME = "small-llm-data"
MODAL_DESTINATION = f"/datasets/{TARGET_RUN_ID}"


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise RuntimeError(f"expected JSON object: {path}")
    return dict(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_cli(name: str) -> str:
    executable = shutil.which(name)
    if executable is None:
        raise RuntimeError(
            f"required CLI {name!r} is not installed in the active VPS environment; "
            "run: uv pip install kaggle 'modal>=1.1,<2'"
        )
    return executable


def _capture(command: Sequence[str]) -> str:
    process = subprocess.run(
        list(command),
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if process.returncode:
        detail = process.stderr.strip() or process.stdout.strip()
        raise RuntimeError(f"command failed ({process.returncode}): {' '.join(command)}\n{detail}")
    return process.stdout


def _run_live(command: Sequence[str]) -> None:
    print("$ " + " ".join(str(item) for item in command), flush=True)
    process = subprocess.run(list(command), cwd=REPO, check=False)
    if process.returncode:
        raise RuntimeError(f"command failed ({process.returncode}): {' '.join(command)}")


def _dataset_refs(output: str) -> set[str]:
    refs: set[str] = set()
    for row in csv.reader(io.StringIO(output)):
        for cell in row:
            value = cell.strip().strip('"')
            if value.endswith("/" + SOURCE_DATASET_SLUG):
                refs.add(value)
    return refs


def _recorded_publish_handle() -> str | None:
    if not KAGGLE_PUBLISH_STATE.is_file():
        return None
    try:
        handle = _read_object(KAGGLE_PUBLISH_STATE).get("handle")
    except Exception:
        return None
    if isinstance(handle, str) and handle.endswith("/" + SOURCE_DATASET_SLUG):
        return handle
    return None


def discover_kaggle_handle(kaggle_cli: str) -> str:
    """Resolve the exact owner/slug without relying on search ranking.

    Priority is explicit override, the durable publication state written when this
    exact dataset was uploaded from the VPS, KAGGLE_USERNAME, then authenticated
    API discovery. The API fallback first tries search and then walks owned pages
    because Kaggle search is title/relevance based rather than an exact slug lookup.
    """

    explicit = os.environ.get("SMALL_LLM_2B_KAGGLE_DATASET_HANDLE", "").strip()
    if explicit:
        if not explicit.endswith("/" + SOURCE_DATASET_SLUG):
            raise RuntimeError(
                "SMALL_LLM_2B_KAGGLE_DATASET_HANDLE does not reference the frozen 2B dataset slug"
            )
        return explicit

    recorded = _recorded_publish_handle()
    if recorded:
        print(f"Using Kaggle handle recorded by VPS publication state: {recorded}", flush=True)
        return recorded

    username = os.environ.get("KAGGLE_USERNAME", "").strip()
    if username:
        return f"{username}/{SOURCE_DATASET_SLUG}"

    searched = _capture(
        [
            kaggle_cli,
            "datasets",
            "list",
            "--mine",
            "--search",
            SOURCE_DATASET_SLUG,
            "--csv",
        ]
    )
    matches = _dataset_refs(searched)
    if len(matches) == 1:
        return next(iter(matches))
    if len(matches) > 1:
        raise RuntimeError(f"multiple owned Kaggle datasets match the frozen slug: {sorted(matches)}")

    seen_pages: set[str] = set()
    for page in range(1, MAX_KAGGLE_LIST_PAGES + 1):
        output = _capture(
            [
                kaggle_cli,
                "datasets",
                "list",
                "--mine",
                "--page",
                str(page),
                "--csv",
            ]
        )
        fingerprint = output.strip()
        if not fingerprint or fingerprint in seen_pages:
            break
        seen_pages.add(fingerprint)
        matches.update(_dataset_refs(output))
        if len(matches) == 1:
            return next(iter(matches))
        if len(matches) > 1:
            raise RuntimeError(f"multiple owned Kaggle datasets match the frozen slug: {sorted(matches)}")

    raise RuntimeError(
        f"authenticated Kaggle API could not resolve an owned dataset ending in /{SOURCE_DATASET_SLUG}. "
        f"Checked publication state {KAGGLE_PUBLISH_STATE}, KAGGLE_USERNAME, search, and up to "
        f"{MAX_KAGGLE_LIST_PAGES} owned-dataset pages. Set "
        f"SMALL_LLM_2B_KAGGLE_DATASET_HANDLE=owner/{SOURCE_DATASET_SLUG} only if the dataset is "
        "owned by a different account/organization than the current Kaggle token."
    )


def _profile_matches(root: Path, profile_key: str, run_id: str) -> bool:
    manifest_path = root / config.MANIFEST_FILENAME
    if not manifest_path.is_file() or not (root / "train").is_dir() or not (root / "validation").is_dir():
        return False
    try:
        manifest = _read_object(manifest_path)
        profile = get_profile(profile_key)
        production = manifest.get("production")
        if not isinstance(production, Mapping) or production.get("run_id") != run_id:
            return False
        expected = {
            "schema_version": 2,
            "sequence_format": "context_plus_one",
            "context_length": profile.context_length,
            "stored_tokens_per_sequence": profile.context_length + 1,
            "sequences_per_block": profile.sequences_per_block,
            "target_shard_bytes": profile.target_shard_bytes,
        }
        return all(manifest.get(key) == value for key, value in expected.items())
    except Exception:
        return False


def find_source(download_dir: Path) -> Path | None:
    if not download_dir.is_dir():
        return None
    roots = sorted({path.parent for path in download_dir.rglob(config.MANIFEST_FILENAME)})
    matches = [root for root in roots if _profile_matches(root, SOURCE_PROFILE, SOURCE_RUN_ID)]
    if not matches and _profile_matches(download_dir, SOURCE_PROFILE, SOURCE_RUN_ID):
        matches = [download_dir]
    if len(matches) > 1:
        raise RuntimeError(f"multiple verified 2B source trees found under {download_dir}: {matches}")
    return matches[0] if matches else None


def verify_source(root: Path) -> dict[str, Any]:
    report = verify(root, full_scan=False)
    if not report.passed:
        raise RuntimeError("downloaded Kaggle dataset failed verification: " + "; ".join(report.problems))
    manifest_path = root / config.MANIFEST_FILENAME
    manifest = _read_object(manifest_path)
    plan = derive_plan(manifest, profile=SOURCE_PROFILE, manifest_path=manifest_path)
    return {
        "root": str(root),
        "manifest_sha256": _sha256(manifest_path),
        "train_target_tokens": int(plan["train"]["target_tokens"]),
        "validation_target_tokens": int(plan["validation"]["target_tokens"]),
    }


def target_status(output_dir: Path) -> dict[str, Any] | None:
    if not _profile_matches(output_dir, TARGET_PROFILE, TARGET_RUN_ID):
        return None
    try:
        report = verify(output_dir, full_scan=False)
        if not report.passed:
            return None
        manifest_path = output_dir / config.MANIFEST_FILENAME
        manifest = _read_object(manifest_path)
        plan = derive_plan(manifest, profile=TARGET_PROFILE, manifest_path=manifest_path)
        return {
            "root": str(output_dir),
            "manifest_sha256": _sha256(manifest_path),
            "train_blocks": int(plan["train"]["block_count"]),
            "validation_blocks": int(plan["validation"]["block_count"]),
            "train_target_tokens": int(plan["train"]["target_tokens"]),
        }
    except Exception:
        return None


def download_source(kaggle_cli: str, handle: str, download_dir: Path) -> Path:
    if download_dir.exists():
        shutil.rmtree(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    _run_live(
        [
            kaggle_cli,
            "datasets",
            "download",
            handle,
            "--path",
            str(download_dir),
            "--unzip",
        ]
    )
    source = find_source(download_dir)
    if source is None:
        raise RuntimeError(
            f"Kaggle download completed but no exact {SOURCE_RUN_ID} schema-v2 dataset was found under {download_dir}"
        )
    return source


def _modal_object_name(value: Any, *, kind: str) -> str:
    name = getattr(value, "name", None)
    if callable(name):
        name = name()
    if not isinstance(name, str) or not name:
        raise RuntimeError(f"Modal {kind} did not expose a usable name")
    return name


def _expected_remote_files(output_dir: Path) -> dict[str, int]:
    manifest_path = output_dir / config.MANIFEST_FILENAME
    manifest = _read_object(manifest_path)
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise RuntimeError("derived dataset manifest has no shard inventory")

    root = output_dir.resolve()
    expected = {config.MANIFEST_FILENAME: manifest_path.stat().st_size}
    for row in shards:
        if not isinstance(row, Mapping):
            raise RuntimeError("derived dataset manifest contains a non-object shard entry")
        filename = row.get("filename")
        byte_size = row.get("byte_size")
        if not isinstance(filename, str) or not filename or filename.startswith("/"):
            raise RuntimeError(f"invalid shard filename in derived manifest: {filename!r}")
        if not isinstance(byte_size, int) or byte_size <= 0:
            raise RuntimeError(f"invalid shard byte size in derived manifest: {filename!r}")
        local_path = (output_dir / filename).resolve()
        try:
            local_path.relative_to(root)
        except ValueError as error:
            raise RuntimeError(f"derived manifest shard escapes dataset root: {filename!r}") from error
        if not local_path.is_file():
            raise RuntimeError(f"derived manifest shard is missing locally: {filename}")
        actual_size = local_path.stat().st_size
        if actual_size != byte_size:
            raise RuntimeError(
                f"derived manifest shard size mismatch for {filename}: expected {byte_size}, got {actual_size}"
            )
        expected[filename] = byte_size
    return expected


def _read_modal_file_sha256(volume: Any, remote_path: str) -> str:
    digest = hashlib.sha256()
    for chunk in volume.read_file(remote_path):
        digest.update(chunk)
    return digest.hexdigest()


def modal_remote_status(output_dir: Path) -> dict[str, Any]:
    """Verify the canonical dataset against the actually authenticated Modal context.

    The SDK context honors the active profile as well as MODAL_TOKEN_ID /
    MODAL_TOKEN_SECRET and MODAL_ENVIRONMENT overrides. The Volume is created
    lazily in a new workspace/environment so an account switch needs no manual
    storage setup. Readiness is determined from remote state, never the VPS marker.
    """

    try:
        import modal as modal_sdk
        from modal.exception import NotFoundError
    except ImportError as error:
        raise RuntimeError(
            "Modal Python SDK is unavailable in the active VPS environment; "
            "run: uv pip install 'modal>=1.1,<2'"
        ) from error

    workspace = modal_sdk.Workspace.from_context()
    environment = modal_sdk.Environment.from_context()
    workspace.hydrate()
    environment.hydrate()
    workspace_name = _modal_object_name(workspace, kind="workspace")
    environment_name = _modal_object_name(environment, kind="environment")

    volume = modal_sdk.Volume.from_name(
        MODAL_VOLUME,
        environment_name=environment_name,
        create_if_missing=True,
    )
    volume.hydrate()

    expected = _expected_remote_files(output_dir)
    remote_root = MODAL_DESTINATION.strip("/")
    prefix = remote_root + "/"
    observed: dict[str, int] = {}
    try:
        entries = volume.iterdir(remote_root, recursive=True)
        for entry in entries:
            path = str(entry.path).lstrip("/")
            if not path.startswith(prefix):
                continue
            relative = path[len(prefix) :]
            if relative in expected:
                observed[relative] = int(entry.size)
    except (NotFoundError, FileNotFoundError):
        observed = {}

    missing = sorted(set(expected) - set(observed))
    size_mismatches = sorted(
        path for path, expected_size in expected.items() if path in observed and observed[path] != expected_size
    )
    expected_manifest_sha = _sha256(output_dir / config.MANIFEST_FILENAME)
    remote_manifest_sha: str | None = None
    manifest_name = config.MANIFEST_FILENAME
    if manifest_name in observed and observed[manifest_name] == expected[manifest_name]:
        try:
            remote_manifest_sha = _read_modal_file_sha256(
                volume, f"{remote_root}/{manifest_name}"
            )
        except (NotFoundError, FileNotFoundError):
            remote_manifest_sha = None

    verified = (
        not missing
        and not size_mismatches
        and remote_manifest_sha == expected_manifest_sha
    )
    return {
        "workspace": workspace_name,
        "environment": environment_name,
        "volume": MODAL_VOLUME,
        "destination": MODAL_DESTINATION,
        "verified": verified,
        "expected_files": len(expected),
        "matched_files": len(expected) - len(missing) - len(size_mismatches),
        "missing_files": missing[:20],
        "size_mismatches": size_mismatches[:20],
        "expected_manifest_sha256": expected_manifest_sha,
        "remote_manifest_sha256": remote_manifest_sha,
    }


def _write_upload_marker(output_dir: Path, remote: Mapping[str, Any]) -> None:
    UPLOAD_MARKER.parent.mkdir(parents=True, exist_ok=True)
    UPLOAD_MARKER.write_text(
        json.dumps(
            {
                "version": 2,
                "workspace": remote.get("workspace"),
                "environment": remote.get("environment"),
                "volume": MODAL_VOLUME,
                "destination": MODAL_DESTINATION,
                "manifest_sha256": _sha256(output_dir / config.MANIFEST_FILENAME),
                "remote_verified": bool(remote.get("verified")),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def upload_to_modal(
    modal_cli: str, output_dir: Path, *, force: bool
) -> tuple[bool, dict[str, Any]]:
    before = modal_remote_status(output_dir)
    identity = f"workspace={before['workspace']} environment={before['environment']}"
    if before["verified"] and not force:
        print(
            f"Current Modal {identity} already has the verified dataset; skipping upload.",
            flush=True,
        )
        _write_upload_marker(output_dir, before)
        return False, before

    if before["verified"]:
        print(f"Current Modal {identity} already verifies; --force-upload will re-upload it.", flush=True)
    else:
        print(
            f"Current Modal {identity} does not yet verify "
            f"({before['matched_files']}/{before['expected_files']} expected files match); uploading.",
            flush=True,
        )

    _run_live(
        [modal_cli, "volume", "put", "--force", MODAL_VOLUME, str(output_dir), MODAL_DESTINATION]
    )
    after = modal_remote_status(output_dir)
    if not after["verified"]:
        raise RuntimeError(
            "Modal upload command returned success but remote verification failed in "
            f"workspace={after['workspace']} environment={after['environment']}: "
            f"matched {after['matched_files']}/{after['expected_files']} expected files; "
            f"missing={after['missing_files']}; size_mismatches={after['size_mismatches']}; "
            f"remote_manifest_sha256={after['remote_manifest_sha256']}"
        )
    _write_upload_marker(output_dir, after)
    print(
        f"Verified Modal dataset after upload in workspace={after['workspace']} "
        f"environment={after['environment']} ({after['expected_files']} files).",
        flush=True,
    )
    return True, after


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Download the frozen 2B Kaggle corpus on the VPS, reblock it for Modal, and upload it."
    )
    result.add_argument("--no-upload", action="store_true", help="Prepare locally on the VPS but do not upload to Modal.")
    result.add_argument("--force-download", action="store_true", help="Redownload the Kaggle dataset even if a verified source is cached.")
    result.add_argument("--force-reblock", action="store_true", help="Rebuild the block-64 derivative even if it verifies.")
    result.add_argument("--force-upload", action="store_true", help="Re-upload even when the active Modal workspace already verifies.")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        kaggle_cli = _require_cli("kaggle")
        modal_cli = None if args.no_upload else _require_cli("modal")
        handle = discover_kaggle_handle(kaggle_cli)

        existing_target = None if args.force_reblock else target_status(OUTPUT_DIR)
        downloaded = False
        reblocked = False
        source_info: dict[str, Any] | None = None

        if existing_target is None:
            source = None if args.force_download else find_source(KAGGLE_DOWNLOAD_DIR)
            if source is None:
                source = download_source(kaggle_cli, handle, KAGGLE_DOWNLOAD_DIR)
                downloaded = True
            source_info = verify_source(source)

            if OUTPUT_DIR.exists():
                shutil.rmtree(OUTPUT_DIR)
            OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)
            reblock_dataset(source, OUTPUT_DIR, target_profile_key=TARGET_PROFILE)
            reblocked = True
            existing_target = target_status(OUTPUT_DIR)
            if existing_target is None:
                raise RuntimeError("block-64 derivative failed post-reblock verification")

        uploaded = False
        modal_remote: dict[str, Any] | None = None
        if modal_cli is not None:
            uploaded, modal_remote = upload_to_modal(
                modal_cli, OUTPUT_DIR, force=bool(args.force_upload)
            )

        result = {
            "status": "ready",
            "kaggle_dataset": handle,
            "downloaded_this_run": downloaded,
            "reblocked_this_run": reblocked,
            "uploaded_this_run": uploaded,
            "vps_download_dir": str(KAGGLE_DOWNLOAD_DIR),
            "vps_output_dir": str(OUTPUT_DIR),
            "source": source_info,
            "target": existing_target,
            "modal_volume": None if args.no_upload else MODAL_VOLUME,
            "modal_destination": None if args.no_upload else MODAL_DESTINATION,
            "modal_remote": modal_remote,
            "next_command": "modal run --detach modal/launch.py --model 100M --tokens 2B --gpu H100",
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception as error:  # noqa: BLE001 - operator-facing command boundary
        print(f"Modal dataset preparation error: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
