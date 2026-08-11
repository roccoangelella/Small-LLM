#!/usr/bin/env python3
"""Prepare the canonical 2B block-64 Modal corpus entirely from the VPS.

The workflow intentionally keeps Kaggle and Modal interaction on the operator VPS:
1. discover the authenticated user's existing Kaggle 2B dataset by its frozen slug;
2. download/unzip it into a fixed VPS cache only when a verified source is absent;
3. verify the exact schema-v2/run identity;
4. byte-preservingly reblock the corpus to 64 sequences per optimizer block;
5. optionally upload the derived directory to the canonical Modal data Volume.

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


def discover_kaggle_handle(kaggle_cli: str) -> str:
    explicit = os.environ.get("SMALL_LLM_2B_KAGGLE_DATASET_HANDLE", "").strip()
    if explicit:
        if not explicit.endswith("/" + SOURCE_DATASET_SLUG):
            raise RuntimeError(
                "SMALL_LLM_2B_KAGGLE_DATASET_HANDLE does not reference the frozen 2B dataset slug"
            )
        return explicit

    output = _capture(
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
    matches: set[str] = set()
    for row in csv.reader(io.StringIO(output)):
        for cell in row:
            value = cell.strip().strip('"')
            if value.endswith("/" + SOURCE_DATASET_SLUG):
                matches.add(value)
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one authenticated Kaggle dataset ending in /{SOURCE_DATASET_SLUG}; "
            f"found {sorted(matches)}. Set SMALL_LLM_2B_KAGGLE_DATASET_HANDLE=owner/{SOURCE_DATASET_SLUG} "
            "only if automatic discovery cannot resolve it."
        )
    return next(iter(matches))


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


def _upload_marker_matches(output_dir: Path) -> bool:
    if not UPLOAD_MARKER.is_file():
        return False
    try:
        marker = _read_object(UPLOAD_MARKER)
        return (
            marker.get("volume") == MODAL_VOLUME
            and marker.get("destination") == MODAL_DESTINATION
            and marker.get("manifest_sha256") == _sha256(output_dir / config.MANIFEST_FILENAME)
        )
    except Exception:
        return False


def upload_to_modal(modal_cli: str, output_dir: Path, *, force: bool) -> bool:
    if not force and _upload_marker_matches(output_dir):
        print("Modal upload marker matches the derived manifest; skipping duplicate upload.", flush=True)
        return False
    # --force makes a rerun safe after a partially completed prior upload. The
    # destination is frozen to this dataset identity and first-use training
    # verification still checks every shard against the manifest.
    _run_live(
        [modal_cli, "volume", "put", "--force", MODAL_VOLUME, str(output_dir), MODAL_DESTINATION]
    )
    UPLOAD_MARKER.parent.mkdir(parents=True, exist_ok=True)
    UPLOAD_MARKER.write_text(
        json.dumps(
            {
                "version": 1,
                "volume": MODAL_VOLUME,
                "destination": MODAL_DESTINATION,
                "manifest_sha256": _sha256(output_dir / config.MANIFEST_FILENAME),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return True


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Download the frozen 2B Kaggle corpus on the VPS, reblock it for Modal, and upload it."
    )
    result.add_argument("--no-upload", action="store_true", help="Prepare locally on the VPS but do not upload to Modal.")
    result.add_argument("--force-download", action="store_true", help="Redownload the Kaggle dataset even if a verified source is cached.")
    result.add_argument("--force-reblock", action="store_true", help="Rebuild the block-64 derivative even if it verifies.")
    result.add_argument("--force-upload", action="store_true", help="Upload even when the local upload marker matches.")
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
        if modal_cli is not None:
            uploaded = upload_to_modal(modal_cli, OUTPUT_DIR, force=bool(args.force_upload))

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
            "next_command": "modal run --detach modal/launch.py --model 100M --tokens 2B --gpu H100",
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception as error:  # noqa: BLE001 - operator-facing command boundary
        print(f"Modal dataset preparation error: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
