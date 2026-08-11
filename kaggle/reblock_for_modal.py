#!/usr/bin/env python3
"""Reblock the attached 2B Kaggle dataset into the canonical Modal block-64 corpus."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from dataset.reblock import reblock_dataset  # noqa: E402

SOURCE_DATASET_SLUG = "small-llm-20m-2b-dataset-001"
SOURCE_RUN_ID = "20m-2b-dataset-001"
DEFAULT_INPUT_ROOT = Path("/kaggle/input")
DEFAULT_OUTPUT_DIR = Path("/kaggle/working/modal-2b-b64-dataset-001")
MODAL_VOLUME_NAME = "small-llm-data"
MODAL_VOLUME_DESTINATION = "/datasets/modal-2b-b64-dataset-001"


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as error:
        raise RuntimeError(f"cannot read dataset manifest {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise RuntimeError(f"dataset manifest is not a JSON object: {path}")
    return dict(value)


def _matches_source(root: Path) -> tuple[bool, dict[str, object]]:
    manifest_path = root / "manifest.json"
    row: dict[str, object] = {
        "root": str(root),
        "manifest": manifest_path.is_file(),
        "train": (root / "train").is_dir(),
        "validation": (root / "validation").is_dir(),
    }
    if not all(bool(row[key]) for key in ("manifest", "train", "validation")):
        return False, row
    manifest = _read_manifest(manifest_path)
    production = manifest.get("production")
    run_id = production.get("run_id") if isinstance(production, Mapping) else None
    row["run_id"] = run_id
    matched = (
        manifest.get("schema_version") == 2
        and manifest.get("sequence_format") == "context_plus_one"
        and manifest.get("context_length") == 2048
        and manifest.get("stored_tokens_per_sequence") == 2049
        and manifest.get("sequences_per_block") == 16
        and run_id == SOURCE_RUN_ID
    )
    return bool(matched), row


def find_attached_source(input_root: Path = DEFAULT_INPUT_ROOT) -> tuple[Path, list[dict[str, object]]]:
    input_root = input_root.resolve()
    preferred = input_root / SOURCE_DATASET_SLUG
    manifests: list[Path] = []
    if preferred.is_dir():
        manifests.extend(sorted(preferred.rglob("manifest.json")))
    if not manifests and input_root.is_dir():
        manifests.extend(sorted(input_root.rglob("manifest.json")))

    inspected: list[dict[str, object]] = []
    matches: list[Path] = []
    seen: set[Path] = set()
    for manifest_path in manifests:
        root = manifest_path.parent.resolve()
        if root in seen:
            continue
        seen.add(root)
        matched, row = _matches_source(root)
        inspected.append(row)
        if matched:
            matches.append(root)

    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one attached Kaggle dataset {SOURCE_DATASET_SLUG!r} "
            f"with run_id {SOURCE_RUN_ID!r}; found {len(matches)}\n"
            + json.dumps(inspected, indent=2, sort_keys=True)
        )
    return matches[0], inspected


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Derive the Modal block-64 2B corpus directly from the attached Kaggle dataset."
    )
    result.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    result.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    result.add_argument(
        "--replace-output",
        action="store_true",
        help="Delete an existing derived output directory before rebuilding it.",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    output_dir = args.output_dir.resolve()
    try:
        source_dir, inspected = find_attached_source(args.input_root)
        if output_dir.exists():
            if not args.replace_output:
                raise FileExistsError(
                    f"derived output already exists: {output_dir}; use --replace-output to rebuild"
                )
            shutil.rmtree(output_dir)
        result = reblock_dataset(source_dir, output_dir)
        payload = {
            **result,
            "kaggle_source_slug": SOURCE_DATASET_SLUG,
            "kaggle_source_dir": str(source_dir),
            "kaggle_output_dir": str(output_dir),
            "datasets_inspected": inspected,
            "next_command": (
                f"modal volume put {MODAL_VOLUME_NAME} {output_dir} "
                f"{MODAL_VOLUME_DESTINATION}"
            ),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except Exception as error:  # noqa: BLE001 - concise notebook-facing boundary
        print(f"Kaggle Modal reblock error: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
