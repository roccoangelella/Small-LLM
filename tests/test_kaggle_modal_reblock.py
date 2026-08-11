from __future__ import annotations

import json
from pathlib import Path

from kaggle.reblock_for_modal import (
    DEFAULT_OUTPUT_DIR,
    SOURCE_DATASET_SLUG,
    SOURCE_RUN_ID,
    find_attached_source,
)


def _write_attached_dataset(root: Path, *, run_id: str = SOURCE_RUN_ID) -> Path:
    dataset = root / SOURCE_DATASET_SLUG
    (dataset / "train").mkdir(parents=True)
    (dataset / "validation").mkdir(parents=True)
    manifest = {
        "schema_version": 2,
        "sequence_format": "context_plus_one",
        "context_length": 2048,
        "stored_tokens_per_sequence": 2049,
        "sequences_per_block": 16,
        "production": {"run_id": run_id},
    }
    (dataset / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return dataset


def test_kaggle_wrapper_selects_known_2b_dataset(tmp_path: Path) -> None:
    expected = _write_attached_dataset(tmp_path)
    selected, inspected = find_attached_source(tmp_path)

    assert selected == expected.resolve()
    assert len(inspected) == 1
    assert inspected[0]["run_id"] == SOURCE_RUN_ID


def test_kaggle_wrapper_rejects_wrong_run_id(tmp_path: Path) -> None:
    _write_attached_dataset(tmp_path, run_id="wrong-dataset")

    try:
        find_attached_source(tmp_path)
    except RuntimeError as error:
        assert "found 0" in str(error)
    else:
        raise AssertionError("wrong dataset run ID must not be selected")


def test_kaggle_modal_output_path_is_fixed() -> None:
    assert DEFAULT_OUTPUT_DIR == Path("/kaggle/working/modal-2b-b64-dataset-001")
