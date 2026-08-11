from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "small_llm_modal_prepare_dataset",
    ROOT / "modal" / "prepare_dataset.py",
)
assert SPEC is not None and SPEC.loader is not None
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


def _write_source(root: Path, *, run_id: str = prepare.SOURCE_RUN_ID) -> Path:
    dataset = root / "payload"
    (dataset / "train").mkdir(parents=True)
    (dataset / "validation").mkdir(parents=True)
    manifest = {
        "schema_version": 2,
        "sequence_format": "context_plus_one",
        "context_length": 2048,
        "stored_tokens_per_sequence": 2049,
        "sequences_per_block": 16,
        "target_shard_bytes": 8 * 1024 * 1024,
        "production": {"run_id": run_id},
    }
    (dataset / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return dataset


def test_vps_paths_are_fixed_outside_repository() -> None:
    assert prepare.VPS_DATA_ROOT == Path.home() / "small-llm-data"
    assert prepare.KAGGLE_DOWNLOAD_DIR == Path.home() / "small-llm-data" / "kaggle" / prepare.SOURCE_DATASET_SLUG
    assert prepare.OUTPUT_DIR == Path.home() / "small-llm-data" / prepare.TARGET_RUN_ID
    assert prepare.MODAL_DESTINATION == "/datasets/modal-2b-b64-dataset-001"


def test_handle_discovery_uses_authenticated_mine_search(monkeypatch) -> None:
    monkeypatch.delenv("SMALL_LLM_2B_KAGGLE_DATASET_HANDLE", raising=False)
    expected = f"owner/{prepare.SOURCE_DATASET_SLUG}"
    seen: list[list[str]] = []

    def fake_capture(command):
        seen.append(list(command))
        return f"ref,title\n{expected},Small LLM 2B\n"

    prepare._capture = fake_capture
    assert prepare.discover_kaggle_handle("kaggle") == expected
    assert seen == [[
        "kaggle", "datasets", "list", "--mine", "--search",
        prepare.SOURCE_DATASET_SLUG, "--csv",
    ]]


def test_explicit_handle_must_match_frozen_slug(monkeypatch) -> None:
    good = f"owner/{prepare.SOURCE_DATASET_SLUG}"
    monkeypatch.setenv("SMALL_LLM_2B_KAGGLE_DATASET_HANDLE", good)
    assert prepare.discover_kaggle_handle("kaggle") == good

    monkeypatch.setenv("SMALL_LLM_2B_KAGGLE_DATASET_HANDLE", "owner/wrong")
    try:
        prepare.discover_kaggle_handle("kaggle")
    except RuntimeError as error:
        assert "frozen 2B dataset slug" in str(error)
    else:
        raise AssertionError("wrong explicit Kaggle handle must be rejected")


def test_find_source_requires_exact_run_identity(tmp_path: Path) -> None:
    expected = _write_source(tmp_path)
    assert prepare.find_source(tmp_path) == expected.resolve()

    manifest = json.loads((expected / "manifest.json").read_text(encoding="utf-8"))
    manifest["production"]["run_id"] = "wrong-run"
    (expected / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert prepare.find_source(tmp_path) is None
