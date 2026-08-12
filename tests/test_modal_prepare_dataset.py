from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

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


def _clear_handle_env(monkeypatch) -> None:
    monkeypatch.delenv("SMALL_LLM_2B_KAGGLE_DATASET_HANDLE", raising=False)
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)


def test_vps_paths_are_fixed_outside_repository() -> None:
    assert prepare.VPS_DATA_ROOT == Path.home() / "small-llm-data"
    assert prepare.KAGGLE_DOWNLOAD_DIR == Path.home() / "small-llm-data" / "kaggle" / prepare.SOURCE_DATASET_SLUG
    assert prepare.OUTPUT_DIR == Path.home() / "small-llm-data" / prepare.TARGET_RUN_ID
    assert prepare.KAGGLE_PUBLISH_STATE == Path("/data/small-llm/20m-2b-ops/kaggle-publish-state.json")
    assert prepare.MODAL_DESTINATION == "/datasets/modal-2b-b64-dataset-001"


def test_handle_discovery_prefers_recorded_publish_state(tmp_path: Path, monkeypatch) -> None:
    _clear_handle_env(monkeypatch)
    state = tmp_path / "kaggle-publish-state.json"
    expected = f"owner/{prepare.SOURCE_DATASET_SLUG}"
    state.write_text(json.dumps({"handle": expected}), encoding="utf-8")
    monkeypatch.setattr(prepare, "KAGGLE_PUBLISH_STATE", state)

    def fail_capture(command):
        raise AssertionError(f"Kaggle API should not be queried: {command}")

    monkeypatch.setattr(prepare, "_capture", fail_capture)
    assert prepare.discover_kaggle_handle("kaggle") == expected


def test_handle_discovery_uses_kaggle_username_before_api(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("SMALL_LLM_2B_KAGGLE_DATASET_HANDLE", raising=False)
    monkeypatch.setenv("KAGGLE_USERNAME", "owner")
    monkeypatch.setattr(prepare, "KAGGLE_PUBLISH_STATE", tmp_path / "missing.json")

    def fail_capture(command):
        raise AssertionError(f"Kaggle API should not be queried: {command}")

    monkeypatch.setattr(prepare, "_capture", fail_capture)
    assert prepare.discover_kaggle_handle("kaggle") == f"owner/{prepare.SOURCE_DATASET_SLUG}"


def test_handle_discovery_uses_authenticated_mine_search(tmp_path: Path, monkeypatch) -> None:
    _clear_handle_env(monkeypatch)
    monkeypatch.setattr(prepare, "KAGGLE_PUBLISH_STATE", tmp_path / "missing.json")
    expected = f"owner/{prepare.SOURCE_DATASET_SLUG}"
    seen: list[list[str]] = []

    def fake_capture(command):
        seen.append(list(command))
        return f"ref,title\n{expected},Small LLM 2B\n"

    monkeypatch.setattr(prepare, "_capture", fake_capture)
    assert prepare.discover_kaggle_handle("kaggle") == expected
    assert seen == [[
        "kaggle", "datasets", "list", "--mine", "--search",
        prepare.SOURCE_DATASET_SLUG, "--csv",
    ]]


def test_handle_discovery_falls_back_to_owned_pages(tmp_path: Path, monkeypatch) -> None:
    _clear_handle_env(monkeypatch)
    monkeypatch.setattr(prepare, "KAGGLE_PUBLISH_STATE", tmp_path / "missing.json")
    expected = f"owner/{prepare.SOURCE_DATASET_SLUG}"
    seen: list[list[str]] = []

    def fake_capture(command):
        command = list(command)
        seen.append(command)
        if "--search" in command:
            return "ref,title\n"
        page = command[command.index("--page") + 1]
        if page == "1":
            return "ref,title\nowner/other-dataset,Other\n"
        if page == "2":
            return f"ref,title\n{expected},Small LLM 2B\n"
        return "ref,title\n"

    monkeypatch.setattr(prepare, "_capture", fake_capture)
    assert prepare.discover_kaggle_handle("kaggle") == expected
    assert seen[-1][-2:] == ["2", "--csv"]


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


def _write_upload_target(root: Path) -> Path:
    dataset = root / "target"
    (dataset / "train").mkdir(parents=True)
    (dataset / "validation").mkdir(parents=True)
    train = dataset / "train" / "train-000000.bin"
    validation = dataset / "validation" / "validation-000000.bin"
    train.write_bytes(b"train-bytes")
    validation.write_bytes(b"validation-bytes")
    manifest = {
        "shards": [
            {"filename": "train/train-000000.bin", "byte_size": train.stat().st_size},
            {"filename": "validation/validation-000000.bin", "byte_size": validation.stat().st_size},
        ]
    }
    (dataset / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return dataset


def _remote_status(*, verified: bool, workspace: str = "new-workspace") -> dict[str, object]:
    return {
        "workspace": workspace,
        "environment": "main",
        "volume": prepare.MODAL_VOLUME,
        "destination": prepare.MODAL_DESTINATION,
        "verified": verified,
        "expected_files": 3,
        "matched_files": 3 if verified else 0,
        "missing_files": [] if verified else ["manifest.json"],
        "size_mismatches": [],
        "expected_manifest_sha256": "expected",
        "remote_manifest_sha256": "expected" if verified else None,
    }


def test_expected_remote_files_comes_from_manifest_and_exact_local_sizes(tmp_path: Path) -> None:
    target = _write_upload_target(tmp_path)
    expected = prepare._expected_remote_files(target)
    assert expected == {
        "manifest.json": (target / "manifest.json").stat().st_size,
        "train/train-000000.bin": len(b"train-bytes"),
        "validation/validation-000000.bin": len(b"validation-bytes"),
    }


def test_modal_remote_status_uses_authenticated_workspace_and_exact_remote_inventory(
    tmp_path: Path, monkeypatch
) -> None:
    target = _write_upload_target(tmp_path)
    manifest_bytes = (target / "manifest.json").read_bytes()
    expected = prepare._expected_remote_files(target)

    class Named:
        def __init__(self, name: str):
            self.name = name

        def hydrate(self):
            return self

    class Entry:
        def __init__(self, path: str, size: int):
            self.path = path
            self.size = size

    class FakeVolume(Named):
        def iterdir(self, path: str, *, recursive: bool):
            assert path == prepare.MODAL_DESTINATION.strip("/")
            assert recursive is True
            base = prepare.MODAL_DESTINATION.strip("/")
            return [Entry(f"{base}/{relative}", size) for relative, size in expected.items()]

        def read_file(self, path: str):
            assert path == f"{prepare.MODAL_DESTINATION.strip('/')}/manifest.json"
            return [manifest_bytes]

    volume = FakeVolume(prepare.MODAL_VOLUME)

    class VolumeFactory:
        @staticmethod
        def from_name(name: str, *, environment_name: str, create_if_missing: bool):
            assert name == prepare.MODAL_VOLUME
            assert environment_name == "main"
            assert create_if_missing is True
            return volume

    fake_modal = types.ModuleType("modal")
    fake_modal.Workspace = types.SimpleNamespace(from_context=lambda: Named("workspace-b"))
    fake_modal.Environment = types.SimpleNamespace(from_context=lambda: Named("main"))
    fake_modal.Volume = VolumeFactory
    fake_exception = types.ModuleType("modal.exception")
    fake_exception.NotFoundError = type("NotFoundError", (Exception,), {})
    monkeypatch.setitem(sys.modules, "modal", fake_modal)
    monkeypatch.setitem(sys.modules, "modal.exception", fake_exception)

    status = prepare.modal_remote_status(target)
    assert status["verified"] is True
    assert status["workspace"] == "workspace-b"
    assert status["environment"] == "main"
    assert status["matched_files"] == len(expected)
    assert status["remote_manifest_sha256"] == prepare._sha256(target / "manifest.json")


def test_matching_local_marker_does_not_skip_new_workspace_upload(tmp_path: Path, monkeypatch) -> None:
    target = _write_upload_target(tmp_path)
    marker = tmp_path / "upload-marker.json"
    marker.write_text(
        json.dumps(
            {
                "version": 1,
                "volume": prepare.MODAL_VOLUME,
                "destination": prepare.MODAL_DESTINATION,
                "manifest_sha256": prepare._sha256(target / "manifest.json"),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(prepare, "UPLOAD_MARKER", marker)
    statuses = iter([_remote_status(verified=False), _remote_status(verified=True)])
    monkeypatch.setattr(prepare, "modal_remote_status", lambda output_dir: next(statuses))
    commands: list[list[str]] = []
    monkeypatch.setattr(prepare, "_run_live", lambda command: commands.append(list(command)))

    uploaded, remote = prepare.upload_to_modal("modal", target, force=False)

    assert uploaded is True
    assert remote["verified"] is True
    assert commands == [[
        "modal", "volume", "put", "--force", prepare.MODAL_VOLUME, str(target), prepare.MODAL_DESTINATION
    ]]
    written = json.loads(marker.read_text(encoding="utf-8"))
    assert written["version"] == 2
    assert written["workspace"] == "new-workspace"
    assert written["remote_verified"] is True


def test_verified_current_workspace_skips_upload_without_force(tmp_path: Path, monkeypatch) -> None:
    target = _write_upload_target(tmp_path)
    marker = tmp_path / "upload-marker.json"
    monkeypatch.setattr(prepare, "UPLOAD_MARKER", marker)
    monkeypatch.setattr(prepare, "modal_remote_status", lambda output_dir: _remote_status(verified=True))
    monkeypatch.setattr(
        prepare,
        "_run_live",
        lambda command: (_ for _ in ()).throw(AssertionError(f"upload should be skipped: {command}")),
    )

    uploaded, remote = prepare.upload_to_modal("modal", target, force=False)

    assert uploaded is False
    assert remote["verified"] is True
    assert json.loads(marker.read_text(encoding="utf-8"))["workspace"] == "new-workspace"


def test_force_upload_reuploads_even_when_current_workspace_verifies(tmp_path: Path, monkeypatch) -> None:
    target = _write_upload_target(tmp_path)
    monkeypatch.setattr(prepare, "UPLOAD_MARKER", tmp_path / "upload-marker.json")
    statuses = iter([_remote_status(verified=True), _remote_status(verified=True)])
    monkeypatch.setattr(prepare, "modal_remote_status", lambda output_dir: next(statuses))
    commands: list[list[str]] = []
    monkeypatch.setattr(prepare, "_run_live", lambda command: commands.append(list(command)))

    uploaded, _ = prepare.upload_to_modal("modal", target, force=True)

    assert uploaded is True
    assert len(commands) == 1


def test_upload_success_message_is_not_trusted_without_remote_verification(tmp_path: Path, monkeypatch) -> None:
    target = _write_upload_target(tmp_path)
    marker = tmp_path / "upload-marker.json"
    monkeypatch.setattr(prepare, "UPLOAD_MARKER", marker)
    statuses = iter([_remote_status(verified=False), _remote_status(verified=False)])
    monkeypatch.setattr(prepare, "modal_remote_status", lambda output_dir: next(statuses))
    monkeypatch.setattr(prepare, "_run_live", lambda command: None)

    with pytest.raises(RuntimeError, match="remote verification failed"):
        prepare.upload_to_modal("modal", target, force=False)
    assert not marker.exists()
