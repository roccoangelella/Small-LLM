"""Regression guard: active data/training code must remain Google-Drive-free."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Legacy checkpoint/dataset schema names such as drive_manifest.json and
# drive_file_id are intentionally allowed. These markers identify executable
# Google client/auth/upload configuration, which ADR 0054 retired.
BANNED_MARKERS = (
    "GoogleDriveShardStore",
    "googleapiclient",
    "google_auth_oauthlib",
    "google.oauth2",
    "SMALL_LLM_GOOGLE_OAUTH_TOKEN",
    "SMALL_LLM_DRIVE_FOLDER_ID",
)


def _active_files() -> list[Path]:
    files: list[Path] = []
    for directory in (ROOT / "dataset", ROOT / "modal", ROOT / "kaggle"):
        files.extend(path for path in directory.rglob("*.py") if path.is_file())
        files.extend(path for path in directory.rglob("*.txt") if path.is_file())
        files.extend(path for path in directory.rglob("*.example") if path.is_file())
    files.append(ROOT / ".env.example")
    return sorted(set(files))


def test_google_drive_backend_is_absent_from_active_code_and_config() -> None:
    assert not (ROOT / "dataset" / "drive_auth.py").exists()
    violations: list[str] = []
    for path in _active_files():
        text = path.read_text(encoding="utf-8")
        for marker in BANNED_MARKERS:
            if marker in text:
                violations.append(f"{path.relative_to(ROOT)}: {marker}")
    assert violations == []


def test_remote_dependency_lists_are_hf_only() -> None:
    dataset_requirements = (ROOT / "dataset" / "requirements-remote.txt").read_text(
        encoding="utf-8"
    )
    kaggle_requirements = (ROOT / "kaggle" / "requirements-100m-publish.txt").read_text(
        encoding="utf-8"
    )
    assert dataset_requirements.strip() == "huggingface-hub>=1.5,<2"
    assert "huggingface-hub>=1.5,<2" in kaggle_requirements
    assert "google-" not in dataset_requirements.lower()
    assert "google-" not in kaggle_requirements.lower()
