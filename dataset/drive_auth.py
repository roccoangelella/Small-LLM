"""Google Drive OAuth authorization setup and credential management.

This module handles one-time interactive OAuth setup for personal Google Drive
accounts, validates authorized-user credentials, creates application-owned
storage folders, and runs preflight smoke tests.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Any

DRIVE_FILE_SCOPE = "https://www.googleapis.com/auth/drive.file"
DEFAULT_CLIENT_SECRETS_FILE = Path(".secrets/google-drive-oauth-client.json")
DEFAULT_TOKEN_FILE = Path(".secrets/google-drive-authorized-user.json")
DEFAULT_ENV_FILE = Path(".env")

logger = logging.getLogger(__name__)


def _atomic_write_text(target_path: Path, content: str) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_name(f"{target_path.name}.tmp.{uuid.uuid4().hex}")
    try:
        temp_path.write_text(content, encoding="utf-8")
        temp_path.replace(target_path)
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def update_env_file(env_path: Path, updates: dict[str, str]) -> None:
    lines: list[str] = []
    if env_path.is_file():
        lines = env_path.read_text(encoding="utf-8").splitlines()

    existing_keys: set[str] = set()
    new_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in line:
            key = line.split("=", 1)[0].strip()
            if key in updates:
                new_lines.append(f"{key}={updates[key]}")
                existing_keys.add(key)
                continue
        new_lines.append(line)

    for key, val in updates.items():
        if key not in existing_keys:
            new_lines.append(f"{key}={val}")

    content = "\n".join(new_lines) + "\n"
    _atomic_write_text(env_path, content)


def validate_client_secrets(path: str | Path) -> dict[str, Any]:
    client_path = Path(path)
    if not client_path.is_file():
        raise FileNotFoundError(f"OAuth client secrets file not found: {client_path}")
    try:
        data = json.loads(client_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"Malformed OAuth client secrets JSON in {client_path}: {error}") from error

    if not isinstance(data, dict):
        raise ValueError(f"OAuth client secrets file {client_path} must be a JSON object")

    if data.get("type") == "service_account":
        raise ValueError(
            f"File {client_path} is a service-account JSON. Personal Drive storage requires "
            "an installed-app OAuth client JSON file."
        )

    if "installed" not in data:
        client_type = "web" if "web" in data else "unknown"
        raise ValueError(
            f"Expected installed-app OAuth client JSON (containing 'installed' key), but {client_path} "
            f"has type {client_type!r}."
        )

    installed = data["installed"]
    if not isinstance(installed, dict) or not installed.get("client_id") or not installed.get("client_secret"):
        raise ValueError(f"Installed-app OAuth client JSON in {client_path} missing client_id or client_secret.")

    return data


def load_authorized_user_credentials(credentials_path: str | Path | None = None) -> Any:
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
    except ImportError as error:
        raise RuntimeError("Google Drive backend requires google-auth and google-api-python-client") from error

    resolved_path = (
        credentials_path
        or os.environ.get("SMALL_LLM_GOOGLE_OAUTH_TOKEN")
        or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    )
    if not resolved_path:
        raise RuntimeError(
            "No credentials path provided. Pass token path or set SMALL_LLM_GOOGLE_OAUTH_TOKEN environment variable."
        )

    path = Path(resolved_path)
    if not path.is_file():
        raise FileNotFoundError(f"Authorized user token file not found: {path}")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"Malformed credentials JSON in {path}: {error}") from error

    if not isinstance(data, dict):
        raise ValueError(f"Credentials in {path} must be a JSON object")

    if data.get("type") == "service_account":
        raise ValueError(
            f"Service-account JSON in {path} is rejected. Personal Drive storage requires authorized-user OAuth credentials."
        )

    if "installed" in data or "web" in data:
        raise ValueError(
            f"OAuth client secrets file ({path}) provided where an authorized-user token is required. "
            "Run 'uv run python -m dataset.drive_auth setup' first to authorize."
        )

    if not data.get("refresh_token"):
        raise ValueError(f"Authorized-user token file ({path}) is missing a refresh_token.")

    try:
        creds = Credentials.from_authorized_user_info(data, scopes=[DRIVE_FILE_SCOPE])
    except Exception as error:
        raise ValueError(f"Failed to parse authorized-user OAuth token from {path}: {error}") from error

    if creds.expired or not creds.valid:
        try:
            creds.refresh(Request())
            # Save refreshed credentials back to disk atomically
            _atomic_write_text(path, creds.to_json())
        except Exception as error:
            raise RuntimeError(
                f"Authorized-user OAuth credentials in {path} are expired or invalid and cannot be refreshed: {error}"
            ) from error

    if not creds.valid:
        raise RuntimeError(f"Authorized-user OAuth credentials in {path} are invalid after refresh attempt.")

    return creds


def ensure_drive_folder_tree(service: Any) -> tuple[str, str]:
    # 1. Look for 'Small LLM Storage' in root
    query_root = (
        "name = 'Small LLM Storage' and 'root' in parents and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    )
    found_root = service.files().list(q=query_root, fields="files(id)").execute().get("files", [])
    if len(found_root) > 1:
        raise RuntimeError("Drive contains duplicate folders named 'Small LLM Storage' in My Drive root.")

    if found_root:
        root_folder_id = str(found_root[0]["id"])
    else:
        created_root = service.files().create(
            body={"name": "Small LLM Storage", "mimeType": "application/vnd.google-apps.folder", "parents": ["root"]},
            fields="id",
        ).execute()
        root_folder_id = str(created_root["id"])

    # 2. Look for 'dataset-shards' inside 'Small LLM Storage'
    escaped_parent = root_folder_id.replace("\\", "\\\\").replace("'", "\\'")
    query_child = (
        f"name = 'dataset-shards' and '{escaped_parent}' in parents and "
        "mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    )
    found_child = service.files().list(q=query_child, fields="files(id)").execute().get("files", [])
    if len(found_child) > 1:
        raise RuntimeError("Drive contains duplicate folders named 'dataset-shards' in 'Small LLM Storage'.")

    if found_child:
        shards_folder_id = str(found_child[0]["id"])
    else:
        created_child = service.files().create(
            body={"name": "dataset-shards", "mimeType": "application/vnd.google-apps.folder", "parents": [root_folder_id]},
            fields="id",
        ).execute()
        shards_folder_id = str(created_child["id"])

    return root_folder_id, shards_folder_id


def run_smoke_test(service: Any, folder_id: str) -> None:
    try:
        from googleapiclient.http import MediaIoBaseUpload
    except ImportError as error:
        raise RuntimeError("Google Drive backend requires google-api-python-client") from error

    payload = b"small-llm-smoke-test-" + uuid.uuid4().bytes
    expected_sha256 = hashlib.sha256(payload).hexdigest()
    expected_md5 = hashlib.md5(payload).hexdigest()
    temp_name = f".smoke_test_{uuid.uuid4().hex}.tmp"

    media = MediaIoBaseUpload(io.BytesIO(payload), mimetype="application/octet-stream", resumable=False)
    created = service.files().create(
        body={"name": temp_name, "parents": [folder_id]},
        media_body=media,
        fields="id,size,md5Checksum",
    ).execute()
    file_id = created["id"]

    try:
        # Verify metadata
        meta = service.files().get(file_id=file_id, fields="id,size,md5Checksum").execute()
        if int(meta.get("size", -1)) != len(payload):
            raise RuntimeError(f"Smoke test byte size mismatch: expected {len(payload)}, got {meta.get('size')}")
        if meta.get("md5Checksum") and meta["md5Checksum"] != expected_md5:
            raise RuntimeError(f"Smoke test MD5 checksum mismatch: expected {expected_md5}, got {meta.get('md5Checksum')}")

        # Download & check content
        downloaded = service.files().get_media(file_id=file_id).execute()
        if not isinstance(downloaded, bytes):
            downloaded = bytes(downloaded)
        downloaded_sha256 = hashlib.sha256(downloaded).hexdigest()
        if downloaded_sha256 != expected_sha256:
            raise RuntimeError(
                f"Smoke test downloaded content SHA-256 mismatch: expected {expected_sha256}, got {downloaded_sha256}"
            )
    finally:
        # Cleanup temp smoke file
        try:
            service.files().delete(file_id=file_id).execute()
        except Exception as cleanup_error:
            logger.warning("Failed to clean up temporary smoke test file %s: %s", file_id, cleanup_error)


def setup_drive_auth(
    client_secrets_path: Path = DEFAULT_CLIENT_SECRETS_FILE,
    token_file_path: Path = DEFAULT_TOKEN_FILE,
    env_file_path: Path = DEFAULT_ENV_FILE,
) -> None:
    try:
        from googleapiclient.discovery import build
    except ImportError as error:
        raise RuntimeError("Google Drive backend requires google-api-python-client") from error

    creds = None
    if token_file_path.is_file():
        try:
            creds = load_authorized_user_credentials(token_file_path)
            sys.stdout.write(f"Reusing valid authorized-user credentials from {token_file_path}\n")
        except Exception as error:
            sys.stdout.write(f"Existing token at {token_file_path} invalid or expired ({error}); initiating authorization...\n")

    if creds is None:
        validate_client_secrets(client_secrets_path)
        try:
            from google_auth_oauthlib.flow import InstalledAppFlow
        except ImportError as error:
            raise RuntimeError("Installed-app OAuth flow requires google-auth-oauthlib") from error

        flow = InstalledAppFlow.from_client_secrets_file(
            str(client_secrets_path),
            scopes=[DRIVE_FILE_SCOPE],
        )
        creds = flow.run_local_server(
            port=0,
            access_type="offline",
            prompt="consent",
        )
        if not getattr(creds, "refresh_token", None):
            raise RuntimeError("OAuth authorization flow failed to obtain a refresh token.")

        _atomic_write_text(token_file_path, creds.to_json())
        sys.stdout.write(f"Saved OAuth authorized-user token to {token_file_path}\n")

    service = build("drive", "v3", credentials=creds, cache_discovery=False)

    user_email = "unknown"
    try:
        about = service.about().get(fields="user(emailAddress)").execute()
        user_email = about.get("user", {}).get("emailAddress", "unknown")
    except Exception as error:
        logger.warning("Could not fetch user email profile: %s", error)

    root_id, shards_id = ensure_drive_folder_tree(service)

    update_env_file(
        env_file_path,
        {
            "SMALL_LLM_GOOGLE_OAUTH_TOKEN": str(token_file_path),
            "SMALL_LLM_DRIVE_FOLDER_ID": shards_id,
        },
    )

    run_smoke_test(service, shards_id)

    sys.stdout.write("\nGoogle Drive OAuth setup complete.\n")
    sys.stdout.write(f"Authenticated Account: {user_email}\n")
    sys.stdout.write(f"Drive Storage Folder ID: {shards_id}\n")
    sys.stdout.write(f"Folder Structure: Small LLM Storage ({root_id}) / dataset-shards ({shards_id})\n")
    sys.stdout.write(f"Token File: {token_file_path}\n")
    sys.stdout.write("Smoke test: PASSED (upload, metadata read, download hash verify, cleanup)\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m dataset.drive_auth",
        description="Google Drive OAuth setup and credentials verification CLI.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    setup_parser = subparsers.add_parser("setup", help="Authorize Google Drive access and set up storage folders.")
    setup_parser.add_argument(
        "--client-secrets",
        type=Path,
        default=DEFAULT_CLIENT_SECRETS_FILE,
        help="Path to installed-app OAuth client JSON file",
    )
    setup_parser.add_argument(
        "--token-file",
        type=Path,
        default=DEFAULT_TOKEN_FILE,
        help="Target path for authorized-user token JSON file",
    )
    setup_parser.add_argument(
        "--env-file",
        type=Path,
        default=DEFAULT_ENV_FILE,
        help="Target path for .env file to store configuration",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    if args.command == "setup":
        try:
            setup_drive_auth(
                client_secrets_path=args.client_secrets,
                token_file_path=args.token_file,
                env_file_path=args.env_file,
            )
            return 0
        except Exception as error:  # noqa: BLE001
            sys.stderr.write(f"drive_auth setup failed: {type(error).__name__}: {error}\n")
            return 1
    return 1


if __name__ == "__main__":
    sys.exit(main())
