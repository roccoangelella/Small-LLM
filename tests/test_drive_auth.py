"""Offline unit tests for dataset.drive_auth and Google Drive OAuth authorization."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from dataset.drive_auth import (
    ensure_drive_folder_tree,
    load_authorized_user_credentials,
    run_smoke_test,
    setup_drive_auth,
    update_env_file,
    validate_client_secrets,
)


class DriveAuthValidationTest(unittest.TestCase):
    def test_validate_client_secrets_valid_installed_app(self) -> None:
        valid_installed = {
            "installed": {
                "client_id": "test-client-id.apps.googleusercontent.com",
                "project_id": "test-project",
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "client_secret": "test-client-secret",
            }
        }
        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            json.dump(valid_installed, tmp)
            tmp_path = Path(tmp.name)

        try:
            res = validate_client_secrets(tmp_path)
            self.assertEqual(res["installed"]["client_id"], "test-client-id.apps.googleusercontent.com")
        finally:
            tmp_path.unlink()

    def test_validate_client_secrets_rejects_service_account(self) -> None:
        service_account_json = {
            "type": "service_account",
            "project_id": "test-project",
            "private_key_id": "123",
            "private_key": "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n",
            "client_email": "sa@test-project.iam.gserviceaccount.com",
        }
        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            json.dump(service_account_json, tmp)
            tmp_path = Path(tmp.name)

        try:
            with self.assertRaises(ValueError) as ctx:
                validate_client_secrets(tmp_path)
            self.assertIn("service-account JSON", str(ctx.exception))
        finally:
            tmp_path.unlink()

    def test_validate_client_secrets_rejects_web_app(self) -> None:
        web_json = {
            "web": {
                "client_id": "test-web-id.apps.googleusercontent.com",
                "client_secret": "secret",
            }
        }
        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            json.dump(web_json, tmp)
            tmp_path = Path(tmp.name)

        try:
            with self.assertRaises(ValueError) as ctx:
                validate_client_secrets(tmp_path)
            self.assertIn("Expected installed-app OAuth client JSON", str(ctx.exception))
        finally:
            tmp_path.unlink()

    def test_validate_client_secrets_rejects_malformed_json(self) -> None:
        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            tmp.write("{ invalid json ")
            tmp_path = Path(tmp.name)

        try:
            with self.assertRaises(ValueError) as ctx:
                validate_client_secrets(tmp_path)
            self.assertIn("Malformed OAuth client secrets JSON", str(ctx.exception))
        finally:
            tmp_path.unlink()


class AuthorizedUserCredentialsTest(unittest.TestCase):
    def test_load_authorized_user_credentials_valid(self) -> None:
        valid_token = {
            "client_id": "test-client-id",
            "client_secret": "test-client-secret",
            "refresh_token": "test-refresh-token",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            json.dump(valid_token, tmp)
            tmp_path = Path(tmp.name)

        try:
            mock_creds = MagicMock()
            mock_creds.expired = False
            mock_creds.valid = True

            with patch("google.oauth2.credentials.Credentials.from_authorized_user_info", return_value=mock_creds):
                creds = load_authorized_user_credentials(tmp_path)
                self.assertEqual(creds, mock_creds)
        finally:
            tmp_path.unlink()

    def test_load_authorized_user_credentials_refreshes_expired_token(self) -> None:
        valid_token = {
            "client_id": "test-client-id",
            "client_secret": "test-client-secret",
            "refresh_token": "test-refresh-token",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            json.dump(valid_token, tmp)
            tmp_path = Path(tmp.name)

        try:
            mock_creds = MagicMock()
            mock_creds.expired = True
            mock_creds.valid = True
            mock_creds.to_json.return_value = json.dumps(valid_token)

            with patch("google.oauth2.credentials.Credentials.from_authorized_user_info", return_value=mock_creds), \
                 patch("google.auth.transport.requests.Request") as mock_req:
                creds = load_authorized_user_credentials(tmp_path)
                mock_creds.refresh.assert_called_once()
                self.assertEqual(creds, mock_creds)
        finally:
            tmp_path.unlink()

    def test_load_authorized_user_credentials_rejects_missing_refresh_token(self) -> None:
        no_refresh_token = {
            "client_id": "test-client-id",
            "client_secret": "test-client-secret",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            json.dump(no_refresh_token, tmp)
            tmp_path = Path(tmp.name)

        try:
            with self.assertRaises(ValueError) as ctx:
                load_authorized_user_credentials(tmp_path)
            self.assertIn("missing a refresh_token", str(ctx.exception))
        finally:
            tmp_path.unlink()

    def test_load_authorized_user_credentials_rejects_service_account(self) -> None:
        sa_json = {"type": "service_account"}
        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            json.dump(sa_json, tmp)
            tmp_path = Path(tmp.name)

        try:
            with self.assertRaises(ValueError) as ctx:
                load_authorized_user_credentials(tmp_path)
            self.assertIn("Service-account JSON in", str(ctx.exception))
        finally:
            tmp_path.unlink()

    def test_load_authorized_user_credentials_rejects_client_secrets_json(self) -> None:
        client_secrets_json = {"installed": {"client_id": "foo"}}
        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            json.dump(client_secrets_json, tmp)
            tmp_path = Path(tmp.name)

        try:
            with self.assertRaises(ValueError) as ctx:
                load_authorized_user_credentials(tmp_path)
            self.assertIn("OAuth client secrets file", str(ctx.exception))
        finally:
            tmp_path.unlink()

    def test_load_authorized_user_credentials_rejects_unrecoverable_expired(self) -> None:
        valid_token = {
            "client_id": "test-client-id",
            "client_secret": "test-client-secret",
            "refresh_token": "test-refresh-token",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            json.dump(valid_token, tmp)
            tmp_path = Path(tmp.name)

        try:
            mock_creds = MagicMock()
            mock_creds.expired = True
            mock_creds.valid = False
            mock_creds.refresh.side_effect = Exception("Refresh revoked")

            with patch("google.oauth2.credentials.Credentials.from_authorized_user_info", return_value=mock_creds):
                with self.assertRaises(RuntimeError) as ctx:
                    load_authorized_user_credentials(tmp_path)
                self.assertIn("expired or invalid and cannot be refreshed", str(ctx.exception))
        finally:
            tmp_path.unlink()


class DriveFolderTreeTest(unittest.TestCase):
    def test_ensure_drive_folder_tree_creates_new(self) -> None:
        mock_service = MagicMock()
        # First list (root): empty
        # Second list (child): empty
        mock_service.files().list().execute.side_effect = [
            {"files": []},
            {"files": []},
        ]
        mock_service.files().create().execute.side_effect = [
            {"id": "root-folder-id-123"},
            {"id": "shards-folder-id-456"},
        ]

        root_id, shards_id = ensure_drive_folder_tree(mock_service)
        self.assertEqual(root_id, "root-folder-id-123")
        self.assertEqual(shards_id, "shards-folder-id-456")

    def test_ensure_drive_folder_tree_reuses_existing(self) -> None:
        mock_service = MagicMock()
        mock_service.files().list().execute.side_effect = [
            {"files": [{"id": "existing-root-id"}]},
            {"files": [{"id": "existing-shards-id"}]},
        ]

        root_id, shards_id = ensure_drive_folder_tree(mock_service)
        self.assertEqual(root_id, "existing-root-id")
        self.assertEqual(shards_id, "existing-shards-id")
        mock_service.files().create.assert_not_called()

    def test_ensure_drive_folder_tree_rejects_duplicate_folders(self) -> None:
        mock_service = MagicMock()
        mock_service.files().list().execute.return_value = {
            "files": [{"id": "dupe1"}, {"id": "dupe2"}]
        }

        with self.assertRaises(RuntimeError) as ctx:
            ensure_drive_folder_tree(mock_service)
        self.assertIn("duplicate folders", str(ctx.exception))


class DriveSmokeTestAndRedactionTest(unittest.TestCase):
    def test_run_smoke_test_verifies_upload_metadata_download_delete(self) -> None:
        mock_service = MagicMock()
        file_id = "smoke-file-123"

        mock_service.files().create().execute.return_value = {"id": file_id}

        fake_payload = None

        def create_mock(*args: object, **kwargs: object) -> MagicMock:
            nonlocal fake_payload
            media_body = kwargs.get("media_body")
            if media_body and hasattr(media_body, "_fd"):
                fake_payload = media_body._fd.getvalue()
            m = MagicMock()
            m.execute.return_value = {"id": file_id}
            return m

        mock_service.files().create.side_effect = create_mock

        def get_meta_mock(*args: object, **kwargs: object) -> MagicMock:
            import hashlib
            m = MagicMock()
            m.execute.return_value = {
                "id": file_id,
                "size": str(len(fake_payload or b"")),
                "md5Checksum": hashlib.md5(fake_payload or b"").hexdigest(),
            }
            return m

        mock_service.files().get.side_effect = get_meta_mock

        def get_media_mock(*args: object, **kwargs: object) -> MagicMock:
            m = MagicMock()
            m.execute.return_value = fake_payload
            return m

        mock_service.files().get_media.side_effect = get_media_mock

        run_smoke_test(mock_service, "shards-folder-id")

        mock_service.files().delete.assert_called_once_with(file_id=file_id)

    def test_setup_drive_auth_redacts_secrets_and_outputs_account_folder(self) -> None:
        secret_token = "secret-oauth-refresh-token-12345"
        valid_installed = {
            "installed": {
                "client_id": "client-id-abc.apps.googleusercontent.com",
                "client_secret": "super-secret-client-key",
            }
        }
        valid_user_token = {
            "client_id": "client-id-abc.apps.googleusercontent.com",
            "client_secret": "super-secret-client-key",
            "refresh_token": secret_token,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            client_path = tmp_path / "client.json"
            token_path = tmp_path / "token.json"
            env_path = tmp_path / ".env"

            client_path.write_text(json.dumps(valid_installed))
            token_path.write_text(json.dumps(valid_user_token))

            mock_creds = MagicMock()
            mock_creds.expired = False
            mock_creds.valid = True

            mock_service = MagicMock()
            mock_service.about().get().execute.return_value = {
                "user": {"emailAddress": "testuser@example.com"}
            }

            with patch("dataset.drive_auth.load_authorized_user_credentials", return_value=mock_creds), \
                 patch("dataset.drive_auth.ensure_drive_folder_tree", return_value=("root123", "shards456")), \
                 patch("dataset.drive_auth.run_smoke_test") as mock_smoke, \
                 patch("googleapiclient.discovery.build", return_value=mock_service), \
                 patch("sys.stdout", new_callable=io.StringIO) as fake_stdout:

                setup_drive_auth(client_path, token_path, env_path)
                output = fake_stdout.getvalue()

                self.assertIn("Authenticated Account: testuser@example.com", output)
                self.assertIn("Drive Storage Folder ID: shards456", output)
                self.assertNotIn(secret_token, output)
                self.assertNotIn("super-secret-client-key", output)

                mock_smoke.assert_called_once_with(mock_service, "shards456")
                env_content = env_path.read_text()
                self.assertIn("SMALL_LLM_DRIVE_FOLDER_ID=shards456", env_content)
                self.assertIn(f"SMALL_LLM_GOOGLE_OAUTH_TOKEN={token_path}", env_content)


class UpdateEnvFileTest(unittest.TestCase):
    def test_update_env_file_creates_and_updates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text("FOO=bar\nSMALL_LLM_DRIVE_FOLDER_ID=old_id\n")

            update_env_file(env_path, {"SMALL_LLM_DRIVE_FOLDER_ID": "new_id", "BAZ": "qux"})

            content = env_path.read_text()
            self.assertIn("FOO=bar", content)
            self.assertIn("SMALL_LLM_DRIVE_FOLDER_ID=new_id", content)
            self.assertNotIn("old_id", content)
            self.assertIn("BAZ=qux", content)


if __name__ == "__main__":
    unittest.main()
