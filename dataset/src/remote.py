"""Offline-testable durable shard and checkpoint publication primitives.

The cache never treats cloud object names as truth.  Every immutable shard is
addressed by its logical run-relative name and SHA-256; providers only become a
durable mirror after their metadata agrees with the local bytes.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from io import FileIO
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from .storage import write_json_atomic


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RemoteShardStore(Protocol):
    def upload_finalized_shard(self, *, run_id: str, logical_name: str, local_path: Path,
                               resume_state: Mapping[str, object] | None = None) -> dict[str, object]: ...
    def verify_remote_shard(self, *, run_id: str, logical_name: str, file_id: str,
                            byte_size: int, sha256: str) -> dict[str, object]: ...
    def download_shard(self, *, run_id: str, logical_name: str, file_id: str, destination: Path,
                       byte_size: int, sha256: str) -> None: ...
    def list_manifest_entries(self, run_id: str) -> list[dict[str, object]]: ...
    def resume_upload(self, **kwargs: object) -> dict[str, object]: ...
    def resume_download(self, **kwargs: object) -> None: ...


class InMemoryDriveStore:
    """Byte-accurate Drive fake.  It deliberately supports partial downloads."""

    def __init__(self, fail: Callable[[str], None] | None = None) -> None:
        self.files: dict[str, bytes] = {}
        self.names: dict[tuple[str, str], str] = {}
        self.fail = fail

    def _fail(self, stage: str) -> None:
        if self.fail:
            self.fail(stage)

    def upload_finalized_shard(self, *, run_id: str, logical_name: str, local_path: Path,
                               resume_state: Mapping[str, object] | None = None) -> dict[str, object]:
        self._fail("drive_upload")
        data = local_path.read_bytes()
        key = (run_id, logical_name)
        previous = self.names.get(key)
        if previous and self.files[previous] != data:
            raise RuntimeError("refusing to overwrite immutable shard with different bytes")
        file_id = previous or hashlib.sha256((run_id + "\0" + logical_name).encode()).hexdigest()[:24]
        self.files[file_id] = data
        self.names[key] = file_id
        return {"file_id": file_id, "size": len(data), "sha256": sha256_bytes(data), "upload_complete": True}

    def verify_remote_shard(self, *, run_id: str, logical_name: str, file_id: str,
                            byte_size: int, sha256: str) -> dict[str, object]:
        self._fail("drive_verify")
        data = self.files.get(file_id)
        if data is None or self.names.get((run_id, logical_name)) != file_id:
            raise RuntimeError("remote shard is missing or has the wrong logical name")
        if len(data) != byte_size or sha256_bytes(data) != sha256:
            raise RuntimeError("remote shard checksum or size mismatch")
        return {"file_id": file_id, "size": len(data), "sha256": sha256, "verified_at": "offline"}

    def download_shard(self, *, run_id: str, logical_name: str, file_id: str, destination: Path,
                       byte_size: int, sha256: str) -> None:
        self._fail("drive_download")
        data = self.files[file_id]
        destination.parent.mkdir(parents=True, exist_ok=True)
        part = destination.with_name(destination.name + ".part")
        offset = part.stat().st_size if part.exists() else 0
        if offset > len(data):
            raise RuntimeError("partial download is larger than remote object")
        with part.open("ab") as handle:
            handle.write(data[offset:])
            handle.flush()
            os.fsync(handle.fileno())
        if part.stat().st_size != byte_size or sha256_path(part) != sha256:
            raise RuntimeError("downloaded shard checksum or size mismatch")
        os.replace(part, destination)

    def list_manifest_entries(self, run_id: str) -> list[dict[str, object]]:
        return [{"logical_name": name, "file_id": file_id, "size": len(self.files[file_id]),
                 "sha256": sha256_bytes(self.files[file_id])}
                for (rid, name), file_id in self.names.items() if rid == run_id]

    def resume_upload(self, **kwargs: object) -> dict[str, object]:
        return self.upload_finalized_shard(**kwargs)  # type: ignore[arg-type]

    def resume_download(self, **kwargs: object) -> None:
        self.download_shard(**kwargs)  # type: ignore[arg-type]


class GoogleDriveShardStore:
    """Optional official-API backend; credentials are supplied, never stored."""

    def __init__(self, service: Any, folder_id: str) -> None:
        self.service, self.folder_id = service, folder_id

    def _run_folder(self, run_id: str) -> str:
        escaped = run_id.replace("'", "\\'")
        query = (f"name = '{escaped}' and '{self.folder_id}' in parents and "
                 "mimeType = 'application/vnd.google-apps.folder' and trashed = false")
        found = self.service.files().list(q=query, fields="files(id)").execute().get("files", [])
        if found:
            return str(found[0]["id"])
        return str(self.service.files().create(body={"name": run_id, "mimeType": "application/vnd.google-apps.folder",
                                                       "parents": [self.folder_id]}, fields="id").execute()["id"])

    @staticmethod
    def _remote_name(logical_name: str) -> str:
        return logical_name.replace("/", "__")

    def _retry(self, action: Callable[[], Any]) -> Any:
        try:
            from googleapiclient.errors import HttpError
        except ImportError:
            HttpError = Exception  # type: ignore[assignment,misc]
        for attempt in range(6):
            try:
                return action()
            except HttpError as error:  # type: ignore[misc]
                status = getattr(getattr(error, "resp", None), "status", 0)
                if status not in (429, 500, 502, 503, 504) or attempt == 5:
                    raise
                time.sleep(min(30.0, 1.0 * (2 ** attempt)))
        raise AssertionError("unreachable")

    @classmethod
    def from_credentials(cls, credentials_path: str | None = None, *, folder_id: str) -> "GoogleDriveShardStore":
        try:
            from google.oauth2.service_account import Credentials
            from googleapiclient.discovery import build
        except ImportError as error:
            raise RuntimeError("Google Drive backend requires google-api-python-client") from error
        path = credentials_path or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if not path:
            raise RuntimeError("set GOOGLE_APPLICATION_CREDENTIALS or pass a credentials path")
        credentials = Credentials.from_service_account_file(path, scopes=["https://www.googleapis.com/auth/drive.file"])
        return cls(build("drive", "v3", credentials=credentials, cache_discovery=False), folder_id)

    def upload_finalized_shard(self, *, run_id: str, logical_name: str, local_path: Path,
                               resume_state: Mapping[str, object] | None = None) -> dict[str, object]:
        if local_path.suffix != ".bin" or ".tmp" in local_path.name:
            raise ValueError("Google Drive accepts finalized immutable .bin shards only")
        try:
            from googleapiclient.http import MediaFileUpload
        except ImportError as error:
            raise RuntimeError("Google Drive backend requires google-api-python-client") from error
        size, digest, folder = local_path.stat().st_size, sha256_path(local_path), self._run_folder(run_id)
        name = self._remote_name(logical_name)
        query = f"name = '{name}' and '{folder}' in parents and trashed = false"
        existing = self.service.files().list(q=query, fields="files(id,size,appProperties)").execute().get("files", [])
        if existing:
            item = existing[0]
            props = item.get("appProperties", {})
            if int(item.get("size", -1)) != size or props.get("sha256") != digest:
                raise RuntimeError("refusing to overwrite immutable Drive shard with different bytes")
            return {"file_id": item["id"], "size": size, "sha256": digest, "upload_complete": True}
        media = MediaFileUpload(str(local_path), mimetype="application/octet-stream", resumable=True)
        request = self.service.files().create(
            body={"name": name, "parents": [folder], "appProperties": {"logical_name": logical_name, "sha256": digest}},
            media_body=media, fields="id,size,md5Checksum,appProperties",
        )
        response = None
        while response is None:
            response = self._retry(lambda: request.next_chunk()[1])
        return {"file_id": response["id"], "size": int(response["size"]), "sha256": digest,
                "drive_md5": response.get("md5Checksum"), "upload_complete": True}

    def verify_remote_shard(self, *, run_id: str, logical_name: str, file_id: str,
                            byte_size: int, sha256: str) -> dict[str, object]:
        metadata = self._retry(lambda: self.service.files().get(
            fileId=file_id, fields="id,size,md5Checksum,appProperties").execute())
        if int(metadata.get("size", -1)) != byte_size or metadata.get("appProperties", {}).get("sha256") != sha256:
            raise RuntimeError("Drive metadata checksum or size mismatch")
        if metadata.get("appProperties", {}).get("logical_name") != logical_name:
            raise RuntimeError("Drive logical shard identity mismatch")
        return {"file_id": file_id, "size": byte_size, "sha256": sha256,
                "drive_md5": metadata.get("md5Checksum"), "verified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    def download_shard(self, *, run_id: str, logical_name: str, file_id: str, destination: Path,
                       byte_size: int, sha256: str) -> None:
        self.verify_remote_shard(run_id=run_id, logical_name=logical_name, file_id=file_id,
                                 byte_size=byte_size, sha256=sha256)
        destination.parent.mkdir(parents=True, exist_ok=True)
        part = destination.with_name(destination.name + ".part")
        offset = part.stat().st_size if part.exists() else 0
        if offset > byte_size:
            raise RuntimeError("partial Drive download is larger than expected")
        uri = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media"
        response, content = self._retry(lambda: self.service._http.request(uri, method="GET", headers={"Range": f"bytes={offset}-"}))
        if getattr(response, "status", 200) not in (200, 206):
            raise RuntimeError("Drive range request failed")
        with part.open("ab") as handle:
            handle.write(content); handle.flush(); os.fsync(handle.fileno())
        if part.stat().st_size != byte_size or sha256_path(part) != sha256:
            raise RuntimeError("downloaded Drive shard checksum or size mismatch")
        os.replace(part, destination)

    def list_manifest_entries(self, run_id: str) -> list[dict[str, object]]:
        folder = self._run_folder(run_id)
        files = self.service.files().list(q=f"'{folder}' in parents and trashed = false",
                                          fields="files(id,name,size,md5Checksum,appProperties)").execute().get("files", [])
        return [{"logical_name": item.get("appProperties", {}).get("logical_name"), "file_id": item["id"],
                 "size": int(item.get("size", 0)), "sha256": item.get("appProperties", {}).get("sha256"),
                 "drive_md5": item.get("md5Checksum")} for item in files]

    resume_upload = upload_finalized_shard
    resume_download = download_shard


def mirror_finalized_shard(store: RemoteShardStore, *, run_id: str, cache_root: Path,
                           entry: Mapping[str, object], config_hash: str, schema_hash: str) -> dict[str, object]:
    logical_name = str(entry["filename"])
    local = cache_root / logical_name
    if local.suffix != ".bin" or ".tmp" in local.name:
        raise ValueError("only finalized immutable .bin shards may be mirrored")
    size, digest = local.stat().st_size, sha256_path(local)
    if size != int(entry["byte_size"]) or digest != str(entry["checksum"]):
        raise RuntimeError("local finalized shard disagrees with cache manifest")
    uploaded = store.upload_finalized_shard(run_id=run_id, logical_name=logical_name, local_path=local)
    verified = store.verify_remote_shard(run_id=run_id, logical_name=logical_name,
                                         file_id=str(uploaded["file_id"]), byte_size=size, sha256=digest)
    return {**dict(entry), "run_id": run_id, "local_sha256": digest,
            "drive_file_id": verified["file_id"], "drive_checksums": {"sha256": verified["sha256"]},
            "remote_durable": True, "verification_timestamp": verified["verified_at"],
            "configuration_hash": config_hash, "schema_hash": schema_hash}


def write_drive_manifest(path: Path, *, run_id: str, entries: list[Mapping[str, object]],
                         configuration_hash: str, schema_hash: str) -> dict[str, object]:
    manifest = {"version": 1, "run_id": run_id, "configuration_hash": configuration_hash,
                "schema_hash": schema_hash, "shards": [dict(entry) for entry in entries]}
    write_json_atomic(path, manifest)
    return manifest


class RemoteCheckpointStore(Protocol):
    def upload_tree(self, remote_prefix: str, local_dir: Path) -> dict[str, str]: ...
    def read_json(self, path: str) -> Mapping[str, object] | None: ...
    def write_json(self, path: str, value: Mapping[str, object]) -> None: ...
    def download_tree(self, remote_prefix: str, destination: Path) -> None: ...


class InMemoryHuggingFaceStore:
    """Private-repository fake used by deterministic two-phase tests."""
    def __init__(self, fail: Callable[[str], None] | None = None) -> None:
        self.objects: dict[str, bytes] = {}
        self.fail = fail

    def upload_tree(self, remote_prefix: str, local_dir: Path) -> dict[str, str]:
        if self.fail:
            self.fail("hf_upload")
        result = {}
        for path in sorted(local_dir.rglob("*")):
            if path.is_file():
                key = remote_prefix.rstrip("/") + "/" + str(path.relative_to(local_dir))
                self.objects[key] = path.read_bytes()
                result[key] = sha256_bytes(self.objects[key])
        return result

    def read_json(self, path: str) -> Mapping[str, object] | None:
        data = self.objects.get(path)
        return json.loads(data) if data is not None else None

    def write_json(self, path: str, value: Mapping[str, object]) -> None:
        if self.fail:
            self.fail("hf_pointer")
        self.objects[path] = json.dumps(value, sort_keys=True).encode()

    def download_tree(self, remote_prefix: str, destination: Path) -> None:
        prefix = remote_prefix.rstrip("/") + "/"
        for key, data in self.objects.items():
            if key.startswith(prefix):
                target = destination / key[len(prefix):]
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)


def build_checkpoint_manifest(directory: Path) -> dict[str, object]:
    files = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name != "checkpoint_manifest.json":
            files.append({"name": str(path.relative_to(directory)), "byte_size": path.stat().st_size,
                          "sha256": sha256_path(path)})
    return {"version": 1, "files": files}


class TwoPhaseCheckpointPublisher:
    """Publishes versioned last snapshots and only then moves latest.json."""
    def __init__(self, store: RemoteCheckpointStore, *, run_id: str) -> None:
        self.store, self.run_id = store, run_id

    def publish(self, checkpoint_dir: Path, *, checkpoint_id: str, drive_manifest: Mapping[str, object],
                metric: float | None = None, best_metric: float | None = None) -> dict[str, object]:
        shards = drive_manifest.get("shards", [])
        if not isinstance(shards, list) or any(not item.get("remote_durable") for item in shards if isinstance(item, Mapping)):
            raise RuntimeError("cannot publish checkpoint before every referenced Drive shard is verified")
        write_json_atomic(checkpoint_dir / "drive_manifest.json", dict(drive_manifest))
        manifest = build_checkpoint_manifest(checkpoint_dir)
        write_json_atomic(checkpoint_dir / "checkpoint_manifest.json", manifest)
        prefix = f"run/{self.run_id}/checkpoints/{checkpoint_id}/last"
        uploaded = self.store.upload_tree(prefix, checkpoint_dir)
        if len(uploaded) != len(manifest["files"]) + 1:  # includes checkpoint manifest written after file scan
            raise RuntimeError("remote checkpoint upload is incomplete")
        # The pointer is the single commit boundary.  A failure before it leaves
        # the preceding pointer wholly usable; a failure after it means the new
        # verified snapshot is already authoritative.
        pointer = {"checkpoint_id": checkpoint_id, "last_prefix": prefix,
                   "checkpoint_manifest": manifest, "metric": metric}
        self.store.write_json(f"run/{self.run_id}/latest.json", pointer)
        best_updated = False
        if metric is not None and (best_metric is None or metric > best_metric):
            best_prefix = f"run/{self.run_id}/checkpoints/{checkpoint_id}/best"
            self.store.upload_tree(best_prefix, checkpoint_dir)
            self.store.write_json(f"run/{self.run_id}/best.json", {"checkpoint_id": checkpoint_id,
                                  "best_prefix": best_prefix, "metric": metric,
                                  "source_checkpoint_id": checkpoint_id})
            best_updated = True
        return {"checkpoint_id": checkpoint_id, "latest": pointer, "best_updated": best_updated}

    def cleanup_history(self, *, destructive: bool = False, remote_verified: bool = False) -> None:
        if not destructive or not remote_verified:
            raise RuntimeError("history cleanup is irreversible, disabled by default, and requires verified remote state")
        raise NotImplementedError("destructive cleanup must be invoked by an explicit deployment command")
