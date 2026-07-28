"""Offline-testable durable shard and checkpoint publication primitives.

The cache never treats cloud object names as truth.  Every immutable shard is
addressed by its logical run-relative name and SHA-256; providers only become a
durable mirror after their metadata agrees with the local bytes.  For Google
Drive, the SHA-256 in ``appProperties`` is only an identity precheck; downloaded
bytes are hashed locally because Drive does not independently attest SHA-256.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from io import BytesIO, FileIO
from pathlib import Path
from pathlib import PurePosixPath, PureWindowsPath
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


def md5_path(path: Path) -> str:
    """Return the provider-facing MD5 without replacing the SHA-256 identity."""

    try:
        digest = hashlib.md5(usedforsecurity=False)
    except TypeError:  # pragma: no cover - compatibility with older Python
        digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative_path(name: str) -> Path:
    """Convert a repository-relative name while rejecting traversal."""

    if not isinstance(name, str) or not name or "\x00" in name:
        raise RuntimeError(f"unsafe remote file name: {name!r}")
    # Remote names use POSIX separators even on Windows.  Rejecting a backslash
    # also prevents a name that is safe on this host from becoming traversal on
    # a later restore host.
    if "\\" in name:
        raise RuntimeError(f"unsafe remote file name: {name!r}")
    posix = PurePosixPath(name)
    windows = PureWindowsPath(name)
    if posix.is_absolute() or windows.is_absolute() or windows.drive:
        raise RuntimeError(f"unsafe remote file name: {name!r}")
    parts = name.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise RuntimeError(f"unsafe remote file name: {name!r}")
    return Path(*parts)


def safe_path_component(value: object, *, label: str) -> str:
    """Validate one path component used in local and remote checkpoint keys."""

    if not isinstance(value, str):
        raise RuntimeError(f"{label} must be a string")
    relative = _safe_relative_path(value)
    if len(relative.parts) != 1:
        raise RuntimeError(f"{label} must be a safe single path component")
    return value


def _reject_symlink_components(path: Path) -> None:
    """Reject symlink path components without resolving through them."""

    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.is_symlink():
            raise RuntimeError(f"refusing to follow symlink path component: {current}")


def _safe_download_target(root: Path, relative: Path) -> Path:
    _reject_symlink_components(root)
    root_resolved = root.resolve()
    target = root / relative
    _reject_symlink_components(target)
    try:
        target.resolve().relative_to(root_resolved)
    except ValueError as error:
        raise RuntimeError(f"download target escapes its destination: {relative.as_posix()}") from error
    return target


def ensure_safe_directory(path: Path) -> Path:
    """Create a directory without following symlinked path components."""

    _reject_symlink_components(path)
    if path.exists() and (path.is_symlink() or not path.is_dir()):
        raise RuntimeError(f"expected a real directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(path)
    if path.is_symlink() or not path.is_dir():
        raise RuntimeError(f"expected a real directory: {path}")
    return path


def safe_download_target(root: Path, relative: Path) -> Path:
    """Return a checked child path for a downloaded object."""

    return _safe_download_target(root, relative)


def _prepare_download_paths(destination: Path) -> tuple[Path, Path]:
    """Check a direct shard destination and its resumable sidecar."""

    _reject_symlink_components(destination)
    _reject_symlink_components(destination.parent)
    if destination.exists() and not destination.is_file():
        raise RuntimeError(f"download destination is not a regular file: {destination}")
    part = destination.with_name(destination.name + ".part")
    _reject_symlink_components(part)
    if part.exists() and not part.is_file():
        raise RuntimeError(f"partial download is not a regular file: {part}")
    return destination, part


def _tree_files(root: Path) -> list[Path]:
    """Return regular files while refusing symlinked tree entries."""

    _reject_symlink_components(root)
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"checkpoint tree is not a real directory: {root}")
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        _reject_symlink_components(path)
        if path.is_symlink():
            raise RuntimeError(f"checkpoint tree contains a symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise RuntimeError(f"checkpoint tree contains a non-regular file: {path}")
        files.append(path)
    return files


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
        run_id = safe_path_component(run_id, label="run_id")
        logical_name = _safe_relative_path(logical_name).as_posix()
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
        run_id = safe_path_component(run_id, label="run_id")
        logical_name = _safe_relative_path(logical_name).as_posix()
        data = self.files.get(file_id)
        if data is None or self.names.get((run_id, logical_name)) != file_id:
            raise RuntimeError("remote shard is missing or has the wrong logical name")
        if len(data) != byte_size or sha256_bytes(data) != sha256:
            raise RuntimeError("remote shard checksum or size mismatch")
        return {"file_id": file_id, "size": len(data), "sha256": sha256, "verified_at": "offline"}

    def download_shard(self, *, run_id: str, logical_name: str, file_id: str, destination: Path,
                       byte_size: int, sha256: str) -> None:
        self._fail("drive_download")
        run_id = safe_path_component(run_id, label="run_id")
        logical_name = _safe_relative_path(logical_name).as_posix()
        data = self.files.get(file_id)
        if data is None or self.names.get((run_id, logical_name)) != file_id:
            raise RuntimeError("remote shard is missing or has the wrong logical name")
        destination, part = _prepare_download_paths(destination)
        ensure_safe_directory(destination.parent)
        _prepare_download_paths(destination)
        offset = part.stat().st_size if part.exists() else 0
        if offset > len(data):
            raise RuntimeError("partial download is larger than remote object")
        with part.open("ab") as handle:
            handle.write(data[offset:])
            handle.flush()
            os.fsync(handle.fileno())
        if part.stat().st_size != byte_size or sha256_path(part) != sha256:
            raise RuntimeError("downloaded shard checksum or size mismatch")
        _prepare_download_paths(destination)
        os.replace(part, destination)

    def list_manifest_entries(self, run_id: str) -> list[dict[str, object]]:
        run_id = safe_path_component(run_id, label="run_id")
        return [{"logical_name": name, "file_id": file_id, "size": len(self.files[file_id]),
                 "sha256": sha256_bytes(self.files[file_id])}
                for (rid, name), file_id in self.names.items() if rid == run_id]

    def resume_upload(self, **kwargs: object) -> dict[str, object]:
        return self.upload_finalized_shard(**kwargs)  # type: ignore[arg-type]

    def resume_download(self, **kwargs: object) -> None:
        self.download_shard(**kwargs)  # type: ignore[arg-type]


class GoogleDriveShardStore:
    """Optional official-API backend; credentials are supplied, never stored.

    Drive ``appProperties`` carry the locally computed SHA-256 as an identity
    precheck.  They are not an independent Drive attestation of the bytes.  A
    download is authoritative only after the bytes have been hashed locally.
    """

    DOWNLOAD_RANGE_SIZE = 8 * 1024 * 1024

    def __init__(self, service: Any, folder_id: str) -> None:
        self.service, self.folder_id = service, folder_id

    @staticmethod
    def _escape_query_value(value: str) -> str:
        return value.replace("\\", "\\\\").replace("'", "\\'")

    def _run_folder(self, run_id: str) -> str:
        run_id = safe_path_component(run_id, label="run_id")
        escaped = self._escape_query_value(run_id)
        escaped_parent = self._escape_query_value(self.folder_id)
        query = (f"name = '{escaped}' and '{escaped_parent}' in parents and "
                 "mimeType = 'application/vnd.google-apps.folder' and trashed = false")
        found = self.service.files().list(q=query, fields="files(id)").execute().get("files", [])
        if len(found) > 1:
            raise RuntimeError(f"Drive contains duplicate run folders for {run_id!r}")
        if found:
            return str(found[0]["id"])
        return str(self.service.files().create(body={"name": run_id, "mimeType": "application/vnd.google-apps.folder",
                                                       "parents": [self.folder_id]}, fields="id").execute()["id"])

    @staticmethod
    def _remote_name(logical_name: str) -> str:
        _safe_relative_path(logical_name)
        return logical_name.replace("/", "__")

    def _retry(self, action: Callable[[], Any]) -> Any:
        try:
            from googleapiclient.errors import HttpError
        except ImportError:
            class HttpError(Exception):
                pass
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
    def from_credentials(cls, credentials_path: str | Path | None = None, *, folder_id: str) -> "GoogleDriveShardStore":
        try:
            from googleapiclient.discovery import build
        except ImportError as error:
            raise RuntimeError("Google Drive backend requires google-api-python-client") from error
        from dataset.drive_auth import load_authorized_user_credentials

        credentials = load_authorized_user_credentials(credentials_path)
        return cls(build("drive", "v3", credentials=credentials, cache_discovery=False), folder_id)


    def upload_finalized_shard(self, *, run_id: str, logical_name: str, local_path: Path,
                               resume_state: Mapping[str, object] | None = None) -> dict[str, object]:
        run_id = safe_path_component(run_id, label="run_id")
        logical_name = _safe_relative_path(logical_name).as_posix()
        if local_path.suffix != ".bin" or ".tmp" in local_path.name:
            raise ValueError("Google Drive accepts finalized immutable .bin shards only")
        try:
            from googleapiclient.http import MediaFileUpload
        except ImportError as error:
            raise RuntimeError("Google Drive backend requires google-api-python-client") from error
        size, digest, folder = local_path.stat().st_size, sha256_path(local_path), self._run_folder(run_id)
        local_md5 = md5_path(local_path)
        name = self._remote_name(logical_name)
        query = (f"name = '{self._escape_query_value(name)}' and "
                 f"'{self._escape_query_value(folder)}' in parents and trashed = false")
        existing = self.service.files().list(
            q=query, fields="files(id,size,md5Checksum,appProperties)"
        ).execute().get("files", [])
        if len(existing) > 1:
            raise RuntimeError(f"Drive contains duplicate shard objects for {logical_name!r}")
        if existing:
            item = existing[0]
            props = item.get("appProperties", {})
            if not isinstance(props, Mapping):
                raise RuntimeError("Drive shard metadata has invalid appProperties")
            if int(item.get("size", -1)) != size or props.get("sha256") != digest:
                raise RuntimeError("refusing to overwrite immutable Drive shard with different bytes")
            if item.get("md5Checksum") and item["md5Checksum"] != local_md5:
                raise RuntimeError("existing Drive shard MD5 does not match local bytes")
            return {"file_id": item["id"], "size": size, "sha256": digest,
                    "drive_md5": item.get("md5Checksum"), "upload_complete": True}
        media = MediaFileUpload(str(local_path), mimetype="application/octet-stream", resumable=True)
        request = self.service.files().create(
            body={"name": name, "parents": [folder], "appProperties": {"logical_name": logical_name, "sha256": digest}},
            media_body=media, fields="id,size,md5Checksum,appProperties",
        )
        response = None
        while response is None:
            response = self._retry(lambda: request.next_chunk()[1])
        if int(response.get("size", -1)) != size:
            raise RuntimeError("Drive upload returned an unexpected byte size")
        drive_md5 = response.get("md5Checksum")
        if drive_md5 and drive_md5 != local_md5:
            raise RuntimeError("Drive upload MD5 does not match local bytes")
        return {"file_id": response["id"], "size": size, "sha256": digest,
                "drive_md5": drive_md5, "upload_complete": True}

    def verify_remote_shard(self, *, run_id: str, logical_name: str, file_id: str,
                            byte_size: int, sha256: str) -> dict[str, object]:
        run_id = safe_path_component(run_id, label="run_id")
        logical_name = _safe_relative_path(logical_name).as_posix()
        metadata = self._retry(lambda: self.service.files().get(
            fileId=file_id, fields="id,size,md5Checksum,appProperties").execute())
        props = metadata.get("appProperties", {})
        if not isinstance(props, Mapping):
            raise RuntimeError("Drive metadata has invalid appProperties")
        if int(metadata.get("size", -1)) != byte_size or props.get("sha256") != sha256:
            raise RuntimeError("Drive metadata checksum or size mismatch")
        if props.get("logical_name") != logical_name:
            raise RuntimeError("Drive logical shard identity mismatch")
        return {"file_id": file_id, "size": byte_size, "sha256": sha256,
                "sha256_source": "Drive appProperties identity precheck; local bytes still require hashing",
                "drive_md5": metadata.get("md5Checksum"),
                "verified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    def download_shard(self, *, run_id: str, logical_name: str, file_id: str, destination: Path,
                       byte_size: int, sha256: str) -> None:
        run_id = safe_path_component(run_id, label="run_id")
        logical_name = _safe_relative_path(logical_name).as_posix()
        destination, part = _prepare_download_paths(destination)
        verified = self.verify_remote_shard(run_id=run_id, logical_name=logical_name, file_id=file_id,
                                            byte_size=byte_size, sha256=sha256)
        drive_md5 = verified.get("drive_md5")
        ensure_safe_directory(destination.parent)
        _prepare_download_paths(destination)
        offset = part.stat().st_size if part.exists() else 0
        if offset > byte_size:
            raise RuntimeError("partial Drive download is larger than expected")
        if offset == byte_size:
            if not part.exists():
                with FileIO(str(part), "w") as handle:
                    handle.flush()
                    os.fsync(handle.fileno())
            if sha256_path(part) != sha256:
                raise RuntimeError("complete partial Drive download checksum mismatch")
            if drive_md5 and md5_path(part) != drive_md5:
                raise RuntimeError("complete partial Drive download MD5 mismatch")
            _prepare_download_paths(destination)
            os.replace(part, destination)
            return
        # MediaIoBaseDownload manages a bounded chunk for a fresh FileIO, but
        # its resume offset is not an API contract for an arbitrary existing
        # .part file.  Explicit finite ranges make the resume point and the
        # amount buffered by the current HTTP transport unambiguous.
        uri = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media"
        mode = "r+" if part.exists() else "w+"
        with FileIO(str(part), mode) as handle:
            handle.seek(offset)
            while offset < byte_size:
                end = min(offset + self.DOWNLOAD_RANGE_SIZE, byte_size) - 1
                requested_size = end - offset + 1
                response, content = self._retry(
                    lambda start=offset, finish=end: self.service._http.request(
                        uri, method="GET", headers={"Range": f"bytes={start}-{finish}"}
                    )
                )
                status = int(getattr(response, "status", response.get("status", 0) if isinstance(response, Mapping) else 0))
                if status != 206:
                    raise RuntimeError(f"Drive range request failed with status {status}")
                if len(content) != requested_size:
                    raise RuntimeError(
                        "Drive range response length mismatch: "
                        f"requested {requested_size}, received {len(content)}"
                    )
                content_length = self._response_header(response, "content-length")
                if content_length is None or int(content_length) != len(content):
                    raise RuntimeError("Drive range response Content-Length mismatch")
                content_range = self._response_header(response, "content-range")
                match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", (content_range or "").strip())
                if (not match or int(match.group(1)) != offset or int(match.group(2)) != end
                        or int(match.group(3)) != byte_size):
                    raise RuntimeError("Drive range response Content-Range mismatch")
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
                offset += len(content)
        if part.stat().st_size != byte_size or sha256_path(part) != sha256:
            raise RuntimeError("downloaded Drive shard checksum or size mismatch")
        if drive_md5 and md5_path(part) != drive_md5:
            raise RuntimeError("downloaded Drive shard MD5 mismatch")
        _prepare_download_paths(destination)
        os.replace(part, destination)

    @staticmethod
    def _response_header(response: Any, name: str) -> str | None:
        """Read a header from httplib2's response or a small test double."""

        if isinstance(response, Mapping):
            value = response.get(name) or response.get(name.title()) or response.get(name.upper())
            return str(value) if value is not None else None
        getter = getattr(response, "get", None)
        if callable(getter):
            value = getter(name) or getter(name.title()) or getter(name.upper())
            if value is not None:
                return str(value)
        headers = getattr(response, "headers", None)
        if headers is not None:
            value = headers.get(name) or headers.get(name.title()) or headers.get(name.upper())
            return str(value) if value is not None else None
        return None

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
    run_id = safe_path_component(run_id, label="run_id")
    logical_name = _safe_relative_path(entry["filename"]).as_posix()
    local = cache_root / logical_name
    if local.suffix != ".bin" or ".tmp" in local.name:
        raise ValueError("only finalized immutable .bin shards may be mirrored")
    size, digest = local.stat().st_size, sha256_path(local)
    if size != int(entry["byte_size"]) or digest != str(entry["checksum"]):
        raise RuntimeError("local finalized shard disagrees with cache manifest")
    uploaded = store.upload_finalized_shard(run_id=run_id, logical_name=logical_name, local_path=local)
    verified = store.verify_remote_shard(run_id=run_id, logical_name=logical_name,
                                         file_id=str(uploaded["file_id"]), byte_size=size, sha256=digest)
    drive_checksums: dict[str, object] = {
        "sha256": verified["sha256"],
        "sha256_source": verified.get("sha256_source", "provider metadata identity precheck"),
    }
    if verified.get("drive_md5") is not None:
        drive_checksums["md5"] = verified["drive_md5"]
    return {**dict(entry), "run_id": run_id, "local_sha256": digest,
            "drive_file_id": verified["file_id"], "drive_checksums": drive_checksums,
            "remote_durable": True, "verification_timestamp": verified["verified_at"],
            "configuration_hash": config_hash, "schema_hash": schema_hash}


def write_drive_manifest(path: Path, *, run_id: str, entries: list[Mapping[str, object]],
                         configuration_hash: str, schema_hash: str) -> dict[str, object]:
    run_id = safe_path_component(run_id, label="run_id")
    manifest = {"version": 1, "run_id": run_id, "configuration_hash": configuration_hash,
                "schema_hash": schema_hash, "shards": [dict(entry) for entry in entries]}
    write_json_atomic(path, manifest)
    return manifest


class RemoteCheckpointStore(Protocol):
    def upload_tree(self, remote_prefix: str, local_dir: Path) -> dict[str, str]: ...
    def read_json(self, path: str) -> Mapping[str, object] | None: ...
    def write_json(self, path: str, value: Mapping[str, object]) -> None: ...
    def download_tree(self, remote_prefix: str, destination: Path) -> None: ...


class HuggingFaceCheckpointStore:
    """Durable checkpoint store backed by a private Hugging Face repository.

    The Hub upload API returns commit information, not a provider-independent
    SHA-256 attestation for each uploaded byte stream.  ``upload_tree`` reads
    each uploaded file back through ``hf_hub_download`` and returns hashes of
    those bytes before the publisher moves a pointer.  Restore still verifies
    the downloaded checkpoint manifests.  A caller-supplied downloader keeps
    this class straightforward to test without a live Hub.
    Read-back is still an API observation at the configured revision, not an
    independent cryptographic attestation of the upload stream; mutable Hub
    revisions can also have ordinary service-side propagation delay.
    The Hub dependency is intentionally imported only when this backend is
    constructed or used.
    """

    def __init__(self, repo_id: str, *, token: str | None = None,
                 private: bool = True, revision: str | None = None,
                 repo_type: str = "model", api: Any | None = None,
                 create_repo: bool = False,
                 downloader: Callable[..., str | Path] | None = None) -> None:
        self.repo_id = repo_id
        self.token = token
        self.private = private
        self.revision = revision
        self.repo_type = repo_type
        self.downloader = downloader
        if api is None:
            try:
                from huggingface_hub import HfApi
            except ImportError as error:
                raise RuntimeError(
                    "Hugging Face checkpoint backend requires huggingface_hub"
                ) from error
            api = HfApi(token=token)
        self.api = api
        if create_repo:
            self.ensure_repo()

    def ensure_repo(self) -> None:
        """Create the configured repository if it does not exist yet."""

        create = getattr(self.api, "create_repo", None)
        if not callable(create):
            raise RuntimeError("the configured Hugging Face API cannot create repositories")
        create_kwargs = self._hub_kwargs()
        create_kwargs.pop("revision", None)
        create(**create_kwargs, private=self.private, exist_ok=True)

    def _hub_kwargs(self) -> dict[str, object]:
        kwargs: dict[str, object] = {"repo_id": self.repo_id, "repo_type": self.repo_type}
        if self.token is not None:
            kwargs["token"] = self.token
        if self.revision is not None:
            kwargs["revision"] = self.revision
        return kwargs

    @staticmethod
    def _remote_prefix(remote_prefix: str) -> str:
        prefix = remote_prefix.rstrip("/")
        if not prefix:
            raise RuntimeError("Hugging Face checkpoint prefix cannot be empty")
        _safe_relative_path(prefix)
        return prefix

    def _upload_file(self, path: Path, path_in_repo: str) -> str:
        before = sha256_path(path)
        kwargs = self._hub_kwargs()
        self.api.upload_file(
            path_or_fileobj=str(path), path_in_repo=path_in_repo, **kwargs
        )
        # The API does not independently attest the hash.  Detect a local
        # writer changing the file while upload_file was reading it rather
        # than returning a digest for bytes that may not have been sent.
        after = sha256_path(path)
        if after != before:
            raise RuntimeError(f"local file changed during Hugging Face upload: {path}")
        remote_path = self._downloaded_path(path_in_repo, force_download=True)
        if not remote_path.is_file():
            raise RuntimeError(f"Hugging Face download did not return a regular file: {path_in_repo}")
        return sha256_path(remote_path)

    def _downloaded_path(self, path: str, *, force_download: bool = False) -> Path:
        if self.downloader is not None:
            return Path(self.downloader(filename=path, **self._hub_kwargs()))
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as error:
            raise RuntimeError(
                "Hugging Face checkpoint backend requires huggingface_hub"
            ) from error
        kwargs = self._hub_kwargs()
        if force_download:
            kwargs["force_download"] = True
        return Path(hf_hub_download(filename=path, **kwargs))

    def upload_tree(self, remote_prefix: str, local_dir: Path) -> dict[str, str]:
        prefix = self._remote_prefix(remote_prefix)
        result: dict[str, str] = {}
        for path in _tree_files(local_dir):
            relative = path.relative_to(local_dir).as_posix()
            key = f"{prefix}/{relative}"
            result[key] = self._upload_file(path, key)
        return result

    def read_json(self, path: str) -> Mapping[str, object] | None:
        _safe_relative_path(path)
        try:
            local_path = self._downloaded_path(path)
        except Exception as error:
            # ``EntryNotFoundError`` is optional and version-specific; keep the
            # protocol's missing-object result without importing that class.
            if error.__class__.__name__ in {
                "EntryNotFoundError", "RemoteEntryNotFoundError", "LocalEntryNotFoundError"
            }:
                return None
            raise
        payload = json.loads(Path(local_path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise RuntimeError(f"Hugging Face JSON object is not a mapping: {path}")
        return payload

    def write_json(self, path: str, value: Mapping[str, object]) -> None:
        _safe_relative_path(path)
        payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
        try:
            from huggingface_hub import CommitOperationAdd
        except ImportError:
            CommitOperationAdd = None  # type: ignore[assignment]
        kwargs = self._hub_kwargs()
        if CommitOperationAdd is not None and hasattr(self.api, "create_commit"):
            operation = CommitOperationAdd(path_in_repo=path, path_or_fileobj=BytesIO(payload))
            self.api.create_commit(operations=[operation], commit_message=f"update {path}", **kwargs)
            return
        self.api.upload_file(
            path_or_fileobj=BytesIO(payload), path_in_repo=path, **kwargs
        )

    def download_tree(self, remote_prefix: str, destination: Path) -> None:
        prefix = self._remote_prefix(remote_prefix)
        ensure_safe_directory(destination)
        list_files = getattr(self.api, "list_repo_files", None)
        if callable(list_files):
            files = list_files(**self._hub_kwargs())
            marker = prefix + "/"
            found = False
            for name in files:
                if not isinstance(name, str) or not name.startswith(marker):
                    continue
                found = True
                relative_name = name[len(marker):]
                relative = _safe_relative_path(relative_name)
                source = self._downloaded_path(name)
                target = _safe_download_target(destination, relative)
                ensure_safe_directory(target.parent)
                if source.resolve() != target.resolve():
                    shutil.copyfile(source, target)
            if not found:
                raise RuntimeError(f"Hugging Face checkpoint prefix is missing: {prefix}")
            return

        try:
            from huggingface_hub import snapshot_download
        except ImportError as error:
            raise RuntimeError(
                "Hugging Face checkpoint backend requires huggingface_hub"
            ) from error
        snapshot = Path(snapshot_download(
            allow_patterns=[prefix + "/**"], **self._hub_kwargs()
        ))
        source_root = snapshot / Path(*prefix.split("/"))
        if not source_root.exists():
            raise RuntimeError(f"Hugging Face checkpoint prefix is missing: {prefix}")
        for source in sorted(source_root.rglob("*")):
            if not source.is_file():
                continue
            relative = source.relative_to(source_root)
            safe_relative = _safe_relative_path(relative.as_posix())
            target = _safe_download_target(destination, safe_relative)
            ensure_safe_directory(target.parent)
            shutil.copyfile(source, target)


class InMemoryHuggingFaceStore:
    """Private-repository fake used by deterministic two-phase tests."""
    def __init__(self, fail: Callable[[str], None] | None = None) -> None:
        self.objects: dict[str, bytes] = {}
        self.fail = fail

    def upload_tree(self, remote_prefix: str, local_dir: Path) -> dict[str, str]:
        if self.fail:
            self.fail("hf_upload")
        remote_prefix = _safe_relative_path(remote_prefix.rstrip("/")).as_posix()
        result = {}
        for path in _tree_files(local_dir):
            key = remote_prefix.rstrip("/") + "/" + path.relative_to(local_dir).as_posix()
            self.objects[key] = path.read_bytes()
            result[key] = sha256_bytes(self.objects[key])
        return result

    def read_json(self, path: str) -> Mapping[str, object] | None:
        path = _safe_relative_path(path).as_posix()
        data = self.objects.get(path)
        return json.loads(data) if data is not None else None

    def write_json(self, path: str, value: Mapping[str, object]) -> None:
        if self.fail:
            self.fail("hf_pointer")
        path = _safe_relative_path(path).as_posix()
        self.objects[path] = json.dumps(value, sort_keys=True).encode()

    def download_tree(self, remote_prefix: str, destination: Path) -> None:
        remote_prefix = _safe_relative_path(remote_prefix.rstrip("/")).as_posix()
        prefix = remote_prefix.rstrip("/") + "/"
        for key, data in self.objects.items():
            if key.startswith(prefix):
                relative = _safe_relative_path(key[len(prefix):])
                target = _safe_download_target(destination, relative)
                ensure_safe_directory(target.parent)
                target.write_bytes(data)


def build_checkpoint_manifest(directory: Path) -> dict[str, object]:
    files = []
    for path in _tree_files(directory):
        relative = path.relative_to(directory)
        if relative.as_posix() != "checkpoint_manifest.json":
            files.append({"name": relative.as_posix(), "byte_size": path.stat().st_size,
                          "sha256": sha256_path(path)})
    return {"version": 1, "files": files}


class TwoPhaseCheckpointPublisher:
    """Publishes versioned last snapshots and only then moves latest.json."""
    def __init__(self, store: RemoteCheckpointStore, *, run_id: str) -> None:
        self.store = store
        self.run_id = safe_path_component(run_id, label="run_id")

    @staticmethod
    def _expected_upload_hashes(prefix: str, local_dir: Path) -> dict[str, str]:
        expected: dict[str, str] = {}
        for path in _tree_files(local_dir):
            relative = path.relative_to(local_dir).as_posix()
            expected[prefix.rstrip("/") + "/" + relative] = sha256_path(path)
        return expected

    @classmethod
    def _verify_upload_response(cls, prefix: str, local_dir: Path,
                                uploaded: Mapping[str, str] | object) -> None:
        if not isinstance(uploaded, Mapping):
            raise RuntimeError("remote checkpoint upload did not return a hash mapping")
        expected = cls._expected_upload_hashes(prefix, local_dir)
        actual = dict(uploaded)
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        if missing or unexpected:
            details = []
            if missing:
                details.append(f"missing={missing}")
            if unexpected:
                details.append(f"unexpected={unexpected}")
            raise RuntimeError("remote checkpoint upload file set mismatch (" + ", ".join(details) + ")")
        mismatched = sorted(
            key for key, digest in expected.items()
            if not isinstance(actual[key], str) or actual[key] != digest
        )
        if mismatched:
            raise RuntimeError(f"remote checkpoint upload SHA-256 mismatch: {mismatched}")

    def publish(self, checkpoint_dir: Path, *, checkpoint_id: str, drive_manifest: Mapping[str, object],
                metric: float | None = None, best_metric: float | None = None) -> dict[str, object]:
        run_id = safe_path_component(self.run_id, label="run_id")
        checkpoint_id = safe_path_component(checkpoint_id, label="checkpoint_id")
        _tree_files(checkpoint_dir)
        if drive_manifest.get("version") != 1 or drive_manifest.get("run_id") != run_id:
            raise RuntimeError("Drive manifest version or run_id does not match this publication")
        shards = drive_manifest.get("shards", [])
        if not isinstance(shards, list) or any(
            not isinstance(item, Mapping) or item.get("remote_durable") is not True
            for item in shards
        ):
            raise RuntimeError("cannot publish checkpoint before every referenced Drive shard is verified")
        write_json_atomic(checkpoint_dir / "drive_manifest.json", dict(drive_manifest))
        manifest = build_checkpoint_manifest(checkpoint_dir)
        write_json_atomic(checkpoint_dir / "checkpoint_manifest.json", manifest)
        prefix = f"run/{run_id}/checkpoints/{checkpoint_id}/last"
        uploaded = self.store.upload_tree(prefix, checkpoint_dir)
        self._verify_upload_response(prefix, checkpoint_dir, uploaded)
        # The pointer is the single commit boundary.  A failure before it leaves
        # the preceding pointer wholly usable; a failure after it means the new
        # verified snapshot is already authoritative.
        pointer = {"checkpoint_id": checkpoint_id, "last_prefix": prefix,
                   "checkpoint_manifest": manifest, "metric": metric}
        self.store.write_json(f"run/{run_id}/latest.json", pointer)
        best_updated = False
        if metric is not None and (best_metric is None or metric > best_metric):
            best_prefix = f"run/{run_id}/checkpoints/{checkpoint_id}/best"
            best_uploaded = self.store.upload_tree(best_prefix, checkpoint_dir)
            self._verify_upload_response(best_prefix, checkpoint_dir, best_uploaded)
            self.store.write_json(f"run/{run_id}/best.json", {"checkpoint_id": checkpoint_id,
                                  "best_prefix": best_prefix, "metric": metric,
                                  "source_checkpoint_id": checkpoint_id})
            best_updated = True
        return {"checkpoint_id": checkpoint_id, "latest": pointer, "best_updated": best_updated}

    def cleanup_history(self, *, destructive: bool = False, remote_verified: bool = False) -> None:
        if not destructive or not remote_verified:
            raise RuntimeError("history cleanup is irreversible, disabled by default, and requires verified remote state")
        raise NotImplementedError("destructive cleanup must be invoked by an explicit deployment command")
