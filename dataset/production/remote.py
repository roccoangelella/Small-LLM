"""Verified, idempotent Google Drive shard publication."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Mapping

from dataset.src.remote import RemoteShardStore, mirror_finalized_shard, write_drive_manifest
from dataset.src.storage import read_json

DRIVE_MANIFEST_FILENAME = "drive_manifest.json"
_TRANSIENT_STATUSES = {429, 500, 502, 503, 504}


def _remote_status(error: BaseException) -> int:
    try:
        return int(getattr(getattr(error, "resp", None), "status", 0))
    except (TypeError, ValueError):
        return 0


def remote_call(action: Callable[[], object]) -> object:
    for attempt in range(6):
        try:
            return action()
        except (TimeoutError, ConnectionError, OSError):
            if attempt == 5:
                raise
        except Exception as error:
            if _remote_status(error) not in _TRANSIENT_STATUSES or attempt == 5:
                raise
        time.sleep(min(30.0, float(2**attempt)))
    raise AssertionError("unreachable")


def _load_entries(
    path: Path,
    *,
    run_id: str,
    configuration_hash: str,
    schema_hash: str,
) -> list[dict[str, object]]:
    if not path.exists():
        return []
    payload = read_json(path)
    if not isinstance(payload, Mapping) or payload.get("version") != 1:
        raise RuntimeError("existing Drive manifest has an unsupported structure")
    if payload.get("run_id") != run_id:
        raise RuntimeError("existing Drive manifest belongs to a different run_id")
    if payload.get("configuration_hash") != configuration_hash:
        raise RuntimeError("existing Drive manifest configuration hash does not match this run")
    if payload.get("schema_hash") != schema_hash:
        raise RuntimeError("existing Drive manifest schema hash does not match this run")
    shards = payload.get("shards")
    if not isinstance(shards, list) or any(not isinstance(item, Mapping) for item in shards):
        raise RuntimeError("existing Drive manifest has an invalid shards list")
    entries = [dict(item) for item in shards]
    names = [entry.get("filename") for entry in entries]
    if any(not isinstance(name, str) for name in names) or len(names) != len(set(names)):
        raise RuntimeError("existing Drive manifest has invalid or duplicate filenames")
    return entries


def mirror_shards(
    store: RemoteShardStore,
    *,
    output_dir: Path,
    run_id: str,
    shard_entries: list[Mapping[str, object]],
    configuration_hash: str,
    schema_hash: str,
    verify_existing: bool = False,
    prune_unreferenced: bool = False,
) -> dict[str, object]:
    manifest_path = output_dir / DRIVE_MANIFEST_FILENAME
    mirrored = _load_entries(
        manifest_path,
        run_id=run_id,
        configuration_hash=configuration_hash,
        schema_hash=schema_hash,
    )
    by_name = {str(entry["filename"]): entry for entry in mirrored}

    for shard in shard_entries:
        filename = shard.get("filename")
        if not isinstance(filename, str):
            raise RuntimeError("cache shard metadata has no filename")
        existing = by_name.get(filename)
        if existing is not None:
            if (
                int(existing.get("byte_size", -1)) != int(shard.get("byte_size", -2))
                or existing.get("local_sha256") != shard.get("checksum")
            ):
                raise RuntimeError(f"Drive manifest disagrees with local immutable shard {filename}")
            if verify_existing:
                remote_call(lambda: store.verify_remote_shard(
                    run_id=run_id,
                    logical_name=filename,
                    file_id=str(existing["drive_file_id"]),
                    byte_size=int(existing["byte_size"]),
                    sha256=str(existing["local_sha256"]),
                ))
            continue

        raw = remote_call(lambda: mirror_finalized_shard(
            store,
            run_id=run_id,
            cache_root=output_dir,
            entry=shard,
            config_hash=configuration_hash,
            schema_hash=schema_hash,
        ))
        if not isinstance(raw, Mapping):
            raise RuntimeError("remote shard mirror returned invalid metadata")
        entry = dict(raw)
        mirrored.append(entry)
        by_name[filename] = entry
        write_drive_manifest(
            manifest_path,
            run_id=run_id,
            entries=mirrored,
            configuration_hash=configuration_hash,
            schema_hash=schema_hash,
        )

    if prune_unreferenced:
        referenced = {str(entry["filename"]) for entry in shard_entries}
        mirrored = [entry for entry in mirrored if str(entry.get("filename")) in referenced]

    return write_drive_manifest(
        manifest_path,
        run_id=run_id,
        entries=mirrored,
        configuration_hash=configuration_hash,
        schema_hash=schema_hash,
    )
