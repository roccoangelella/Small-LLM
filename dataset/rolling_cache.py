"""Rolling local cache for remotely stored immutable dataset shards.

The cache deliberately keeps only the current training shard plus a small
prefetch window. Validation shards are staged once and retained because they are
reused at every evaluation boundary. Dataset identity and ordering remain owned
by the completed schema-v2 manifest; this module only manages transport.
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol

from dataset.src.hf_bucket_shards import HuggingFaceBucketShardStore
from dataset.src.remote import ensure_safe_directory, sha256_path
from dataset.src.storage import write_json_atomic

STAGING_MARKER = "rolling_cache_stage.json"


@dataclass(frozen=True, slots=True)
class RemoteShard:
    filename: str
    split: str
    byte_size: int
    checksum: str
    first_block_id: int
    last_block_id: int


class ShardStore(Protocol):
    bucket_id: str

    @staticmethod
    def object_key(run_id: str, logical_name: str) -> str: ...

    def download_shard(
        self,
        *,
        run_id: str,
        logical_name: str,
        file_id: str,
        destination: Path,
        byte_size: int,
        sha256: str,
    ) -> None: ...


def _manifest_sha256(path: Path) -> str:
    return sha256_path(path)


def _parse_shards(manifest: Mapping[str, object]) -> tuple[list[RemoteShard], list[RemoteShard]]:
    raw = manifest.get("shards")
    if not isinstance(raw, list) or not raw:
        raise RuntimeError("dataset manifest has no shard inventory")
    train: list[RemoteShard] = []
    validation: list[RemoteShard] = []
    seen: set[str] = set()
    for index, row in enumerate(raw):
        if not isinstance(row, Mapping):
            raise RuntimeError(f"dataset shard entry {index} is not an object")
        filename = row.get("filename")
        split = row.get("split")
        byte_size = row.get("byte_size")
        checksum = row.get("checksum")
        first = row.get("first_block_id")
        last = row.get("last_block_id")
        if not isinstance(filename, str) or filename in seen:
            raise RuntimeError(f"dataset shard entry {index} has an invalid or duplicate filename")
        seen.add(filename)
        relative = Path(filename)
        if relative.is_absolute() or relative.parent != Path(str(split)) or relative.suffix != ".bin":
            raise RuntimeError(f"dataset shard entry {index} has an unsafe filename")
        if split not in {"train", "validation"}:
            raise RuntimeError(f"dataset shard entry {index} has an invalid split")
        if isinstance(byte_size, bool) or not isinstance(byte_size, int) or byte_size <= 0:
            raise RuntimeError(f"dataset shard entry {index} has an invalid byte size")
        if not isinstance(checksum, str) or len(checksum) != 64:
            raise RuntimeError(f"dataset shard entry {index} has an invalid checksum")
        if (
            isinstance(first, bool) or not isinstance(first, int) or first < 0
            or isinstance(last, bool) or not isinstance(last, int) or last < first
        ):
            raise RuntimeError(f"dataset shard entry {index} has an invalid block range")
        item = RemoteShard(filename, split, byte_size, checksum, first, last)
        (train if split == "train" else validation).append(item)

    train.sort(key=lambda item: (item.first_block_id, item.last_block_id, item.filename))
    validation.sort(key=lambda item: (item.first_block_id, item.last_block_id, item.filename))
    for rows, label in ((train, "train"), (validation, "validation")):
        expected = 0
        for item in rows:
            if item.first_block_id != expected:
                raise RuntimeError(f"{label} shard block ranges are not contiguous")
            expected = item.last_block_id + 1
    if not train or not validation:
        raise RuntimeError("dataset manifest must contain train and validation shards")
    return train, validation


def _train_index_for_block(train: list[RemoteShard], block_id: int) -> int | None:
    if isinstance(block_id, bool) or not isinstance(block_id, int) or block_id < 0:
        raise ValueError("train block ID must be a non-negative integer")
    for index, shard in enumerate(train):
        if shard.first_block_id <= block_id <= shard.last_block_id:
            return index
    if block_id == train[-1].last_block_id + 1:
        return None
    raise RuntimeError(f"dataset manifest has no train shard containing block {block_id}")


def _file_matches(root: Path, shard: RemoteShard) -> bool:
    path = root / shard.filename
    if path.is_symlink() or not path.is_file():
        return False
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return path.stat().st_size == shard.byte_size and sha256_path(path) == shard.checksum


def _download_verified(store: ShardStore, *, run_id: str, root: Path, shard: RemoteShard) -> Path:
    destination = root / shard.filename
    if _file_matches(root, shard):
        return destination
    if destination.exists() or destination.is_symlink():
        if destination.is_dir() and not destination.is_symlink():
            raise RuntimeError(f"dataset shard destination is unexpectedly a directory: {destination}")
        destination.unlink(missing_ok=True)
    ensure_safe_directory(destination.parent)
    store.download_shard(
        run_id=run_id,
        logical_name=shard.filename,
        file_id=store.object_key(run_id, shard.filename),
        destination=destination,
        byte_size=shard.byte_size,
        sha256=shard.checksum,
    )
    if not _file_matches(root, shard):
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"downloaded dataset shard failed local verification: {shard.filename}")
    return destination


def stage_dataset_window(
    *,
    store: HuggingFaceBucketShardStore,
    run_id: str,
    destination: Path,
    start_block_id: int = 0,
    train_shards: int = 1,
) -> dict[str, object]:
    """CPU-stage validation plus the train shard required by the next optimizer step.

    For a fresh run ``start_block_id=0`` stages ``train-000000.bin``. On resume,
    the caller supplies the completed optimizer-step count, which equals the
    next unconsumed block ID. This preserves the no-idle-H100 rule across
    provider/workspace restarts rather than only for the first launch.
    """

    if isinstance(train_shards, bool) or not isinstance(train_shards, int) or train_shards <= 0:
        raise ValueError("train_shards must be a positive integer")
    destination = ensure_safe_directory(destination)
    manifest_path = destination / "manifest.json"
    manifest = store.download_dataset_manifest(run_id=run_id, destination=manifest_path)
    train, validation = _parse_shards(manifest)
    start_index = _train_index_for_block(train, start_block_id)
    training_complete = start_index is None
    selected_train = (
        []
        if start_index is None
        else train[start_index : min(len(train), start_index + train_shards)]
    )
    selected_names = {item.filename for item in selected_train}

    # CPU staging is also the cleanup boundary after interrupted Modal sessions:
    # retain only the checkpoint-aligned train window. Validation stays resident.
    train_dir = ensure_safe_directory(destination / "train")
    ensure_safe_directory(destination / "validation")
    for path in train_dir.glob("train-*.bin"):
        relative = path.relative_to(destination).as_posix()
        if relative not in selected_names:
            path.unlink()

    if not training_complete:
        for shard in validation:
            _download_verified(store, run_id=run_id, root=destination, shard=shard)
        for shard in selected_train:
            _download_verified(store, run_id=run_id, root=destination, shard=shard)

    marker = {
        "version": 1,
        "transport": "hf-bucket-rolling-shards-v1",
        "bucket_id": store.bucket_id,
        "run_id": run_id,
        "manifest_sha256": _manifest_sha256(manifest_path),
        "start_block_id": start_block_id,
        "training_complete": training_complete,
        "staged_train_shards": [item.filename for item in selected_train],
        "validation_shards": [] if training_complete else [item.filename for item in validation],
    }
    write_json_atomic(destination / STAGING_MARKER, marker)
    return {
        "status": "training_complete" if training_complete else "ready",
        "dataset_dir": str(destination),
        **marker,
    }


def verify_staged_dataset(
    *,
    destination: Path,
    bucket_id: str,
    run_id: str,
    required_train_block: int | None = 0,
) -> dict[str, object]:
    marker_path = destination / STAGING_MARKER
    manifest_path = destination / "manifest.json"
    if not marker_path.is_file() or not manifest_path.is_file():
        raise RuntimeError("rolling dataset cache has not been CPU-staged")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(marker, Mapping) or marker.get("version") != 1:
        raise RuntimeError("rolling dataset staging marker is invalid")
    if marker.get("transport") != "hf-bucket-rolling-shards-v1":
        raise RuntimeError("rolling dataset staging marker has the wrong transport")
    if marker.get("bucket_id") != bucket_id or marker.get("run_id") != run_id:
        raise RuntimeError("rolling dataset staging identity mismatch")
    if marker.get("manifest_sha256") != _manifest_sha256(manifest_path):
        raise RuntimeError("rolling dataset manifest changed after CPU staging")
    if not isinstance(manifest, Mapping):
        raise RuntimeError("rolling dataset manifest is not a JSON object")
    production = manifest.get("production")
    if not isinstance(production, Mapping) or production.get("run_id") != run_id:
        raise RuntimeError("rolling dataset manifest run ID mismatch")
    train, validation = _parse_shards(manifest)
    if marker.get("training_complete") is True:
        if required_train_block is not None and required_train_block <= train[-1].last_block_id:
            raise RuntimeError("rolling dataset staging marker incorrectly claims training is complete")
        return {
            "status": "training_complete",
            "manifest_sha256": _manifest_sha256(manifest_path),
            "validation_shards": 0,
        }
    for shard in validation:
        if not _file_matches(destination, shard):
            raise RuntimeError(f"staged validation shard is missing or corrupt: {shard.filename}")
    required_name: str | None = None
    if required_train_block is not None:
        index = _train_index_for_block(train, required_train_block)
        if index is None:
            raise RuntimeError("required train block is beyond the completed dataset")
        required = train[index]
        required_name = required.filename
        if not _file_matches(destination, required):
            raise RuntimeError(
                f"train shard for block {required_train_block} is not ready before GPU dispatch"
            )
    return {
        "status": "verified",
        "manifest_sha256": _manifest_sha256(manifest_path),
        "required_train_block": required_train_block,
        "required_train_shard": required_name,
        "validation_shards": len(validation),
    }


class RollingShardCache:
    """Keep current train shard plus a bounded asynchronous look-ahead window."""

    def __init__(
        self,
        *,
        root: Path,
        run_id: str,
        manifest: Mapping[str, object],
        store: ShardStore,
        prefetch_shards: int = 1,
        evict_consumed: bool = True,
    ) -> None:
        if isinstance(prefetch_shards, bool) or not isinstance(prefetch_shards, int) or prefetch_shards < 1:
            raise ValueError("prefetch_shards must be at least one")
        self.root = root
        self.run_id = run_id
        self.store = store
        self.prefetch_shards = prefetch_shards
        self.evict_consumed = bool(evict_consumed)
        self.train, _ = _parse_shards(manifest)
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dataset-shard-prefetch")
        self._futures: dict[int, Future[Path]] = {}
        self._index_by_block: dict[int, int] = {}
        for index, shard in enumerate(self.train):
            for block_id in range(shard.first_block_id, shard.last_block_id + 1):
                self._index_by_block[block_id] = index
        self._current_index: int | None = None

    def _download(self, index: int) -> Path:
        return _download_verified(
            self.store,
            run_id=self.run_id,
            root=self.root,
            shard=self.train[index],
        )

    def _future(self, index: int) -> Future[Path]:
        with self._lock:
            future = self._futures.get(index)
            if future is None:
                future = self._executor.submit(self._download, index)
                self._futures[index] = future
            return future

    def _prefetch_after(self, index: int) -> None:
        for candidate in range(index + 1, min(len(self.train), index + 1 + self.prefetch_shards)):
            self._future(candidate)

    def ensure_block(self, block_id: int) -> None:
        index = self._index_by_block.get(block_id)
        if index is None:
            raise RuntimeError(f"rolling dataset cache has no shard for train block {block_id}")
        self._future(index).result()
        self._current_index = index
        self._prune_before(index)
        self._prefetch_after(index)

    def _prune_before(self, index: int) -> None:
        if not self.evict_consumed:
            return
        for prior in range(index):
            path = self.root / self.train[prior].filename
            if path.is_file() and not path.is_symlink():
                path.unlink()

    def acknowledge(self, block_id: int) -> None:
        index = self._index_by_block.get(block_id)
        if index is None:
            raise RuntimeError(f"rolling dataset cache cannot acknowledge unknown block {block_id}")
        shard = self.train[index]
        if block_id != shard.last_block_id:
            return
        if self.evict_consumed:
            path = self.root / shard.filename
            if path.is_file() and not path.is_symlink():
                path.unlink()
        next_index = index + 1
        if next_index < len(self.train):
            self._future(next_index)
            self._prefetch_after(next_index)

    def restore_after_acknowledged(self, block_id: int) -> None:
        next_block = block_id + 1
        index = self._index_by_block.get(next_block)
        if index is None:
            if next_block > self.train[-1].last_block_id:
                return
            raise RuntimeError(f"rolling dataset cache cannot restore after block {block_id}")
        self._prune_before(index)
        self._future(index)
        self._prefetch_after(index)

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


__all__ = [
    "RollingShardCache",
    "RemoteShard",
    "stage_dataset_window",
    "verify_staged_dataset",
]
