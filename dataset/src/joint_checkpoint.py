"""Framework-independent fixed-window joint checkpoint coordinator."""

from __future__ import annotations

import json
import os
import pickle
import random
import re
import shutil
import tempfile
from pathlib import Path
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Mapping, Protocol

from .remote import (
    TwoPhaseCheckpointPublisher,
    ensure_safe_directory,
    safe_download_target,
    safe_path_component,
    sha256_path,
)
from .storage import write_json_atomic


class TrainerCheckpointAdapter(Protocol):
    def state_dict(self) -> Mapping[str, object]: ...
    def load_state_dict(self, state: Mapping[str, object]) -> None: ...


_POST_SAVE_METADATA = frozenset({"local_manifest.json", "drive_manifest.json", "checkpoint_manifest.json"})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _manifest_relative_path(name: object) -> Path:
    """Return a safe checkpoint-relative path or reject an untrusted name."""

    if not isinstance(name, str) or not name or "\x00" in name or "\\" in name:
        raise RuntimeError(f"unsafe checkpoint manifest name: {name!r}")
    posix = PurePosixPath(name)
    windows = PureWindowsPath(name)
    if posix.is_absolute() or windows.is_absolute() or windows.drive:
        raise RuntimeError(f"unsafe checkpoint manifest name: {name!r}")
    parts = name.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise RuntimeError(f"unsafe checkpoint manifest name: {name!r}")
    return Path(*parts)


def _file_under(root: Path, relative: Path) -> Path:
    """Resolve a manifest file while also rejecting symlink escapes."""

    root_resolved = root.resolve()
    candidate = root / relative
    if candidate.is_symlink():
        raise RuntimeError(f"checkpoint manifest names a symlink: {relative.as_posix()}")
    try:
        candidate.resolve().relative_to(root_resolved)
    except ValueError as error:
        raise RuntimeError(f"checkpoint manifest path escapes its root: {relative.as_posix()}") from error
    return candidate


def _tree_file_names(root: Path) -> set[str]:
    names: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"checkpoint tree contains a symlink: {path.relative_to(root)}")
        if path.is_file():
            names.add(path.relative_to(root).as_posix())
    return names


def _validate_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise RuntimeError(f"checkpoint manifest {field} is not a SHA-256 hex digest")
    return value


def verify_local_manifest(root: Path) -> dict[str, object]:
    """Verify the local checkpoint manifest before any opaque state is read.

    The save-time manifest intentionally covers the trainer state and the
    inspectable checkpoint JSON, but not itself or publication-time metadata.
    Published trees may therefore contain the three known metadata files in
    addition to the manifest's listed files; every other file is rejected.
    """

    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"checkpoint root is not a directory: {root}")
    manifest_path = root / "local_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise RuntimeError("checkpoint local_manifest.json is missing or not a regular file")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as error:
        raise RuntimeError("checkpoint local_manifest.json is not valid JSON") from error
    if not isinstance(manifest, Mapping) or set(manifest) not in ({"files"}, {"version", "files"}):
        raise RuntimeError("checkpoint local_manifest.json has an invalid structure")
    if "version" in manifest and manifest.get("version") != 1:
        raise RuntimeError("checkpoint local_manifest.json has an invalid version")
    if not isinstance(manifest.get("files"), list):
        raise RuntimeError("checkpoint local_manifest.json has an invalid files list")

    listed: set[str] = set()
    for index, item in enumerate(manifest["files"]):
        if not isinstance(item, Mapping):
            raise RuntimeError(f"local manifest entry {index} is not an object")
        keys = set(item)
        if keys not in ({"name", "sha256"}, {"name", "sha256", "byte_size"}):
            raise RuntimeError(f"local manifest entry {index} has an invalid structure")
        relative = _manifest_relative_path(item.get("name"))
        normalized = relative.as_posix()
        if normalized in listed:
            raise RuntimeError(f"local manifest lists a duplicate file: {normalized}")
        if normalized in _POST_SAVE_METADATA:
            raise RuntimeError(f"local manifest cannot list publication metadata: {normalized}")
        listed.add(normalized)
        digest = _validate_sha256(item.get("sha256"), field=f"entry {index}")
        candidate = _file_under(root, relative)
        if not candidate.is_file():
            raise RuntimeError(f"local manifest file is missing: {normalized}")
        if "byte_size" in item:
            byte_size = item["byte_size"]
            if isinstance(byte_size, bool) or not isinstance(byte_size, int) or byte_size < 0:
                raise RuntimeError(f"local manifest byte_size is invalid: {normalized}")
            if candidate.stat().st_size != byte_size:
                raise RuntimeError(f"local manifest byte_size mismatch: {normalized}")
        if sha256_path(candidate) != digest:
            raise RuntimeError(f"local manifest SHA-256 mismatch: {normalized}")

    required = {"trainer_state.pkl", "checkpoint.json"}
    if not required.issubset(listed):
        raise RuntimeError(
            "local manifest does not cover required checkpoint files: "
            f"{sorted(required - listed)}"
        )
    actual = _tree_file_names(root)
    allowed_unlisted = _POST_SAVE_METADATA
    unexpected = sorted(actual - listed - allowed_unlisted)
    if unexpected:
        raise RuntimeError(f"checkpoint contains files outside local_manifest.json: {unexpected}")
    return dict(manifest)


def _verify_published_checkpoint_manifest(root: Path, supplied: object) -> dict[str, object]:
    """Verify a publisher manifest and the downloaded tree it describes.

    ``build_checkpoint_manifest`` excludes ``checkpoint_manifest.json`` so it
    cannot contain a self-hash. We instead require the downloaded manifest
    file to parse to exactly the pointer's supplied object, and require that it
    is the only file outside the supplied coverage set.
    """

    if not isinstance(supplied, Mapping) or set(supplied) != {"version", "files"}:
        raise RuntimeError("checkpoint pointer has an invalid checkpoint_manifest")
    if supplied.get("version") != 1 or not isinstance(supplied.get("files"), list):
        raise RuntimeError("checkpoint pointer has an invalid checkpoint_manifest version or files list")
    listed: set[str] = set()
    for index, item in enumerate(supplied["files"]):
        if not isinstance(item, Mapping) or set(item) != {"name", "byte_size", "sha256"}:
            raise RuntimeError(f"published checkpoint manifest entry {index} is malformed")
        relative = _manifest_relative_path(item.get("name"))
        normalized = relative.as_posix()
        if normalized in listed:
            raise RuntimeError(f"published checkpoint manifest lists a duplicate file: {normalized}")
        if normalized == "checkpoint_manifest.json":
            raise RuntimeError("published checkpoint manifest cannot list itself")
        listed.add(normalized)
        byte_size = item.get("byte_size")
        if isinstance(byte_size, bool) or not isinstance(byte_size, int) or byte_size < 0:
            raise RuntimeError(f"published checkpoint manifest byte_size is invalid: {normalized}")
        digest = _validate_sha256(item.get("sha256"), field=f"entry {index}")
        candidate = _file_under(root, relative)
        if not candidate.is_file():
            raise RuntimeError(f"published checkpoint manifest file is missing: {normalized}")
        if candidate.stat().st_size != byte_size or sha256_path(candidate) != digest:
            raise RuntimeError(f"published checkpoint manifest checksum mismatch: {normalized}")

    required = {"local_manifest.json", "drive_manifest.json", "checkpoint.json", "trainer_state.pkl"}
    if not required.issubset(listed):
        raise RuntimeError(
            "published checkpoint manifest does not cover required files: "
            f"{sorted(required - listed)}"
        )
    checkpoint_manifest_path = _file_under(root, Path("checkpoint_manifest.json"))
    if not checkpoint_manifest_path.is_file():
        raise RuntimeError("downloaded checkpoint_manifest.json is missing")
    try:
        downloaded = json.loads(checkpoint_manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as error:
        raise RuntimeError("downloaded checkpoint_manifest.json is not valid JSON") from error
    if downloaded != dict(supplied):
        raise RuntimeError("downloaded checkpoint_manifest.json disagrees with checkpoint pointer")

    actual = _tree_file_names(root)
    expected = listed | {"checkpoint_manifest.json"}
    if actual != expected:
        raise RuntimeError(
            "downloaded checkpoint tree does not match checkpoint_manifest.json: "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
        )
    return dict(supplied)


def _validate_drive_manifest(manifest: object, *, run_id: str) -> list[tuple[Mapping[str, object], Path]]:
    """Validate every shard reference before starting any prefetch download."""

    if not isinstance(manifest, Mapping) or manifest.get("version") != 1:
        raise RuntimeError("downloaded drive_manifest.json has an invalid version or structure")
    if manifest.get("run_id") != run_id or not isinstance(manifest.get("run_id"), str):
        raise RuntimeError("downloaded drive_manifest.json run_id mismatch")
    shards = manifest.get("shards")
    if not isinstance(shards, list):
        raise RuntimeError("downloaded drive_manifest.json has an invalid shards list")

    seen_names: set[str] = set()
    seen_ids: set[str] = set()
    validated: list[tuple[Mapping[str, object], Path]] = []
    for index, item in enumerate(shards):
        if not isinstance(item, Mapping):
            raise RuntimeError(f"drive manifest shard entry {index} is not an object")
        filename = item.get("filename")
        relative = _manifest_relative_path(filename)
        normalized = relative.as_posix()
        if normalized in seen_names:
            raise RuntimeError(f"drive manifest has a duplicate shard filename: {normalized}")
        seen_names.add(normalized)

        file_id = item.get("drive_file_id")
        if not isinstance(file_id, str) or not file_id:
            raise RuntimeError(f"drive manifest shard entry {index} has an invalid file ID")
        if file_id in seen_ids:
            raise RuntimeError(f"drive manifest has a duplicate Drive file ID: {file_id!r}")
        seen_ids.add(file_id)

        byte_size = item.get("byte_size")
        if isinstance(byte_size, bool) or not isinstance(byte_size, int) or byte_size < 0:
            raise RuntimeError(f"drive manifest shard entry {index} has an invalid byte_size")
        local_sha256 = _validate_sha256(
            item.get("local_sha256"), field=f"drive manifest shard {index}"
        )
        checksum = item.get("checksum")
        if checksum is not None and _validate_sha256(
                checksum, field=f"drive manifest shard {index} checksum"
        ) != local_sha256:
            raise RuntimeError(f"drive manifest shard entry {index} checksum mismatch")
        drive_checksums = item.get("drive_checksums")
        if drive_checksums is not None:
            if not isinstance(drive_checksums, Mapping):
                raise RuntimeError(f"drive manifest shard entry {index} has invalid drive_checksums")
            drive_sha256 = drive_checksums.get("sha256")
            if drive_sha256 is not None and _validate_sha256(
                    drive_sha256, field=f"drive manifest shard {index} Drive SHA-256"
            ) != local_sha256:
                raise RuntimeError(f"drive manifest shard entry {index} Drive SHA-256 mismatch")
            drive_md5 = drive_checksums.get("md5")
            if drive_md5 is not None and (
                    not isinstance(drive_md5, str) or not re.fullmatch(r"[0-9a-f]{32}", drive_md5)
            ):
                raise RuntimeError(f"drive manifest shard entry {index} has an invalid Drive MD5")
        if item.get("remote_durable") is not True:
            raise RuntimeError(f"drive manifest shard entry {index} is not remotely durable")
        entry_run_id = item.get("run_id")
        if entry_run_id is not None and entry_run_id != run_id:
            raise RuntimeError(f"drive manifest shard entry {index} has a run_id mismatch")
        validated.append((item, relative))
    return validated


def _prefetch_from_next_unconsumed_block(
    shards: list[tuple[Mapping[str, object], Path]],
    checkpoint_payload: object,
    prefetch_shards: int,
) -> list[tuple[Mapping[str, object], Path]]:
    """Select the train shard containing the next block, then its successors."""

    if prefetch_shards == 0:
        return []
    if not isinstance(checkpoint_payload, Mapping):
        raise RuntimeError("checkpoint.json is not an object")
    pipeline = checkpoint_payload.get("pipeline_state")
    if not isinstance(pipeline, Mapping):
        raise RuntimeError("checkpoint.json has no pipeline_state object")
    last_consumed = pipeline.get("last_consumed_block_id")
    if isinstance(last_consumed, bool) or not isinstance(last_consumed, int) or last_consumed < -1:
        raise RuntimeError("checkpoint pipeline state has an invalid last_consumed_block_id")
    next_block = last_consumed + 1

    train: list[tuple[int, int, Mapping[str, object], Path]] = []
    for entry, relative in shards:
        if entry.get("split") != "train":
            continue
        first, last = entry.get("first_block_id"), entry.get("last_block_id")
        if (
            isinstance(first, bool) or not isinstance(first, int)
            or isinstance(last, bool) or not isinstance(last, int)
            or first < 0 or last < first
        ):
            raise RuntimeError("drive manifest train shard has an invalid block range")
        train.append((first, last, entry, relative))
    train.sort(key=lambda item: (item[0], item[1], item[3].as_posix()))
    selected_index = next(
        (index for index, (first, last, _, _) in enumerate(train) if first <= next_block <= last),
        None,
    )
    if selected_index is None:
        raise RuntimeError(
            f"Drive manifest has no train shard containing next unconsumed block {next_block}"
        )
    return [(entry, relative) for _, _, entry, relative in train[selected_index:selected_index + prefetch_shards]]


def _fsync_tree(path: Path) -> None:
    for item in path.rglob("*"):
        if item.is_file():
            with item.open("rb") as handle:
                os.fsync(handle.fileno())
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _save_trainer_state(trainer: TrainerCheckpointAdapter, path: Path) -> None:
    serializer = getattr(trainer, "save_checkpoint_state", None)
    if callable(serializer):
        serializer(path)
        if not path.is_file():
            raise RuntimeError("trainer checkpoint serializer did not create trainer_state.pkl")
        return
    trainer_state = dict(trainer.state_dict())
    trainer_state.setdefault("python_rng_state", random.getstate())
    with path.open("wb") as handle:
        pickle.dump(trainer_state, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())


def _load_trainer_state(trainer: TrainerCheckpointAdapter, path: Path) -> None:
    loader = getattr(trainer, "load_checkpoint_state", None)
    if callable(loader):
        loader(path)
        return
    with path.open("rb") as handle:
        state = pickle.load(handle)
    trainer.load_state_dict(state)
    if "python_rng_state" in state:
        random.setstate(state["python_rng_state"])


class CheckpointCoordinator:
    """Makes a checkpoint only between completed optimizer steps.

    Trainer state is opaque to the coordinator. Framework adapters may provide
    streaming file serializers; simpler adapters retain the historical pickle
    fallback. The accompanying JSON keeps the data-pipeline contract inspectable.
    """

    def __init__(self, root: Path, *, configuration_hash: str, source_hash: str, schema_hash: str) -> None:
        self.root = root
        self.configuration_hash, self.source_hash, self.schema_hash = configuration_hash, source_hash, schema_hash
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, *, checkpoint_id: str, trainer: TrainerCheckpointAdapter,
             pipeline_state: Mapping[str, object], optimizer_step_complete: bool,
             validation_metrics: Mapping[str, object] | None = None) -> Path:
        checkpoint_id = safe_path_component(checkpoint_id, label="checkpoint_id")
        if not optimizer_step_complete:
            raise RuntimeError("joint checkpoints are legal only at completed optimizer-step boundaries")
        if int(pipeline_state.get("gradient_accumulation_position", 0)) != 0:
            raise RuntimeError("checkpoint has a partial gradient accumulation window")
        staging = Path(tempfile.mkdtemp(prefix=f".{checkpoint_id}.", dir=self.root))
        try:
            trainer_state_path = staging / "trainer_state.pkl"
            _save_trainer_state(trainer, trainer_state_path)
            payload = {
                "version": 1, "checkpoint_id": checkpoint_id,
                "configuration_hash": self.configuration_hash, "source_hash": self.source_hash,
                "schema_hash": self.schema_hash, "optimizer_step_complete": True,
                "pipeline_state": dict(pipeline_state), "validation_metrics": dict(validation_metrics or {}),
            }
            write_json_atomic(staging / "checkpoint.json", payload)
            manifest = {"files": [{"name": "trainer_state.pkl", "sha256": sha256_path(trainer_state_path)},
                                  {"name": "checkpoint.json", "sha256": sha256_path(staging / "checkpoint.json")}]} 
            write_json_atomic(staging / "local_manifest.json", manifest)
            _fsync_tree(staging)
            final = self.root / checkpoint_id
            if final.exists():
                raise FileExistsError(f"checkpoint already exists: {checkpoint_id}")
            os.replace(staging, final)
            parent_fd = os.open(self.root, os.O_RDONLY)
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
            return final
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    def load(self, checkpoint_id: str, trainer: TrainerCheckpointAdapter) -> dict[str, object]:
        checkpoint_id = safe_path_component(checkpoint_id, label="checkpoint_id")
        root = self.root / checkpoint_id
        verify_local_manifest(root)
        payload = json.loads((root / "checkpoint.json").read_text())
        if (payload["configuration_hash"], payload["source_hash"], payload["schema_hash"]) != (
            self.configuration_hash, self.source_hash, self.schema_hash):
            raise RuntimeError("checkpoint configuration/source/schema identity mismatch")
        _load_trainer_state(trainer, root / "trainer_state.pkl")
        return dict(payload["pipeline_state"])

    def publish(self, publisher: TwoPhaseCheckpointPublisher, *, checkpoint_id: str,
                 drive_manifest: Mapping[str, object], metric: float | None = None,
                 best_metric: float | None = None) -> dict[str, object]:
        checkpoint_id = safe_path_component(checkpoint_id, label="checkpoint_id")
        return publisher.publish(self.root / checkpoint_id, checkpoint_id=checkpoint_id,
                                 drive_manifest=drive_manifest, metric=metric, best_metric=best_metric)


def restore_on_empty_vps(*, publisher: TwoPhaseCheckpointPublisher, store: Any, run_id: str,
                         destination: Path, checkpoint_pointer: Mapping[str, object],
                         prefetch_shards: int) -> Path:
    """Stage and verify an empty-VPS checkpoint and shard window before install."""
    if isinstance(prefetch_shards, bool) or not isinstance(prefetch_shards, int) or prefetch_shards < 0:
        raise ValueError("prefetch_shards must be non-negative")
    run_id = safe_path_component(run_id, label="run_id")
    raw_prefix = checkpoint_pointer["last_prefix"]
    if not isinstance(raw_prefix, str):
        raise RuntimeError("checkpoint pointer has an invalid last_prefix")
    prefix = _manifest_relative_path(raw_prefix).as_posix()
    checkpoint_id = safe_path_component(checkpoint_pointer["checkpoint_id"], label="checkpoint_id")
    expected_prefix = f"run/{run_id}/checkpoints/{checkpoint_id}/last"
    if prefix != expected_prefix:
        raise RuntimeError("checkpoint pointer prefix does not match its run and checkpoint IDs")
    checkpoints_root = ensure_safe_directory(destination / "checkpoints")
    checkpoint_root = checkpoints_root / checkpoint_id
    if checkpoint_root.exists() or checkpoint_root.is_symlink():
        raise FileExistsError(f"checkpoint destination already exists: {checkpoint_root}")
    staging = Path(tempfile.mkdtemp(prefix=f".{checkpoint_id}.restore.", dir=checkpoints_root))
    cache_root = destination / "cache"
    cache_staging: Path | None = None
    cache_installed = False
    installed = False
    try:
        publisher.store.download_tree(prefix, staging)
        verify_local_manifest(staging)
        _verify_published_checkpoint_manifest(staging, checkpoint_pointer.get("checkpoint_manifest"))
        try:
            manifest = json.loads((staging / "drive_manifest.json").read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as error:
            raise RuntimeError("downloaded drive_manifest.json is not valid JSON") from error
        validated_shards = _validate_drive_manifest(manifest, run_id=run_id)
        try:
            checkpoint_payload = json.loads((staging / "checkpoint.json").read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as error:
            raise RuntimeError("downloaded checkpoint.json is not valid JSON") from error
        selected_shards = _prefetch_from_next_unconsumed_block(
            validated_shards, checkpoint_payload, prefetch_shards
        )
        if selected_shards:
            if cache_root.is_symlink():
                raise RuntimeError(f"cache destination is a symlink: {cache_root}")
            if cache_root.exists():
                raise FileExistsError(f"cache destination already exists: {cache_root}")
            cache_staging = Path(tempfile.mkdtemp(
                prefix=f".cache.{checkpoint_id}.restore.", dir=destination
            ))
            for entry, relative in selected_shards:
                target = safe_download_target(cache_staging, relative)
                store.download_shard(run_id=run_id, logical_name=relative.as_posix(),
                                     file_id=str(entry["drive_file_id"]), destination=target,
                                     byte_size=int(entry["byte_size"]), sha256=str(entry["local_sha256"]))
            os.replace(cache_staging, cache_root)
            cache_staging = None
            cache_installed = True

        if checkpoint_root.exists() or checkpoint_root.is_symlink():
            raise FileExistsError(f"checkpoint destination already exists: {checkpoint_root}")
        os.replace(staging, checkpoint_root)
        installed = True
        parent_fd = os.open(checkpoints_root, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        return checkpoint_root
    finally:
        if not installed:
            if cache_installed:
                shutil.rmtree(cache_root, ignore_errors=True)
            if cache_staging is not None:
                shutil.rmtree(cache_staging, ignore_errors=True)
            if staging.is_symlink() or (staging.exists() and not staging.is_dir()):
                staging.unlink(missing_ok=True)
            else:
                shutil.rmtree(staging, ignore_errors=True)
