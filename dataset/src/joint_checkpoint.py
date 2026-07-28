"""Framework-independent fixed-window joint checkpoint coordinator."""

from __future__ import annotations

import json
import os
import pickle
import random
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Protocol

from .remote import TwoPhaseCheckpointPublisher, sha256_path
from .storage import write_json_atomic


class TrainerCheckpointAdapter(Protocol):
    def state_dict(self) -> Mapping[str, object]: ...
    def load_state_dict(self, state: Mapping[str, object]) -> None: ...


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


class CheckpointCoordinator:
    """Makes a checkpoint only between completed optimizer steps.

    Trainer state is opaque pickle here so PyTorch/JAX/other adapters can retain
    their native model, optimizer, scheduler, scaler and device RNG objects.
    The accompanying JSON makes the data-pipeline contract inspectable.
    """

    def __init__(self, root: Path, *, configuration_hash: str, source_hash: str, schema_hash: str) -> None:
        self.root = root
        self.configuration_hash, self.source_hash, self.schema_hash = configuration_hash, source_hash, schema_hash
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, *, checkpoint_id: str, trainer: TrainerCheckpointAdapter,
             pipeline_state: Mapping[str, object], optimizer_step_complete: bool,
             validation_metrics: Mapping[str, object] | None = None) -> Path:
        if not optimizer_step_complete:
            raise RuntimeError("joint checkpoints are legal only at completed optimizer-step boundaries")
        if int(pipeline_state.get("gradient_accumulation_position", 0)) != 0:
            raise RuntimeError("checkpoint has a partial gradient accumulation window")
        staging = Path(tempfile.mkdtemp(prefix=f".{checkpoint_id}.", dir=self.root))
        try:
            trainer_state = dict(trainer.state_dict())
            trainer_state.setdefault("python_rng_state", random.getstate())
            with (staging / "trainer_state.pkl").open("wb") as handle:
                pickle.dump(trainer_state, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.flush(); os.fsync(handle.fileno())
            payload = {
                "version": 1, "checkpoint_id": checkpoint_id,
                "configuration_hash": self.configuration_hash, "source_hash": self.source_hash,
                "schema_hash": self.schema_hash, "optimizer_step_complete": True,
                "pipeline_state": dict(pipeline_state), "validation_metrics": dict(validation_metrics or {}),
            }
            write_json_atomic(staging / "checkpoint.json", payload)
            manifest = {"files": [{"name": "trainer_state.pkl", "sha256": sha256_path(staging / "trainer_state.pkl")},
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
        root = self.root / checkpoint_id
        payload = json.loads((root / "checkpoint.json").read_text())
        if (payload["configuration_hash"], payload["source_hash"], payload["schema_hash"]) != (
            self.configuration_hash, self.source_hash, self.schema_hash):
            raise RuntimeError("checkpoint configuration/source/schema identity mismatch")
        with (root / "trainer_state.pkl").open("rb") as handle:
            state = pickle.load(handle)
        trainer.load_state_dict(state)
        if "python_rng_state" in state:
            random.setstate(state["python_rng_state"])
        return dict(payload["pipeline_state"])

    def publish(self, publisher: TwoPhaseCheckpointPublisher, *, checkpoint_id: str,
                drive_manifest: Mapping[str, object], metric: float | None = None,
                best_metric: float | None = None) -> dict[str, object]:
        return publisher.publish(self.root / checkpoint_id, checkpoint_id=checkpoint_id,
                                 drive_manifest=drive_manifest, metric=metric, best_metric=best_metric)


def restore_on_empty_vps(*, publisher: TwoPhaseCheckpointPublisher, store: Any, run_id: str,
                         destination: Path, checkpoint_pointer: Mapping[str, object],
                         prefetch_shards: int) -> Path:
    """Install only the requested initial immutable-shard window atomically."""
    prefix = str(checkpoint_pointer["last_prefix"])
    checkpoint_id = str(checkpoint_pointer["checkpoint_id"])
    checkpoint_root = destination / "checkpoints" / checkpoint_id
    publisher.store.download_tree(prefix, checkpoint_root)
    manifest = json.loads((checkpoint_root / "drive_manifest.json").read_text())
    for entry in list(manifest["shards"])[:prefetch_shards]:
        target = destination / "cache" / str(entry["filename"])
        store.download_shard(run_id=run_id, logical_name=str(entry["filename"]),
                             file_id=str(entry["drive_file_id"]), destination=target,
                             byte_size=int(entry["byte_size"]), sha256=str(entry["local_sha256"]))
    return checkpoint_root
