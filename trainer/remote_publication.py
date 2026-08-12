"""Fail-closed live two-phase checkpoint publication for the trainer CLI."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from dataset.src.remote import HuggingFaceCheckpointStore, TwoPhaseCheckpointPublisher


@dataclass(frozen=True, slots=True)
class RemotePublication:
    """Resolved remote publication objects and immutable durability evidence."""

    publisher: TwoPhaseCheckpointPublisher
    drive_manifest: dict[str, object]
    every_steps: int
    rolling_latest_only: bool = False


def _read_drive_manifest(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise SystemExit(
            f"--remote-drive-manifest must name a regular file: {path}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as error:
        raise SystemExit(
            f"--remote-drive-manifest is not valid JSON: {path}"
        ) from error
    if not isinstance(payload, Mapping):
        raise SystemExit("--remote-drive-manifest must contain a JSON object")
    if payload.get("version") != 1:
        raise SystemExit("--remote-drive-manifest must use Drive manifest version 1")
    run_id = payload.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise SystemExit("--remote-drive-manifest has no valid run_id")
    shards = payload.get("shards")
    if not isinstance(shards, list):
        raise SystemExit("--remote-drive-manifest has no valid shards list")
    if any(
        not isinstance(item, Mapping) or item.get("remote_durable") is not True
        for item in shards
    ):
        raise SystemExit(
            "--remote-drive-manifest contains a shard that is not verified remote_durable"
        )
    return dict(payload)


def configure_remote_publication(args: object) -> RemotePublication | None:
    """Build the private-Hub publisher only when live publication is enabled."""

    every_steps = int(args.remote_publish_every_steps)
    if every_steps == 0:
        return None
    if every_steps < 0:
        raise SystemExit("--remote-publish-every-steps cannot be negative")
    if args.remote_drive_manifest is None:
        raise SystemExit(
            "--remote-drive-manifest is required when remote publication is enabled"
        )

    drive_manifest = _read_drive_manifest(Path(args.remote_drive_manifest))
    repo_id = args.remote_checkpoint_repo or os.environ.get("SMALL_LLM_HF_REPO_ID")
    if not repo_id:
        raise SystemExit(
            "set --remote-checkpoint-repo or SMALL_LLM_HF_REPO_ID when remote publication is enabled"
        )

    token_env = str(args.remote_token_env)
    token = os.environ.get(token_env)
    store = HuggingFaceCheckpointStore(
        str(repo_id),
        token=token,
        private=True,
        revision=args.remote_checkpoint_revision,
        create_repo=bool(args.remote_create_repo),
    )
    publisher = TwoPhaseCheckpointPublisher(
        store,
        run_id=str(drive_manifest["run_id"]),
    )
    return RemotePublication(
        publisher=publisher,
        drive_manifest=drive_manifest,
        every_steps=every_steps,
        rolling_latest_only=bool(getattr(args, "remote_rolling_latest_only", False)),
    )


def cleanup_remote_publication(
    remote: RemotePublication,
    *,
    checkpoint_id: str,
) -> dict[str, object] | None:
    """Prune old Hub checkpoints only after the new latest pointer is durable.

    This mode is deliberately Hugging-Face-specific. It removes older checkpoint
    folders from the branch head and then super-squashes the branch so periodic
    resume checkpoints do not accumulate unbounded Git/LFS/Xet history. The
    current latest pointer is read back after the squash before training may
    continue.
    """

    if not bool(getattr(remote, "rolling_latest_only", False)):
        return None

    run_id = remote.drive_manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise RuntimeError("rolling remote publication has no valid run_id")

    store = getattr(remote.publisher, "store", None)
    api = getattr(store, "api", None)
    repo_id = getattr(store, "repo_id", None)
    repo_type = getattr(store, "repo_type", "model")
    revision = getattr(store, "revision", None) or "main"
    token = getattr(store, "token", None)
    if api is None or not isinstance(repo_id, str) or not repo_id:
        raise RuntimeError("rolling remote publication requires a Hugging Face checkpoint store")

    list_repo_files = getattr(api, "list_repo_files", None)
    delete_folder = getattr(api, "delete_folder", None)
    delete_file = getattr(api, "delete_file", None)
    super_squash_history = getattr(api, "super_squash_history", None)
    if not all(callable(item) for item in (list_repo_files, delete_folder, delete_file, super_squash_history)):
        raise RuntimeError("configured Hugging Face API lacks rolling checkpoint cleanup methods")

    files = list_repo_files(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        token=token,
    )
    checkpoint_root = f"run/{run_id}/checkpoints/"
    checkpoint_ids: set[str] = set()
    for name in files:
        if not isinstance(name, str) or not name.startswith(checkpoint_root):
            continue
        remainder = name[len(checkpoint_root):]
        candidate = remainder.split("/", 1)[0]
        if candidate:
            checkpoint_ids.add(candidate)

    removed: list[str] = []
    for old_id in sorted(checkpoint_ids - {checkpoint_id}):
        delete_folder(
            path_in_repo=f"run/{run_id}/checkpoints/{old_id}",
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            token=token,
            commit_message=f"Prune superseded checkpoint {old_id}",
        )
        removed.append(old_id)

    best_path = f"run/{run_id}/best.json"
    if best_path in files:
        delete_file(
            path_in_repo=best_path,
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            token=token,
            commit_message="Remove best pointer in rolling latest-only mode",
        )

    super_squash_history(
        repo_id=repo_id,
        repo_type=repo_type,
        branch=revision,
        token=token,
        commit_message=f"Rolling checkpoint {checkpoint_id}",
    )

    pointer = remote.publisher.store.read_json(f"run/{run_id}/latest.json")
    if not isinstance(pointer, Mapping) or pointer.get("checkpoint_id") != checkpoint_id:
        raise RuntimeError("latest checkpoint pointer did not survive Hugging Face history squash")

    return {
        "status": "pruned_and_squashed",
        "checkpoint_id": checkpoint_id,
        "removed_checkpoint_ids": removed,
        "branch": revision,
    }


__all__ = [
    "RemotePublication",
    "cleanup_remote_publication",
    "configure_remote_publication",
]
