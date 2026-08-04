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
    """Resolved remote publication objects and immutable Drive evidence."""

    publisher: TwoPhaseCheckpointPublisher
    drive_manifest: dict[str, object]
    every_steps: int


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
    )


__all__ = ["RemotePublication", "configure_remote_publication"]
