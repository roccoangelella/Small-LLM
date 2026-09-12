"""Complete-checkpoint discovery and local retention over a step-checkpoint directory.

The trainer, the pilot runner and the production runner all read the same
``step-XXXXXXXX`` sequence. Keeping the completeness contract in one module
prevents a second, subtly different definition of "this checkpoint can be
resumed from".
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
import pickle
import re
import shutil
import time

from .joint_checkpoint import verify_local_manifest
from .storage import read_json


CHECKPOINT_ID = re.compile(r"^step-(\d{1,12})$")

INVALID_CHECKPOINT = (
    OSError, EOFError, ValueError, TypeError, AttributeError, ImportError,
    IndexError, pickle.PickleError, RuntimeError,
)


def checkpoint_step(path: Path) -> int | None:
    match = CHECKPOINT_ID.fullmatch(path.name)
    return None if match is None else int(match.group(1))


def complete_checkpoint(path: Path, *, verify_state: bool = True) -> dict[str, object]:
    """Verify one checkpoint directory describes a completed optimizer update.

    ``verify_state`` also loads the opaque trainer state. Retention only needs
    the manifest hashes, which already prove the state file was written whole,
    and must not pay a multi-gigabyte unpickle on every new checkpoint.
    """

    step = checkpoint_step(path)
    if step is None or path.is_symlink() or not path.is_dir():
        raise ValueError("invalid checkpoint directory")
    verify_local_manifest(path)
    payload = read_json(path / "checkpoint.json")
    pipeline = payload.get("pipeline_state") if isinstance(payload, Mapping) else None
    if (not isinstance(payload, Mapping) or payload.get("checkpoint_id") != path.name or
        payload.get("optimizer_step_complete") is not True or not isinstance(pipeline, Mapping) or
        pipeline.get("gradient_accumulation_position", 0) != 0 or
        pipeline.get("last_consumed_block_id") != step - 1):
        raise ValueError("checkpoint metadata does not describe this completed update")
    result: dict[str, object] = {"checkpoint_id": path.name, "step": step, "path": path}
    if not verify_state:
        return result
    with (path / "trainer_state.pkl").open("rb") as handle:
        state = pickle.load(handle)
    if not isinstance(state, Mapping) or state.get("global_step") != step:
        raise ValueError("checkpoint state step differs from its identity")
    tokens = state.get("consumed_tokens")
    if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
        raise ValueError("invalid checkpoint target count")
    result["consumed_tokens"] = tokens
    return result


def find_latest_complete_checkpoint(checkpoint_dir: Path) -> dict[str, object] | None:
    """Quarantine invalid step directories and return the newest complete checkpoint."""

    if not checkpoint_dir.is_dir() or checkpoint_dir.is_symlink():
        return None
    latest = None
    for path in list(checkpoint_dir.iterdir()):
        if checkpoint_step(path) is None or not path.is_dir():
            continue
        try:
            checkpoint = complete_checkpoint(path)
        except INVALID_CHECKPOINT:
            quarantine = path.with_name(f"{path.name}.invalid-{time.time_ns()}")
            path.rename(quarantine)
            continue
        if latest is None or checkpoint["step"] > latest["step"]:
            latest = checkpoint
    return latest


def prune_checkpoints(checkpoint_dir: Path, *, keep_last: int,
                      protected: Iterable[str] = (), milestone_every_steps: int = 0) -> list[str]:
    """Delete superseded step checkpoints once the newest one is complete.

    Returns the removed checkpoint IDs. Zero ``keep_last`` retains everything,
    which is the historical behaviour.
    """

    if keep_last <= 0:
        return []
    if milestone_every_steps < 0:
        raise ValueError("milestone_every_steps cannot be negative")
    if not checkpoint_dir.is_dir() or checkpoint_dir.is_symlink():
        return []
    steps: list[tuple[int, Path]] = []
    for path in checkpoint_dir.iterdir():
        step = checkpoint_step(path)
        if step is not None and path.is_dir() and not path.is_symlink():
            steps.append((step, path))
    if not steps:
        return []
    steps.sort(key=lambda item: item[0], reverse=True)
    try:
        complete_checkpoint(steps[0][1], verify_state=False)
    except INVALID_CHECKPOINT:
        # The newest checkpoint is absent, partial or corrupt: keep every older one.
        return []
    keep = set(protected)
    removed: list[str] = []
    for step, path in steps[keep_last:]:
        if path.name in keep:
            continue
        if milestone_every_steps and step % milestone_every_steps == 0:
            continue
        shutil.rmtree(path)
        removed.append(path.name)
    return removed


__all__ = [
    "CHECKPOINT_ID",
    "INVALID_CHECKPOINT",
    "checkpoint_step",
    "complete_checkpoint",
    "find_latest_complete_checkpoint",
    "prune_checkpoints",
]
