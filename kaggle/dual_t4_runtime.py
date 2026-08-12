#!/usr/bin/env python3
"""Kaggle-only adapter that promotes online training to exact-batch 2xT4 DDP.

The per-experiment worktree remains the source of model/trainer semantics.  This
adapter only rewrites the online trainer subprocess so ``torchrun`` executes the
controlling checkout's DDP shim against that pinned worktree.  Offline
microbatch probes and every Modal path remain untouched.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

REPO = Path(__file__).resolve().parents[1]
DDP_ENTRYPOINT = REPO / "kaggle" / "dual_t4_train.py"
WORLD_SIZE = 2


def _trainer_module_index(command: Sequence[str]) -> int:
    values = list(command)
    for index in range(len(values) - 2):
        if values[index : index + 3] == ["python", "-m", "trainer"]:
            return index
    raise RuntimeError("Kaggle trainer command no longer contains `python -m trainer`")


def _append_wandb_tag(command: list[str], tag: str) -> list[str]:
    if "--wandb-tags" not in command or tag in command:
        return command
    start = command.index("--wandb-tags") + 1
    stop = start
    while stop < len(command) and not command[stop].startswith("--"):
        stop += 1
    command.insert(stop, tag)
    return command


def distributed_trainer_command(command: Sequence[str], *, worktree: Path) -> list[str]:
    """Replace only the Python trainer entrypoint; preserve every trainer flag."""

    rewritten = list(command)
    index = _trainer_module_index(rewritten)
    replacement = [
        "python",
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={WORLD_SIZE}",
        str(DDP_ENTRYPOINT),
        "--worktree",
        str(worktree),
    ]
    rewritten[index : index + 3] = replacement
    return _append_wandb_tag(rewritten, "dual-t4-ddp")


def install(runtime: Any) -> None:
    """Install the Kaggle-only command rewrite before ``runtime.train`` loads its engine."""

    if getattr(runtime, "_KAGGLE_DUAL_T4_INSTALLED", False):
        return
    original_load = runtime._load

    def load(path: Path, name: str) -> Any:
        module = original_load(path, name)
        if Path(path).resolve() != Path(runtime.TRAINING_ENGINE).resolve():
            return module
        original_trainer_command = module.trainer_command

        def trainer_command(*args: Any, **kwargs: Any) -> list[str]:
            command = original_trainer_command(*args, **kwargs)
            # The historical launcher uses online=False for disposable microbatch
            # probes.  Production training is online=True and is the only path
            # promoted to DDP.
            if not bool(kwargs.get("online", False)):
                return command
            return distributed_trainer_command(command, worktree=Path(module.WORKTREE))

        module.trainer_command = trainer_command
        return module

    runtime._load = load
    runtime._KAGGLE_DUAL_T4_INSTALLED = True


__all__ = ["DDP_ENTRYPOINT", "WORLD_SIZE", "distributed_trainer_command", "install"]
