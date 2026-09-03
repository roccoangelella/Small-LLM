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
TORCH_VERSION = "2.10.0"
TRITON_VERSION = "3.6.0"
FLA_VERSION = "0.5.2"
HF_HUB_VERSION = "1.5.0"
WANDB_VERSION = "0.26.1"
CUDA_WHEEL_INDEX = "https://download.pytorch.org/whl/cu128"


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


def _pin_qualified_runtime(command: list[str], python_index: int) -> list[str]:
    """Constrain the online subprocess to the runtime used by the T4 qualification."""

    command[python_index:python_index] = qualified_runtime_uv_args()
    return command


def qualified_runtime_uv_args() -> list[str]:
    """Return the exact uv arguments for the qualified Kaggle dual-T4 stack."""

    return [
        "--with",
        f"torch=={TORCH_VERSION}",
        "--with",
        f"triton=={TRITON_VERSION}",
        "--with",
        f"fla-core=={FLA_VERSION}",
        "--with",
        f"huggingface_hub=={HF_HUB_VERSION}",
        "--with",
        f"wandb=={WANDB_VERSION}",
        "--with",
        "packaging>=24",
        "--with",
        "numpy>=2.1,<3",
        "--extra-index-url",
        CUDA_WHEEL_INDEX,
    ]


def distributed_trainer_command(command: Sequence[str], *, worktree: Path) -> list[str]:
    """Replace only the Python trainer entrypoint; preserve every trainer flag."""

    rewritten = list(command)
    index = _trainer_module_index(rewritten)
    rewritten = _pin_qualified_runtime(rewritten, index)
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


__all__ = [
    "CUDA_WHEEL_INDEX",
    "DDP_ENTRYPOINT",
    "FLA_VERSION",
    "HF_HUB_VERSION",
    "TORCH_VERSION",
    "TRITON_VERSION",
    "WANDB_VERSION",
    "WORLD_SIZE",
    "distributed_trainer_command",
    "install",
    "qualified_runtime_uv_args",
]
