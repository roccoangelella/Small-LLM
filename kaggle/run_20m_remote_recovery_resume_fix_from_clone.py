#!/usr/bin/env python3
"""Run the 20M remote-recovery gate with the resume-argument hotfix.

Usage:
    %cd /kaggle/working/Small-LLM
    !git pull --ff-only
    !python kaggle/run_20m_remote_recovery_resume_fix_from_clone.py

The first remote-recovery launcher rewrites the command tail beginning at
``--wandb-tags``.  In resume commands, ``--resume`` follows that tail and was
therefore removed.  This wrapper imports the original fail-closed controller,
restores the missing resume argument, verifies its exact value, and then runs
the unchanged qualification sequence.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

IMPLEMENTATION = Path(__file__).with_name(
    "run_20m_remote_recovery_from_clone.py"
)


def load_implementation() -> Any:
    spec = importlib.util.spec_from_file_location(
        "small_llm_remote_recovery_implementation", IMPLEMENTATION
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load remote-recovery implementation: {IMPLEMENTATION}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


implementation = load_implementation()
_original_make_trainer_command = implementation.make_trainer_command


def make_trainer_command(*args: Any, **kwargs: Any) -> list[str]:
    """Preserve the checkpoint resume argument after W&B tag rewriting."""

    resume = kwargs.get("resume")
    command = list(_original_make_trainer_command(*args, **kwargs))
    if resume is None:
        if "--resume" in command:
            raise implementation.GateFailure(
                "Non-resume trainer command unexpectedly contains --resume"
            )
        return command

    if "--resume" not in command:
        command.extend(["--resume", str(resume)])

    positions = [index for index, value in enumerate(command) if value == "--resume"]
    if len(positions) != 1:
        raise implementation.GateFailure(
            f"Resume command must contain exactly one --resume flag, got {len(positions)}"
        )
    index = positions[0]
    if index + 1 >= len(command) or command[index + 1] != str(resume):
        actual = command[index + 1] if index + 1 < len(command) else None
        raise implementation.GateFailure(
            f"Resume checkpoint mismatch: expected {resume!r}, got {actual!r}"
        )
    return command


implementation.make_trainer_command = make_trainer_command


if __name__ == "__main__":
    raise SystemExit(implementation.main())
