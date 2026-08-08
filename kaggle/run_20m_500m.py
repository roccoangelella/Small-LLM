#!/usr/bin/env python3
"""Pinned one-click entry point for the 20M-model/500M-token experiment."""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from typing import Any

# This pinned worktree contains the checkpoint-compatible FLA GDN-2 backend.
# Historical 500M checkpoints keep their saved gdn_chunk_size=32 model config;
# CUDA recurrence execution uses FLA's fixed 64-token kernel internally.
PINNED_LAUNCH_COMMIT = "a1471472ca9b5d07f70c844460acffe5c96c5200"
WANDB_INIT_TIMEOUT_SECONDS = "30"
WANDB_RUN_ID = "20m-500m-data-001"
DURABILITY_EVERY_STEPS = 250
_BASE_QUALIFICATION_REPORT = "dataset.qualification_100m_report"
_PROFILE_QUALIFICATION_REPORT = "dataset.qualification_500m_report"

if any(
    argument == "--launch-commit" or argument.startswith("--launch-commit=")
    for argument in sys.argv[1:]
):
    raise SystemExit("this entry point fixes the launch commit; do not pass --launch-commit")

os.environ["SMALL_LLM_500M_LAUNCH_COMMIT"] = PINNED_LAUNCH_COMMIT
os.environ["WANDB_INIT_TIMEOUT"] = WANDB_INIT_TIMEOUT_SECONDS
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import run_20m_one_click as common  # noqa: E402
from run_20m_100m_console import install_common_console  # noqa: E402

install_common_console(common)

import run_20m_500m_data_scaling as experiment  # noqa: E402


def _rewrite_profile_command(command: Sequence[str]) -> list[str]:
    """Bind inherited 100M report dispatch to the 500M qualification profile."""

    return [
        _PROFILE_QUALIFICATION_REPORT if item == _BASE_QUALIFICATION_REPORT else item
        for item in command
    ]


# The private 500M overlay intentionally reuses the proven 100M launch loop.
# That loop contains one literal module dispatch for qualification-plan creation.
# Rewrite only that exact command at the process boundary so the verified 500M
# manifest is evaluated by its own profile. All other setup/training commands
# continue through the existing console/evidence wrapper unchanged.
_profile_console_run = common.run


def _run_500m_profile(command: Sequence[str], *args: Any, **kwargs: Any) -> dict[str, Any]:
    return _profile_console_run(_rewrite_profile_command(command), *args, **kwargs)


common.run = _run_500m_profile

# Attempt every remaining update in the exact finite 500M one-pass plan in each
# invocation. Fresh training starts directly at microbatch 4; resumed training
# restores the latest verified 500M checkpoint and keeps its exact optimizer,
# scheduler, scaler, RNG, data cursor, and saved model configuration. Only GDN-2
# CUDA recurrence execution changes to the qualified FLA backend.
# Kaggle/runtime interruption is handled by the verified 250-update
# validation/local-checkpoint/remote-publication protocol. The optional session
# cap remains available only for deliberate diagnostics.
experiment.configure_runtime(
    durability_every=DURABILITY_EVERY_STEPS,
    max_steps_per_session=sys.maxsize,
    wandb_run_id=WANDB_RUN_ID,
)
main = experiment.main


if __name__ == "__main__":
    raise SystemExit(main())
