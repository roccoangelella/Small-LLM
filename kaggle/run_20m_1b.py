#!/usr/bin/env python3
"""Pinned one-click entry point for the 20M-model/1B-token experiment."""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from typing import Any

# This immutable worktree contains the complete 1B qualification surface plus
# the T4-qualified mixed FLA GDN-2 implementation and fla-core==0.5.2 runtime.
PINNED_LAUNCH_COMMIT = "13033d82abaadcda9d823a31eb2467771c1a8ca6"
WANDB_INIT_TIMEOUT_SECONDS = "30"
WANDB_RUN_ID = "20m-1b-data-001"
DURABILITY_EVERY_STEPS = 250
_BASE_QUALIFICATION_REPORT = "dataset.qualification_100m_report"
_PROFILE_QUALIFICATION_REPORT = "dataset.qualification_1b_report"

if any(
    argument == "--launch-commit" or argument.startswith("--launch-commit=")
    for argument in sys.argv[1:]
):
    raise SystemExit("this entry point fixes the launch commit; do not pass --launch-commit")

os.environ["SMALL_LLM_1B_LAUNCH_COMMIT"] = PINNED_LAUNCH_COMMIT
os.environ["WANDB_INIT_TIMEOUT"] = WANDB_INIT_TIMEOUT_SECONDS
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import run_20m_one_click as common  # noqa: E402
from run_20m_100m_console import install_common_console  # noqa: E402

install_common_console(common)

import run_20m_1b_data_scaling as experiment  # noqa: E402


def _rewrite_profile_command(command: Sequence[str]) -> list[str]:
    """Bind inherited 100M report dispatch to the 1B qualification profile."""

    return [
        _PROFILE_QUALIFICATION_REPORT if item == _BASE_QUALIFICATION_REPORT else item
        for item in command
    ]


_profile_console_run = common.run


def _run_1b_profile(command: Sequence[str], *args: Any, **kwargs: Any) -> dict[str, Any]:
    return _profile_console_run(_rewrite_profile_command(command), *args, **kwargs)


common.run = _run_1b_profile

# Attempt every remaining update in the exact finite 1B one-pass plan in each
# invocation. Fresh training starts from seed 17 at microbatch 4. The assembled
# GDN-2 model selects qualified mixed FLA automatically on CUDA from update 1.
# Verified 250-update checkpoints make later Kaggle sessions exact resumes.
experiment.configure_runtime(
    durability_every=DURABILITY_EVERY_STEPS,
    max_steps_per_session=sys.maxsize,
    wandb_run_id=WANDB_RUN_ID,
)
main = experiment.main


if __name__ == "__main__":
    raise SystemExit(main())
