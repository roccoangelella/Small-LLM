#!/usr/bin/env python3
"""Pinned one-click entry point for the 20M-model/500M-token experiment."""

from __future__ import annotations

import os
import sys

PINNED_LAUNCH_COMMIT = "01d562ea1845d0dd128a0458e613c9e677b7381d"
WANDB_INIT_TIMEOUT_SECONDS = "30"
WANDB_RUN_ID = "20m-500m-data-001"
DURABILITY_EVERY_STEPS = 250

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

# Attempt every remaining update in the exact finite 500M one-pass plan in each
# invocation. Fresh training starts directly at microbatch 4; the profile records
# the intentionally skipped microbatch probe instead of executing probe updates.
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
