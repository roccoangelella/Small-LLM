#!/usr/bin/env python3
"""Pinned one-click entry point for the 20M-model/100M-token experiment."""

from __future__ import annotations

import os
import sys

PINNED_LAUNCH_COMMIT = "e7a7d333c7720a7cc2b0f333c21416051aae9a04"
WANDB_INIT_TIMEOUT_SECONDS = "30"
WANDB_RUN_ID = "20m-100m-data-004"
DURABILITY_EVERY_STEPS = 250

if any(
    argument == "--launch-commit" or argument.startswith("--launch-commit=")
    for argument in sys.argv[1:]
):
    raise SystemExit("this entry point fixes the launch commit; do not pass --launch-commit")

os.environ["SMALL_LLM_100M_LAUNCH_COMMIT"] = PINNED_LAUNCH_COMMIT
os.environ["WANDB_INIT_TIMEOUT"] = WANDB_INIT_TIMEOUT_SECONDS
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import run_20m_one_click as common  # noqa: E402
from run_20m_100m_console import install_common_console, install_experiment_console  # noqa: E402

install_common_console(common)

import run_20m_100m_data_scaling as experiment  # noqa: E402

# The 100M run now attempts the complete remaining one-pass schedule in one
# Kaggle invocation. Periodic validation, local checkpoints, and verified
# remote publication all occur every 250 successful optimizer updates. The
# optional --max-steps-this-session flag still permits an explicit smaller cap.
experiment.LOCAL_EVERY = DURABILITY_EVERY_STEPS
experiment.EVAL_EVERY = DURABILITY_EVERY_STEPS
experiment.REMOTE_EVERY = DURABILITY_EVERY_STEPS
experiment.MAX_STEPS_PER_SESSION = sys.maxsize
experiment.WANDB_RUN_ID = WANDB_RUN_ID

install_experiment_console(experiment)
main = experiment.main


if __name__ == "__main__":
    raise SystemExit(main())
