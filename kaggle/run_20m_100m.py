#!/usr/bin/env python3
"""Pinned one-click entry point for the 20M-model/100M-token experiment."""

from __future__ import annotations

import os
import sys

PINNED_LAUNCH_COMMIT = "43190cb72443a2de290dc8e6f2c54f29d8dff501"

if any(argument == "--launch-commit" or argument.startswith("--launch-commit=") for argument in sys.argv[1:]):
    raise SystemExit("this entry point fixes the launch commit; do not pass --launch-commit")

os.environ["SMALL_LLM_100M_LAUNCH_COMMIT"] = PINNED_LAUNCH_COMMIT

from run_20m_100m_data_scaling import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
