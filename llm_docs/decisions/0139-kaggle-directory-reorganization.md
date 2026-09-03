# 0139 - Reorganize Kaggle workspace into purpose subdirectories

Date: 2026-09-03
Status: Accepted

## Decision

Reorganize the overloaded `kaggle/` root into a smaller workspace with purpose-specific subdirectories:

- `kaggle/src/` for Python implementation files and runnable lower-level scripts.
- `kaggle/env/` for environment templates and dependency files.
- root-level `kaggle/README.md` as the index.
- root-level compatibility wrappers for `launch.py`, `launch_sft.py`, and `launch_r_sft.py`.

The Python files remain flat inside `kaggle/src/` for this pass. This is intentional: the current Kaggle modules are likely to depend on peer imports and direct script execution, so a deeper package split would require coordinated import rewrites and smoke testing.

## Rationale

The Kaggle root had accumulated many unrelated files: launchers, runtimes, qualification scripts, publication helpers, environment templates, and experiment-specific scripts. A purpose-based directory layout makes the workspace easier to navigate while preserving the canonical high-level launcher commands.

## Consequences

- Existing commands using `python kaggle/launch.py`, `python kaggle/launch_sft.py`, and `python kaggle/launch_r_sft.py` continue through thin wrappers.
- Lower-level scripts now live under `kaggle/src/`.
- Environment templates and publication requirements now live under `kaggle/env/`.
- Future cleanup can split `kaggle/src/` into proper packages after import checks and Kaggle smoke tests.
