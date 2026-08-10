# Unified Kaggle launcher

_Last updated: 2026-08-10 Europe/Rome_

`kaggle/launch.py` is the canonical human entry point for finite-dataset publication and training launch/resume. Profile-specific launchers and publishers remain implementation modules so the already-qualified 20M scaling paths do not need to be rewritten when the command surface is simplified.

## Supported profiles

```text
model  tokens
20M    100M
20M    500M
20M    2B
```

List the registry at any time with:

```bash
python kaggle/launch.py profiles
```

Size arguments are case-insensitive and equivalent units are accepted, so `2B` and `2000M` resolve to the same registered profile.

## Build, verify, and privately publish a dataset

Run from the VPS/repository environment that has the required Kaggle and Google Drive credentials:

```bash
python kaggle/launch.py publish --model 20M --tokens 2B
```

For the earlier profiles:

```bash
python kaggle/launch.py publish --model 20M --tokens 100M
python kaggle/launch.py publish --model 20M --tokens 500M
```

The unified entry point forwards the stable publication controls:

```text
--weights-file PATH
--dataset-dir PATH
--ops-dir PATH
--kaggle-dataset-handle OWNER/DATASET
--force-upload
--remote-ready-timeout-seconds N
```

Publication resume is automatic. If production was interrupted after creating its output directory, rerun the identical command. The profile-specific publisher selects its verified resume path; do not add a separate `--resume` mode.

## Launch or resume training

Run from the Kaggle notebook clone after attaching the exact verified private dataset and configuring the required secrets:

```bash
python kaggle/launch.py train --model 20M --tokens 2B
```

For the earlier profiles:

```bash
python kaggle/launch.py train --model 20M --tokens 100M
python kaggle/launch.py train --model 20M --tokens 500M
```

Training resume is also automatic and fail-closed. Every invocation checks the selected profile's remote checkpoint namespace. If a verified checkpoint exists and its dataset identity matches the attached dataset, it resumes exact model/optimizer/scheduler/scaler/RNG/data-cursor state. If no checkpoint exists, the profile follows its frozen fresh-start behavior.

The launcher intentionally rejects `--resume`; rerun the exact same command instead. This avoids separate fresh/resume command surfaces drifting apart.

Stable training controls exposed by the front door:

```text
--dataset-dir PATH
--max-steps-this-session N
```

`--max-steps-this-session` is for bounded diagnostics only. Normal finite-plan training omits it.

## Inspect without executing

Use `--dry-run` to verify profile selection and forwarded backend arguments without importing or running the profile backend:

```bash
python kaggle/launch.py train --model 20M --tokens 2B --dry-run
python kaggle/launch.py publish --model 20M --tokens 2B --dry-run
```

This path is CPU-only and is covered by the ordinary repository test suite.

## Architecture rule

`kaggle/launch.py` owns only the stable human command surface and the `(model size, token budget) -> implementation module` registry. Experiment geometry, immutable launch commits, dataset identities, W&B identities, checkpoint rules, qualification-plan dispatch, and publication verification remain inside the profile-specific implementation modules.

When a new model/token experiment is approved, add its qualified implementation modules first, then register exactly one new profile in `kaggle/launch.py` and add a dry-run dispatch test. Do not duplicate the full trainer or publisher inside the unified launcher.

The historical `run_20m_*` and `build_and_push_*` modules are therefore implementation details, not the preferred commands for humans. They remain in the repository while active or reproducible profiles depend on them.
