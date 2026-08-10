# Unified Kaggle launcher

_Last updated: 2026-08-10 Europe/Rome_

`kaggle/launch.py` is the **only supported human entry point** for finite-dataset publication and Kaggle training. Model/token differences are data in `kaggle/runtime.py`; there are no separate 100M/500M/2B launcher or publisher commands.

## Supported profiles

```text
model  tokens
20M    100M
20M    500M
20M    2B
```

List the registry with:

```bash
python kaggle/launch.py profiles
```

Size arguments are case-insensitive and equivalent units are accepted, so `2B` and `2000M` resolve to the same profile.

## Publish a finite dataset

From the VPS repository:

```bash
python kaggle/launch.py publish --model 20M --tokens 2B
```

Earlier profiles use the same command surface:

```bash
python kaggle/launch.py publish --model 20M --tokens 100M
python kaggle/launch.py publish --model 20M --tokens 500M
```

The Python launcher now replaces the removed publication shell wrappers. On the first publication process it requires `uv` and `.env`, then re-executes itself under:

```text
Python 3.13
.env loaded by uv
kaggle/requirements-100m-publish.txt installed
```

The runtime then selects the profile-specific producer, report module, dataset identity, paths, and Kaggle handle namespace. Root-level Kaggle transport files matching `<number>.archive` are excluded from publication tree identity; nested files with the same name remain dataset content.

Stable publication controls:

```text
--weights-file PATH
--dataset-dir PATH
--ops-dir PATH
--kaggle-dataset-handle OWNER/DATASET
--force-upload
--remote-ready-timeout-seconds N
```

Publication resume is automatic. Rerun the identical command after interruption; do not pass `--resume`.

## Train or resume

From the Kaggle clone with the exact private dataset attached:

```bash
python kaggle/launch.py train --model 20M --tokens 2B
```

Earlier profiles use:

```bash
python kaggle/launch.py train --model 20M --tokens 100M
python kaggle/launch.py train --model 20M --tokens 500M
```

Training resume is automatic and fail-closed. The selected profile fixes the launch commit, dataset namespace, W&B identity, qualification-report module, token tag, microbatch policy, and 250-update durability cadence. If a verified matching checkpoint exists it restores exact model/optimizer/scheduler/scaler/RNG/data-cursor state; if none exists the profile follows its frozen fresh-start behavior.

Stable training controls:

```text
--dataset-dir PATH
--max-steps-this-session N
```

`--max-steps-this-session` is for deliberate bounded diagnostics only. Normal finite-plan training omits it.

The launcher intentionally rejects `--resume`; rerun the exact same command instead.

## Inspect without executing

```bash
python kaggle/launch.py train --model 20M --tokens 2B --dry-run
python kaggle/launch.py publish --model 20M --tokens 2B --dry-run
```

Dry-run is CPU-only. It reports the resolved runtime profile, immutable launch commit, dataset run ID, W&B run ID, and forwarded stable arguments without bootstrapping publication dependencies or launching training.

## Internal architecture

```text
human
  |
  v
kaggle/launch.py
  |
  v
kaggle/runtime.py        profile registry + train/publish adapters
  |                  \
  v                   v
shared qualified       shared qualified
training engine        publication engine
```

The remaining older 100M-named engine/helper modules are **internal shared implementations**, not human entry points. The former `run_20m_100m.py`, `run_20m_500m.py`, `run_20m_2b.py`, 500M/2B scaling overlays, 500M/2B publisher overlays, compatibility entry, and publication `.sh` wrappers have been removed from `main`.

The profile table preserves the historical contracts rather than re-deriving them from nominal sizes. In particular the 2B profile remains pinned to its already-qualified launch commit and uses its own dataset/W&B namespaces and `dataset.qualification_2b_report` dispatch.

## Adding a future profile

A new model/token experiment should add one qualified `ProfileSpec` row in `kaggle/runtime.py` plus its dataset qualification producer/report modules and regression tests. Do not create another `run_<size>_<tokens>.py`, `build_and_push_<tokens>.py`, or shell wrapper unless a genuinely different execution mechanism requires it.
