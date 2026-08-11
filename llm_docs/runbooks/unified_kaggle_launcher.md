# Unified Kaggle launcher

_Last updated: 2026-08-11 Europe/Rome_

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

The Python launcher replaces the removed publication shell wrappers. On the first publication process it requires `uv` and `.env`, then re-executes itself under:

```text
Python 3.13
.env loaded by uv
kaggle/requirements-100m-publish.txt installed
```

The runtime then selects the profile-specific producer contract, dataset identity, paths, and Kaggle handle namespace. Root-level Kaggle transport files matching `<number>.archive` are excluded from publication tree identity; nested files with the same name remain dataset content.

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

Training resume is automatic and fail-closed. The selected profile fixes the launch commit, dataset namespace, W&B identity, token tag, microbatch policy, and 250-update durability cadence. If a verified matching checkpoint exists it restores exact model/optimizer/scheduler/scaler/RNG/data-cursor state; if none exists the profile follows its frozen fresh-start behavior.

The experiment launch commit still pins model/trainer execution. Dataset verification and trainer-plan derivation are different: they run from the clean controlling checkout through the current consolidated `dataset.main` / `dataset.qualification` control plane. This is required because the accepted historical launch commits predate the unified dataset CLI. It preserves the pinned training implementation while ensuring all current profiles use the one authoritative dataset registry and schema-v2 verifier.

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
kaggle/runtime.py
  |                    \
  |                     +--> current dataset control plane
  |                          dataset.main + dataset.qualification
  v
pinned experiment worktree
model/trainer execution

publication -> current dataset.qualification -> dataset.production
```

The remaining older 100M-named engine/helper modules are **internal shared implementations**, not human entry points. The former `run_20m_100m.py`, `run_20m_500m.py`, `run_20m_2b.py`, 500M/2B scaling overlays, 500M/2B publisher overlays, compatibility entry, and publication `.sh` wrappers have been removed from `main`.

The profile table preserves the historical contracts rather than re-deriving them from nominal sizes. Dataset identity/geometry comes from the shared `dataset.qualification` registry; in particular the 2B profile remains pinned to its already-qualified training launch commit and its own dataset/W&B namespaces, while plan derivation dispatches through `dataset.qualification report --profile 20m-2b` in the controlling checkout.

## Adding a future profile

A new fixed data-scaling experiment should add one qualified `DatasetProfile` row in `dataset/qualification.py`, one matching `ProfileSpec` row in `kaggle/runtime.py`, and regression tests. Do not create another `qualification_<budget>.py`, report wrapper, `run_<size>_<tokens>.py`, `build_and_push_<tokens>.py`, or shell wrapper unless a genuinely different execution mechanism requires it.
