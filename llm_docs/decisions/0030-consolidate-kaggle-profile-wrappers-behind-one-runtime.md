---
status: accepted
date: 2026-08-10
supersedes: 0029
---

# 0030 — Consolidate Kaggle profile wrappers behind one runtime

## Context and problem statement

ADR 0028 established `kaggle/launch.py` as the canonical human command surface but deliberately left separate 100M/500M/2B training wrappers, scaling overlays, publisher overlays, compatibility entries, and shell wrappers underneath it. ADR 0029 then deferred deeper consolidation until after the planned 2B run to avoid unnecessary launch-path churn.

The user subsequently made a new explicit decision: the per-profile differences should be expressed as arguments/profile data behind one or two callable entry points, rather than kept as a growing forest of executable files. Re-inspection showed that the 500M and 2B layers mostly changed fixed identity/configuration values and a small number of dispatch behaviors while reusing the same qualified 100M mechanics.

## Decision outcome

Consolidate the finite-data Kaggle command surface around:

```text
kaggle/launch.py   # only supported human CLI
kaggle/runtime.py  # explicit profile registry + train/publish adapters
```

`kaggle/runtime.py` owns the fixed profile data for 20M/100M, 20M/500M, and 20M/2B, including:

- immutable launch commit;
- dataset profile and run ID;
- accepted-source-token target/minimum/maximum and producer checkpoint cadence;
- producer and qualification-report modules;
- W&B run ID/name/token tag;
- dataset publication slug, path defaults, and environment namespaces;
- microbatch-probe policy;
- selected microbatch;
- 250-update durability/validation/remote-publication cadence.

The runtime adapts the already-qualified shared training and publication engines rather than duplicating those mechanics per profile. The shared engines remain internal implementation modules for now; they are no longer supported human entry points.

Publication bootstrap that previously lived in `.sh` wrappers is now part of the Python runtime. `launch.py publish ...` requires `uv` and the repository `.env`, then re-executes itself with Python 3.13 and `kaggle/requirements-100m-publish.txt` before executing the publisher. This preserves dependency/environment behavior while removing shell entry points.

Root-level Kaggle transport files matching `<number>.archive` remain excluded from publication tree identity for every profile; nested files with the same name remain hashed dataset content.

Training resume remains automatic/fail-closed. The runtime continues to use each profile's exact remote checkpoint namespace, dataset identity, frozen launch worktree, W&B identity, and qualification-plan module. The 2B profile remains fresh relative to 500M and retains mixed FLA from update 1 through its frozen launch commit.

## Removed profile-specific files

The following layers are superseded by `kaggle/runtime.py` and removed from `main`:

```text
kaggle/run_20m_100m.py
kaggle/run_20m_500m.py
kaggle/run_20m_2b.py
kaggle/run_20m_500m_data_scaling.py
kaggle/run_20m_2b_data_scaling.py
kaggle/build_and_push_100m_entry.py
kaggle/build_and_push_500m.py
kaggle/build_and_push_2b.py
kaggle/build_and_push_100m.sh
kaggle/build_and_push_500m.sh
```

`kaggle/build_and_push_2b.sh` had already been removed under ADR 0029.

Obsolete wrapper/overlay-specific tests are removed and replaced by unified launcher/runtime regression coverage.

## Intentionally retained internal modules

The following are retained because the new runtime still reuses their proven mechanics:

```text
kaggle/run_20m_100m_data_scaling.py
kaggle/run_20m_one_click.py
kaggle/run_20m_100m_console.py
kaggle/build_and_push_100m.py
kaggle/wandb_preflight.py
```

Their filenames reflect historical origin, not current human command surface. A later neutral-name extraction may be considered if it can be proven behavior-preserving, but it is not required for the profile consolidation.

FLA qualification/reproducibility scripts remain outside this consolidation because they are diagnostic evidence tools, not training/publication entry points.

## Consequences

### Positive

- Humans use one stable CLI for publication and training across all registered profiles.
- Adding a future model/token point requires a `ProfileSpec` row rather than new wrapper/overlay executables.
- Fixed experiment identities are visible in one table and can be regression-tested directly.
- The old publication shell behavior is preserved without retaining shell launchers.
- 2B training mechanics remain based on the already-qualified shared engine and frozen launch commit.

### Tradeoffs

- The internal shared engines still have historical 100M-oriented filenames.
- `runtime.py` deliberately adapts those proven engines rather than rewriting their checkpoint/training loops in the same change; this minimizes behavioral risk but is not a full low-level code rewrite.

## Validation requirements

- `python kaggle/launch.py profiles` must list 20M/100M, 20M/500M, and 20M/2B.
- Train and publish dry-runs must resolve through `kaggle/runtime.py` and expose the selected fixed identities without importing a removed profile overlay.
- Runtime regression tests must assert all three qualified dataset/W&B/qualification-module identities and 40-hex launch pins.
- Runtime tests must preserve root-level Kaggle transport-archive exclusion.
- Publication bootstrap must preserve `uv`, Python 3.13, `.env`, and the existing publication requirements file.
- Active runbooks must contain only `kaggle/launch.py` commands for these profiles.

## Links

- [`0028-use-one-profile-driven-launcher-for-publication-and-training.md`](0028-use-one-profile-driven-launcher-for-publication-and-training.md)
- [`0029-limit-pre-2b-kaggle-cleanup-to-dead-wrappers-and-dispatch-fixes.md`](0029-limit-pre-2b-kaggle-cleanup-to-dead-wrappers-and-dispatch-fixes.md)
- [`../runbooks/unified_kaggle_launcher.md`](../runbooks/unified_kaggle_launcher.md)
- [`../runbooks/20m_2b_runbook.md`](../runbooks/20m_2b_runbook.md)
