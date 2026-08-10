---
status: superseded
date: 2026-08-10
supersedes: null
superseded_by: 0030
---

# 0029 — Limit pre-2B Kaggle cleanup to dead wrappers and dispatch fixes

## Context and problem statement

After introducing the canonical profile-driven `kaggle/launch.py` front door, the `kaggle/` directory still contained historical wrappers, shared implementation modules, and FLA qualification utilities. The active 20M/2B launch path was already qualified, but it intentionally reused several older 100M-named implementation modules. Removing or renaming those dependencies immediately before the 2B experiment was initially considered unnecessary launch-path churn.

A cleanup audit also found that the unified 100M publication profile was dispatching directly to `build_and_push_100m.py`, while the established `build_and_push_100m_entry.py` compatibility entry excluded Kaggle's generated root-level `*.archive` transport artifact from tree-identity hashing.

## Decision outcome

The initial decision was to perform only cleanup that was clearly behavior-preserving before the 2B run:

- remove `kaggle/build_and_push_2b.sh`, because the canonical 2B runbook and unified launcher no longer called it;
- route the unified 20M/100M publication profile through `build_and_push_100m_entry.py`;
- expose `main = suite.main` from that compatibility entry so the unified dispatcher could call it with forwarded arguments;
- retain the active 2B implementation chain, including the older 100M-named shared modules, until after the 2B run;
- retain current FLA qualification and reproducibility scripts in `kaggle/` because current qualification/evidence documentation still referenced them;
- retain the 100M/500M shell wrappers while historical runbooks documented those exact procedures.

## Supersession

ADR 0030 supersedes this conservative boundary after the user explicitly authorized pre-2B consolidation of profile-specific wrappers/overlays behind one profile-driven runtime. The underlying qualified shared engines remain retained, but the per-profile executable layers and remaining publication shell wrappers are removed.

## Links

- [`0030-consolidate-kaggle-profile-wrappers-behind-one-runtime.md`](0030-consolidate-kaggle-profile-wrappers-behind-one-runtime.md)
- [`0028-use-one-profile-driven-launcher-for-publication-and-training.md`](0028-use-one-profile-driven-launcher-for-publication-and-training.md)
- [`../runbooks/20m_2b_runbook.md`](../runbooks/20m_2b_runbook.md)
