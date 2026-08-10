---
status: accepted
date: 2026-08-10
supersedes: null
---

# 0029 — Limit pre-2B Kaggle cleanup to dead wrappers and dispatch fixes

## Context and problem statement

After introducing the canonical profile-driven `kaggle/launch.py` front door, the `kaggle/` directory still contains historical wrappers, shared implementation modules, and FLA qualification utilities. The active 20M/2B launch path is already qualified, but it intentionally reuses several older 100M-named implementation modules. Removing or renaming those dependencies immediately before the 2B experiment would create unnecessary launch-path churn.

A cleanup audit also found that the unified 100M publication profile was dispatching directly to `build_and_push_100m.py`, while the established `build_and_push_100m_entry.py` compatibility entry excludes Kaggle's generated root-level `*.archive` transport artifact from tree-identity hashing.

## Decision outcome

Perform only cleanup that is clearly behavior-preserving before the 2B run:

- remove `kaggle/build_and_push_2b.sh`, because the canonical 2B runbook and unified launcher no longer call it;
- route the unified 20M/100M publication profile through `build_and_push_100m_entry.py`;
- expose `main = suite.main` from that compatibility entry so the unified dispatcher can call it with forwarded arguments;
- retain the active 2B implementation chain, including the older 100M-named shared modules, until after the 2B run;
- retain current FLA qualification and reproducibility scripts in `kaggle/` for now because current qualification/evidence documentation still references them;
- retain the 100M/500M shell wrappers while historical runbooks still document those exact procedures.

The active 2B runtime dependency chain therefore remains unchanged except for the removal of the unused shell front end.

## Consequences

### Positive

- The human-facing 2B publication command has a single canonical entry: `python kaggle/launch.py publish --model 20M --tokens 2B`.
- No qualified 2B trainer or publisher implementation module is renamed or structurally refactored immediately before execution.
- The unified 100M publisher preserves its established Kaggle transport-archive hashing behavior.
- Historical and diagnostic reproducibility remains intact.

### Deferred cleanup

After the 2B run, reconsider extracting shared training and publication mechanics into neutrally named modules (for example, generic scaling launcher/publisher modules), then remove obsolete profile wrappers and relocate qualification-only tools out of `kaggle/` if their documentation/tests are updated in the same change.

## Validation

- `tests/test_kaggle_launch.py` must assert that the 20M/100M publication profile resolves to `build_and_push_100m_entry`.
- `build_and_push_100m_entry.py` must expose a callable `main` compatible with the unified dispatcher's argv forwarding.
- The active 20M/2B runbook must contain no dependency on `build_and_push_2b.sh`.

## Links

- [`0028-use-one-profile-driven-launcher-for-publication-and-training.md`](0028-use-one-profile-driven-launcher-for-publication-and-training.md)
- [`0026-prune-superseded-one-off-kaggle-diagnostics.md`](0026-prune-superseded-one-off-kaggle-diagnostics.md)
- [`../runbooks/20m_2b_runbook.md`](../runbooks/20m_2b_runbook.md)
- `kaggle/launch.py`
