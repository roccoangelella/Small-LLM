---
status: superseded
date: 2026-08-10
supersedes: null
superseded_by: 0035
---

# 0026 — Prune superseded one-off Kaggle diagnostics

## Context and problem statement

The repository accumulated Kaggle launchers, recovery helpers, and GDN-2/FLA diagnostic scripts while qualifying the 20M training path. The active experiment is now the fresh 20M-model / 2B-token run, but several filenames from the completed 100M and 500M stages remain part of that active implementation: the 2B training launcher overlays the proven 100M data-scaling launcher, and the 2B dataset publisher overlays the proven 100M publisher. The repository test suite is also part of the main-branch CI contract.

At the same time, several one-off recovery and FLA investigation scripts have no live code or test callers. Their results are already preserved in evidence documents and their exact source remains recoverable from Git history. Keeping those scripts on the active Kaggle surface makes it harder to distinguish current operational commands from historical investigation tools.

## Considered options

- Delete all completed-stage Kaggle files and old tests.
- Keep every historical script indefinitely.
- Retain active dependencies, CI tests, reproducibility surfaces, and canonical qualification tools while pruning callerless one-off diagnostics whose evidence is already preserved.

## Decision outcome

Chosen option: **retain active/reproducible surfaces and prune only demonstrably callerless, superseded diagnostics**.

The following files are removed from `main`:

- `kaggle/run_20m_remote_recovery_resume_fix_from_clone.py`
- `kaggle/run_gdn2_step4000_decay_telemetry.py`
- `kaggle/run_gdn2_step4000_decay_telemetry_remote.py`
- `kaggle/run_gdn2_fla_amp_decay_sweep.py`
- `kaggle/run_gdn2_fla_052_amp_decay_sweep.py`
- `kaggle/run_gdn2_fla_strong_decay_amp_probe.py`

The following categories remain deliberately:

- the complete `tests/` tree, because `.github/workflows/tests.yml` compiles it and runs `python -m unittest discover -v` on pushes and pull requests;
- `kaggle/run_20m_one_click.py`, `kaggle/run_20m_100m_console.py`, and `kaggle/run_20m_100m_data_scaling.py`, because the active 2B launcher reuses them;
- `kaggle/build_and_push_100m.py` and its publication dependencies, because `kaggle/build_and_push_2b.py` overlays that implementation;
- the 500M operational launch/recovery surface that remains referenced by the retained 500M runbook for exact historical reproduction and checkpoint interpretation;
- canonical current FLA qualification tools, including the corrected deterministic FP32-oracle sweep and exact step-4000 parity/benchmark tools.

Historical evidence documents are not rewritten merely because an originating one-off script is removed. They remain evidence of what was run at the time; Git history preserves the corresponding source.

## Consequences

### Positive

- The live `kaggle/` surface is smaller and less ambiguous.
- Current 2B training and dataset-publication inheritance remains intact.
- Main-branch CI and regression coverage remain intact.
- Historical experimental evidence remains preserved without treating every temporary probe as permanent production code.

### Negative or limiting

- Re-running one of the removed historical probes requires checking out its earlier Git revision.
- Some immutable historical evidence may mention a script that is no longer present at `main`.
- The current 2B implementation still carries legacy `100m` filenames internally until a separate refactor removes that inheritance without changing behavior.

## Validation

- Main-branch unit CI must continue to pass after cleanup.
- `kaggle/run_20m_2b.py` and `kaggle/build_and_push_2b.py` must retain all inherited implementation files they import or dynamically load.
- Deleted scripts must have no live code/test callers on `main`.
- Canonical FLA qualification evidence/reference documentation must remain available.

## Links

- [`../current/status.md`](../current/status.md)
- [`../current/roadmap.md`](../current/roadmap.md)
- [`../reference/gdn2_fla_backend.md`](../reference/gdn2_fla_backend.md)
- [`../archive/gdn2_fla_investigation/gdn2_fla_investigation_handoff.md`](../archive/gdn2_fla_investigation/gdn2_fla_investigation_handoff.md)
- [`../runbooks/20m_2b_runbook.md`](../runbooks/20m_2b_runbook.md)
- [`README.md`](README.md)
