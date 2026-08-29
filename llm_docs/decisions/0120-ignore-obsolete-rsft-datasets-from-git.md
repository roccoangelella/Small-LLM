---
status: accepted
date: 2026-08-24
supersedes: null
---

# ADR 0120 — Ignore obsolete R-SFT datasets from Git

## Context and problem statement

R-SFT dataset production leaves large local source scans, intermediate pools, and historical corpora under `artifacts/`. The completed 1% corpus is now published on `main`, while several earlier dataset artifacts are no longer needed as repository-resident training inputs.

A repository audit found one important exception: the frozen 16,716-row expanded corpus is still the active default input for the current Kaggle R-SFT production path and the base used by the nested 1% / 2% / 4% dataset builder. Removing it now would break those paths or silently change their semantics.

## Decision outcome

Keep only the currently active/frozen R-SFT corpora in Git:

- `artifacts/rsft-superior-instruction-r0-expanded/` — retained because it is still an active production/scaling dependency;
- `artifacts/rsft-superior-1pct/` — retained as the newly frozen 1% corpus.

Ignore and stop tracking obsolete/generated dataset artifacts, including:

- `artifacts/rsft-r0-pilot-630/`;
- `artifacts/rsft-superior-instruction-r0/`;
- `artifacts/rsft-1pct-work/`;
- `artifacts/rsft-superior-stage2-scaling/`.

Small audit/evidence artifacts such as curation files, manifests needed by active corpora, and prompt-comparison outputs are not treated as disposable datasets by this decision.

## Consequences

- Fresh clones no longer carry obsolete pilot/initial R-SFT corpus files.
- Large Stage-2 preparation and work directories remain local and cannot be accidentally committed.
- The current Kaggle R-SFT production path and nested scaling builder remain reproducible without changing their dataset semantics.
- Once the 16,716-row expanded corpus is no longer an active dependency, a later ADR may retire it as well.

## Validation

- `git check-ignore` must match the retired/scratch dataset paths.
- The 16,716-row expanded corpus and the 1% corpus must remain tracked.
- The working tree must not show Stage-2 scratch directories as untracked after the ignore update.
