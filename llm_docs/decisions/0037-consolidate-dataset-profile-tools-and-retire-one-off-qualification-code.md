---
status: accepted
date: 2026-08-11
supersedes: null
---

# 0037 — Consolidate dataset profile tools and retire one-off qualification code

## Context and problem statement

The active `dataset/` package accumulated experiment-specific wrappers around the same schema-v2 producer and trainer-plan engine. The 100M, 500M, and 2B wrappers duplicated source-token bounds, sequence geometry, shard geometry, checkpoint cadence, and profile names, while `kaggle/runtime.py` repeated much of the same dataset identity again. Adding another scaling point therefore required several near-identical files and created avoidable drift risk.

The package also still contained two qualification programs whose work was already complete: the full-corpus mixture calibration and the original 10M operational-acceptance verifier. Their accepted outputs and provenance are frozen in project memory, evidence, Git history, and—where applicable—the published standalone calibration repository. They are not recurring production dependencies.

At the same time, not every infrequently used file is disposable. Google Drive credential loading is part of normal remote production, `eval_core_v1` is a distinct reusable evaluation artifact, and the schema-v2 producer/streaming primitives remain active runtime code.

## Considered options

- Keep the per-budget producer/report wrappers and one-off qualification executables for historical convenience.
- Delete only obvious one-off scripts but retain duplicated per-budget wrappers.
- Consolidate finite scaling identities into one profile registry and CLI, retire completed one-off tooling, and retain only genuinely reusable production/evaluation primitives.

## Decision outcome

Chosen option: **consolidate finite dataset identities behind one profile-driven surface and retire completed one-off qualification code**, because it removes duplicated policy without changing the frozen source/data semantics.

- `dataset.production` remains the reusable schema-v2 producer implementation.
- Fixed 20M-model data-scaling stages are represented in one dataset profile registry instead of separate `qualification_100m.py`, `qualification_500m.py`, `qualification_2b.py`, and matching report modules.
- `dataset.qualification` accepts an explicit profile for production and trainer-plan derivation.
- The historical initial 10M profile remains reportable data for reproducibility, but its dedicated producer/verifier is retired and rebuilding it is disabled.
- The completed full-corpus mixture-calibration executable, implementation, and active tests are retired. The approved weight-file hash and measured evidence remain authoritative in project memory and the published standalone calibration repository.
- The completed original dataset operational-acceptance verifier is retired from the active package. Its accepted evidence remains in project memory and Git history.
- The obsolete monolithic `train.bin` / `validation.bin` builder path is retired; active verification targets schema-v2 sharded caches.
- Google Drive credential loading/setup remains active because the production remote backend imports it during normal operation.
- `eval_core_v1` remains a separate builder because it produces a different evaluation artifact and is intentionally reusable across pretraining budgets.

## Consequences

### Positive

- Adding another fixed data-scaling budget normally means adding one profile row instead of new producer/report modules.
- Dataset run ID, token envelope, checkpoint cadence, context geometry, and shard geometry have one experiment-facing source of truth.
- Kaggle publication/training dispatch uses the same profile registry as direct dataset production/reporting.
- Completed qualification tooling can no longer be mistaken for a recurring operational dependency.
- The remaining verifier is aligned with the schema-v2 sharded format used by current production.

### Negative or limiting

- Re-running the retired full mixture scan, original 10M acceptance workflow, or obsolete monolithic builder requires checking out historical code (or using the standalone calibration repository for the mixture scan).
- Historical evidence and archived runbooks intentionally retain old module names even though those modules no longer exist on `main`.
- The historical `20m-10m` profile can derive its accepted trainer plan but cannot be produced again through the active CLI.

## Validation

- Unified profile tests must preserve the exact 100M, 500M, and 2B source-token envelopes, run IDs, geometry, and WSD schedules.
- The unified producer surface must reject attempts to override profile identity/geometry and must reject rebuilding the historical 10M profile.
- Selected-profile report validation must fail closed on a wrong production run ID.
- Streaming and production resume/remote-durability tests must continue to pass after legacy-format removal.
- Current Kaggle runtime tests must resolve dataset identity from the shared registry and no active code may reference deleted per-budget/one-off modules.

## Links

- [`../reference/dataset_and_tokenization.md`](../reference/dataset_and_tokenization.md)
- [`../runbooks/unified_kaggle_launcher.md`](../runbooks/unified_kaggle_launcher.md)
- [`../../dataset/README.md`](../../dataset/README.md)
