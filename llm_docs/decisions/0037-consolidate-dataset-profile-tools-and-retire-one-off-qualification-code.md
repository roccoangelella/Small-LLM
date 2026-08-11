---
status: accepted
date: 2026-08-11
---

# Consolidate dataset profile tools and retire one-off qualification code

## Decision

Keep one active finite-dataset implementation and one profile-driven qualification surface.

- `dataset.production` remains the reusable schema-v2 producer implementation.
- Fixed 20M-model data-scaling stages are represented in one dataset profile registry rather than separate `qualification_100m.py`, `qualification_500m.py`, `qualification_2b.py` and matching report modules.
- One CLI accepts a profile argument for production and trainer-plan derivation.
- The historical initial 10M qualification profile may remain as data in the registry only where needed to reproduce its plan; its dedicated producer/verifier modules are retired.
- The completed full-corpus mixture-calibration executable and its active-package implementation/tests are retired. The approved weight-file hash and measured calibration evidence remain authoritative in project memory and the published standalone calibration repository.
- The completed original dataset operational-acceptance verifier is retired from the active package. Its accepted evidence remains in project memory and Git history.
- Google Drive credential loading/setup remains active because the production remote backend imports it during normal operation.
- `eval_core_v1` remains a separate builder because it produces a different evaluation artifact and is intentionally reusable across pretraining budgets.

## Rationale

The dataset directory had accumulated experiment-specific wrappers around the same producer and report engines. The 100M, 500M, and 2B wrappers duplicated source-token bounds, sequence geometry, shard geometry, checkpoint cadence, and profile names, while `kaggle/runtime.py` repeated much of the same identity again. That makes drift more likely and forces every new scaling point to add more files.

The full mixture scan and the initial operational-acceptance suite were qualification activities whose outputs are already frozen and recorded. Keeping their executables in the production package suggests they are recurring runtime dependencies when they are not.

## Consequences

- Adding another fixed data-scaling budget should normally mean adding one profile row, not new producer/report modules.
- Current Kaggle publication/training dispatch must call the unified dataset qualification CLI with an explicit profile.
- Tests should cover the shared registry/CLI and profile invariants instead of repeating near-identical wrapper tests.
- Historical documents may mention removed module paths as historical commands; Git history remains the implementation record for those completed one-off tools.
- Active documentation must distinguish recurring production/eval utilities from completed qualification tooling.
