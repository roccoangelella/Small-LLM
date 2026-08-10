---
status: accepted
date: 2026-08-10
supersedes: null
---

# 0028 — Use one profile-driven launcher for publication and training

## Context and problem statement

The 20M scaling series accumulated separate human-facing commands for 100M, 500M, and 2B dataset publication and training. Those entry points encode useful qualified behavior, but requiring a different filename for every model/token combination makes the operational surface harder to remember and encourages future experiments to add more one-off launch scripts.

The active 2B path is already qualified through profile-specific implementation modules, so simplifying the human command surface must not require rewriting the trainer, publisher, checkpoint, dataset, or immutable-launch-commit logic immediately before the 2B experiment.

## Considered options

- Keep one human-facing training and publication filename per scaling profile.
- Rewrite all profile-specific launch/publisher implementations into one large monolithic script now.
- Add one stable profile-driven front door that dispatches to the already-qualified implementation modules.

## Decision outcome

Chosen option: **use `kaggle/launch.py` as the single canonical human entry point, with explicit `train` and `publish` subcommands plus `--model` and `--tokens` profile selection**.

Canonical examples:

```bash
python kaggle/launch.py publish --model 20M --tokens 2B
python kaggle/launch.py train --model 20M --tokens 2B
```

The same command is rerun after interruption. Resume is automatic and fail-closed in the selected profile implementation; the front door intentionally does not create a separate resume behavior. A supplied `--resume` flag is rejected with guidance to rerun the identical command.

`kaggle/launch.py` owns only the stable command surface, quantity normalization, registered `(model size, token budget)` profiles, dry-run inspection, and forwarding of stable action-specific controls. Experiment identities, immutable launch commits, checkpoint rules, W&B identities, qualification-plan selection, and publication verification remain in the profile-specific implementation modules.

Initially registered profiles are:

```text
20M / 100M
20M / 500M
20M / 2B
```

The historical `run_20m_*` and `build_and_push_*` modules remain implementation details while registered profiles depend on them. They are no longer the preferred commands for humans.

## Consequences

### Positive

- One command pattern covers dataset publication and training for all registered scaling experiments.
- Resume behavior is consistent: rerun the same command rather than remembering a separate mode.
- Future model/token experiments extend a registry instead of creating another human-facing launcher filename.
- The currently qualified 2B launch path is reused rather than rewritten before execution.
- `--dry-run` can validate profile resolution in ordinary CPU CI without importing CUDA/Kaggle backends.

### Negative or limiting

- The repository still contains profile-specific implementation modules behind the unified front door.
- Adding a profile requires both a qualified backend implementation and an explicit registry/test change.
- The unified CLI exposes only stable controls; backend-specific experimental flags should not automatically leak into the human command surface.

## Validation

- Ordinary tests must verify size normalization, supported profile resolution, train/publish dry-run dispatch, argument forwarding, unsupported-profile failure, and rejection of a separate `--resume` mode.
- The active 20M/2B runbook must use `kaggle/launch.py` for both VPS publication and Kaggle training.
- The 20M/2B dry-run must resolve to the existing qualified 2B training and publication modules without importing or executing them.

## Links

- [`../runbooks/unified_kaggle_launcher.md`](../runbooks/unified_kaggle_launcher.md)
- [`../runbooks/20m_2b_runbook.md`](../runbooks/20m_2b_runbook.md)
- `kaggle/launch.py`
- `tests/test_kaggle_launch.py`
