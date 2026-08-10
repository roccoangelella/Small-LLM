# Runbooks

Runbooks contain executable procedures, prerequisites, expected artifacts, checks, and recovery steps.

## Active

- [`unified_kaggle_launcher.md`](unified_kaggle_launcher.md) — canonical `kaggle/launch.py` command surface for registered dataset publication and exact-resume training profiles.
- [`20m_2b_runbook.md`](20m_2b_runbook.md) — authorized fresh 20M-model / 2B-token scaling run; VPS build, private Kaggle publication, exact-resume T4 training, and final evaluation.
- [`eval_core_v1_runbook.md`](eval_core_v1_runbook.md) — build, verify, fast, and full evaluation.
- [`post_pretraining_prompt_suite.md`](post_pretraining_prompt_suite.md) — prompt-only compatibility procedure.
- [`100m_kagglehub_publication_suite.md`](100m_kagglehub_publication_suite.md) — reproducible private dataset publication process reused by the finite scaling publishers.

## Prior scaling-stage procedures

- [`20m_500m_runbook.md`](20m_500m_runbook.md) — completed 500M-stage build/training procedure; retained for exact historical operations and resume interpretation.
- [`20m_100m_runbook.md`](20m_100m_runbook.md) — completed 100M-stage training and reproduction procedure.
- [`20m_100m_data_scaling_plan.md`](20m_100m_data_scaling_plan.md) — historical planning-stage pointer; the original pre-final assumptions remain in Git history rather than as executable commands.

The unrun 1B procedure was superseded by ADR 0023 and is no longer an active runbook. Git history preserves its exact former contents.

Completed qualification procedures moved to `../archive/20m_qualification/`; their results are in `../evidence/20m/`.
