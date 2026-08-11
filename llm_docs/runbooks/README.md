# Runbooks

Runbooks contain executable or intentionally reproducible procedures, prerequisites, expected artifacts, checks, and recovery steps. A document that is only a superseded plan belongs in archive instead.

## Active

- [`unified_kaggle_launcher.md`](unified_kaggle_launcher.md) — canonical `kaggle/launch.py` publication/training command surface for registered finite-data profiles.
- [`20m_2b_runbook.md`](20m_2b_runbook.md) — authorized fresh 20M-model / 2B-token scaling run.
- [`sft_s0_runbook.md`](sft_s0_runbook.md) — canonical `kaggle/launch_sft.py` bundle publication, T4 SFT, exact resume, and parent-versus-SFT qualification procedure.
- [`local_sft_chat.md`](local_sft_chat.md) — local interactive chat over a verified, completed Hugging Face SFT checkpoint.
- [`eval_core_v1_runbook.md`](eval_core_v1_runbook.md) — build, verify, fast, and full evaluation.
- [`post_pretraining_prompt_suite.md`](post_pretraining_prompt_suite.md) — canonical post-pretraining qualitative prompt procedure.
- [`100m_kagglehub_publication_suite.md`](100m_kagglehub_publication_suite.md) — reusable private finite-dataset publication procedure documented from the original 100M implementation.

## Completed-stage reproduction procedures

These are retained because they still describe reproducible completed experiment profiles using the current unified launcher. They are not authorization to change those completed experiments.

- [`20m_500m_runbook.md`](20m_500m_runbook.md) — completed 500M-stage reproduction/resume interpretation.
- [`20m_100m_runbook.md`](20m_100m_runbook.md) — completed 100M-stage reproduction procedure.

The obsolete 100M planning document moved to [`../archive/20m_100m/20m_100m_data_scaling_plan.md`](../archive/20m_100m/20m_100m_data_scaling_plan.md).

The unrun 1B procedure was superseded by ADR 0023 and is not an active run target. Git history preserves its former contents.

Completed 20M/10M qualification procedures live under `../archive/20m_qualification/`; their observed results are under `../evidence/20m/`.
