# Runbooks

Runbooks contain executable or intentionally reproducible procedures, prerequisites, expected artifacts, checks, and recovery steps. Completed-run runbooks are retained for reproduction and are not authorization to change finished experiments.

## Active

- [`100m_10b_deep_decay_modal.md`](100m_10b_deep_decay_modal.md) — active ADR-0114 one-H100 exact-resume procedure for the frozen step-15,500 deep-decay continuation.
- [`100m_10b_beam.md`](100m_10b_beam.md) — authorized full 100M/10B RTX5090 launch, concurrent approximately-5B Kaggle capture, and exact-resume procedure.
- [`100m_10b_incremental_modal.md`](100m_10b_incremental_modal.md) — ADR-0058 CPU producer/frontier/staging/H100 procedure for the conditional fresh 100M/10B trajectory; technical readiness is separate from ADR-0050 launch authorization.
- [`unified_kaggle_launcher.md`](unified_kaggle_launcher.md) — canonical finite-profile Kaggle publication/training command surface.
- [`sft_s0_runbook.md`](sft_s0_runbook.md) — SFT bundle publication/training/qualification procedure; the original S0 behavioral recipe is not promoted by its failed qualification.
- [`rsft_r0_atomic_production.md`](rsft_r0_atomic_production.md) — current 100M/2B R-SFT R0 checkpoint, chat, Hugging Face state, and reproduction procedure.
- [`local_sft_chat.md`](local_sft_chat.md) — local interactive chat over verified completed pretrained, SFT, or R-SFT artifacts.
- [`eval_core_v1_runbook.md`](eval_core_v1_runbook.md) — frozen intrinsic evaluation workflow and stable/live checkpoint transport.
- [`post_pretraining_prompt_suite.md`](post_pretraining_prompt_suite.md) — exact ADR-0025 qualitative comparison and teacher-forced confidence diagnostic.
- [`modal_training_launcher.md`](modal_training_launcher.md) — current Modal launch/checkpoint operation.
- [`100m_kagglehub_publication_suite.md`](100m_kagglehub_publication_suite.md) — reusable private finite-dataset Kaggle publication procedure.

## Completed-stage reproduction procedures

- [`20m_2b_runbook.md`](20m_2b_runbook.md) — completed 20M/2B data-scaling endpoint and reproduction/resume interpretation.
- [`20m_500m_runbook.md`](20m_500m_runbook.md) — completed 20M/500M stage.
- [`20m_100m_runbook.md`](20m_100m_runbook.md) — completed 20M/100M stage.

Completed 20M/10M qualification procedures live under `../archive/20m_qualification/`; observed results are under `../evidence/20m/`. The unrun 1B profile was superseded by ADR 0023 and is preserved only in history.
