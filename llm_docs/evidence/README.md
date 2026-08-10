# Experiment evidence

Evidence records completed observations: measured results, verification reports, incidents, accepted checkpoint selection, and qualitative outputs. Preserve these files unchanged except for explicit factual corrections.

New evidence should normally use a scale/run-specific or topic-specific subdirectory and include exact model, dataset, tokenizer, code, checkpoint, and evaluation identities when applicable.

## Approximately-20M / 10M campaign

- [`20m/t4_first_qualification_result.md`](20m/t4_first_qualification_result.md)
- [`20m/20m_kaggle_preflight_results.md`](20m/20m_kaggle_preflight_results.md)
- [`20m/qualification_dataset_verification_2026-08-04.md`](20m/qualification_dataset_verification_2026-08-04.md)
- [`20m/20m_local_resume_results.md`](20m/20m_local_resume_results.md)
- [`20m/20m_repeatability_results.md`](20m/20m_repeatability_results.md)
- [`20m/20m_remote_recovery_attempt_20260805.md`](20m/20m_remote_recovery_attempt_20260805.md)
- [`20m/20m_remote_recovery_results.md`](20m/20m_remote_recovery_results.md)
- [`20m/20m_post_pretraining_checkpoint_selection.md`](20m/20m_post_pretraining_checkpoint_selection.md)
- [`20m/20m_post_pretraining_qualitative_results.md`](20m/20m_post_pretraining_qualitative_results.md)

## Approximately-20M / 100M campaign

- [`20m_100m/20m_100m_wandb_startup_2026-08-05.md`](20m_100m/20m_100m_wandb_startup_2026-08-05.md)
- [`20m_100m/validation_oom_step_500_2026-08-06.md`](20m_100m/validation_oom_step_500_2026-08-06.md)
- [`20m_100m/gdn2_nonfinite_step_1138_2026-08-06.md`](20m_100m/gdn2_nonfinite_step_1138_2026-08-06.md)
- [`20m_100m/fp16_overflow_step_1497_2026-08-06.md`](20m_100m/fp16_overflow_step_1497_2026-08-06.md)
- [`20m_100m/100m_wandb_final_result_2026-08-07.md`](20m_100m/100m_wandb_final_result_2026-08-07.md)

## Approximately-20M / 500M campaign

- [`20m_500m/qualification_report_dispatch_2026-08-07.md`](20m_500m/qualification_report_dispatch_2026-08-07.md)
- [`20m/20m_500m_post_pretraining_full_suite_2026-08-10.md`](20m/20m_500m_post_pretraining_full_suite_2026-08-10.md)

## GDN-2 / FLA qualification evidence

The FLA evidence predates the current preference for topic subdirectories, so the original file paths are retained to avoid rewriting immutable evidence/link history. The current interpretation is summarized in [`../reference/gdn2_fla_backend.md`](../reference/gdn2_fla_backend.md).

Key final evidence:

- [`gdn2_fla_corrected_oracle_and_step4000_qualification_2026-08-08.md`](gdn2_fla_corrected_oracle_and_step4000_qualification_2026-08-08.md)
- [`gdn2_fla_fp32_qualification_corrected_2026-08-08.json`](gdn2_fla_fp32_qualification_corrected_2026-08-08.json)
- [`gdn2_fla_step4000_parity_2026-08-08.json`](gdn2_fla_step4000_parity_2026-08-08.json)
- [`gdn2_fla_step4000_benchmark_2026-08-08.json`](gdn2_fla_step4000_benchmark_2026-08-08.json)

Historical failed/invalidated qualification evidence remains beside these final records and must not be deleted merely because the later interpretation changed.
