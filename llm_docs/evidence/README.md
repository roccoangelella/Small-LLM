# Experiment evidence

Evidence records completed observations: measured results, verification reports, incidents, accepted checkpoint selection, and qualitative outputs. Preserve evidence unchanged except for explicit factual corrections; add later interpretation in a new record rather than rewriting old observations.

## Current scaling comparison

- [`scaling/100m_10b_step250_beam_fsync_resume_2026-08-14.md`](scaling/100m_10b_step250_beam_fsync_resume_2026-08-14.md) — first validation/checkpoint boundary, Beam distributed-Volume fsync hang, independently verified step-250 checkpoint, and exact continuation.
- [`scaling/100m_10b_beam_launch_2026-08-14.md`](scaling/100m_10b_beam_launch_2026-08-14.md) — live RTX5090 launch, startup incidents and fixes, microbatch-4 qualification, W&B identity, and first production progress.
- [`scaling/100m_10b_dataset_completion_2026-08-14.md`](scaling/100m_10b_dataset_completion_2026-08-14.md) — completed 10B producer state plus authenticated HF and Beam inventory verification.
- [`scaling/100m_2b_behavioral_qualification_2026-08-13.md`](scaling/100m_2b_behavioral_qualification_2026-08-13.md) — exact greedy-32 and supplementary sampled evidence used for the 100M/10B launch decision.
- [`scaling/20m_500m_20m_2b_100m_2b_full_eval_2026-08-13.md`](scaling/20m_500m_20m_2b_100m_2b_full_eval_2026-08-13.md) — same-`eval_core_v1` comparison of 20M/500M, 20M/2B, and 100M/2B, including the sampled-vs-greedy Paris clarification and qualitative-protocol boundary.
- [`scaling/100m_2b_sft_step250_nccl_timeout_2026-08-13.md`](scaling/100m_2b_sft_step250_nccl_timeout_2026-08-13.md) — microbatch-2 live SFT evidence through update 250 and the rank-asymmetric cadence timeout before the first checkpoint.

## Approximately-20M evidence

- [`20m/20m_kaggle_preflight_results.md`](20m/20m_kaggle_preflight_results.md)
- [`20m/qualification_dataset_verification_2026-08-04.md`](20m/qualification_dataset_verification_2026-08-04.md)
- [`20m/20m_local_resume_results.md`](20m/20m_local_resume_results.md)
- [`20m/20m_repeatability_results.md`](20m/20m_repeatability_results.md)
- [`20m/20m_500m_post_pretraining_full_suite_2026-08-10.md`](20m/20m_500m_post_pretraining_full_suite_2026-08-10.md)
- [`20m/20m_500m_sft_full_qualification_2026-08-11.md`](20m/20m_500m_sft_full_qualification_2026-08-11.md)
- [`20m/20m_2b_dual_t4_ddp_qualification_2026-08-12.md`](20m/20m_2b_dual_t4_ddp_qualification_2026-08-12.md)

## Approximately-20M / 100M-token campaign

- [`20m_100m/20m_100m_wandb_startup_2026-08-05.md`](20m_100m/20m_100m_wandb_startup_2026-08-05.md)
- [`20m_100m/validation_oom_step_500_2026-08-06.md`](20m_100m/validation_oom_step_500_2026-08-06.md)
- [`20m_100m/gdn2_nonfinite_step_1138_2026-08-06.md`](20m_100m/gdn2_nonfinite_step_1138_2026-08-06.md)
- [`20m_100m/fp16_overflow_step_1497_2026-08-06.md`](20m_100m/fp16_overflow_step_1497_2026-08-06.md)
- [`20m_100m/100m_wandb_final_result_2026-08-07.md`](20m_100m/100m_wandb_final_result_2026-08-07.md)

## GDN-2 / FLA qualification evidence

Current interpretation is summarized in [`../reference/gdn2_fla_backend.md`](../reference/gdn2_fla_backend.md). Key final evidence:

- [`gdn2_fla_corrected_oracle_and_step4000_qualification_2026-08-08.md`](gdn2_fla_corrected_oracle_and_step4000_qualification_2026-08-08.md)
- [`gdn2_fla_fp32_qualification_corrected_2026-08-08.json`](gdn2_fla_fp32_qualification_corrected_2026-08-08.json)
- [`gdn2_fla_step4000_parity_2026-08-08.json`](gdn2_fla_step4000_parity_2026-08-08.json)
- [`gdn2_fla_step4000_benchmark_2026-08-08.json`](gdn2_fla_step4000_benchmark_2026-08-08.json)

Historical failed/invalidated qualification evidence remains in place because it records what was observed at the time; current status/reference documents carry the corrected interpretation.
