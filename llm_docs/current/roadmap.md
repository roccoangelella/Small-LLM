---
status: current
last_reviewed: 2026-08-10
---

# Current roadmap

## Immediate gate — launch the fresh 20M / 2B scaling run

1. Keep the 2B experiment isolated under `20m-2b-dataset-001` / `20m-2b-data-001`.
2. Build the complete finite dataset on the VPS with the fixed 2B profile; do not live-stream source data during GPU training.
3. Require full local verification, Google Drive durability agreement, private Kaggle publication, fresh round-trip byte verification, and denied anonymous access.
4. Attach only the verified 2B dataset version to the T4 notebook.
5. Start a fresh seed-17 trajectory at microbatch 4 with qualified mixed FLA GDN-2 on CUDA from update 1.
6. Preserve the exact one-pass WSD schedule derived from the completed manifest and the 250-update validation/local-checkpoint/verified-remote-publication cadence.
7. If Kaggle interrupts the job, resume only from a verified `20m-2b-dataset-001` checkpoint whose Drive manifest matches the attached dataset.

## Parallel gate — qualify SFT on the completed 500M checkpoint

ADR 0032 authorizes using the completed 500M checkpoint as the SFT qualification parent while the fresh 2B pretraining trajectory runs. ADR 0033 freezes the comprehensive parent-versus-SFT scorecard and pretraining-equivalent T4 operations. The implementation work is now landed; the active gate is measured qualification.

1. Run the repository SFT unit/integration suite, including global cross-source split identity, masked-target normalization, dynamic microbatch cropping, held-out test evaluation, nested scorecard deltas, checkpoint identity, and behavior-format tests.
2. Build and privately publish the 500M-parent SFT bundle through `kaggle/launch_sft.py publish`; require a byte-identical Kaggle round trip, full bundle re-verification, and denied anonymous access.
3. Keep the frozen S0 data contract during qualification: 85% instruction / 15% ClimbMix replay, 75/10/7.5/7.5 instruction allocation, and identity-safe 95% train / 2.5% validation / 2.5% test.
4. Require the trainer to recompute the exact 4% budget from the verified parent counter before optimizer update 1. For the 500M parent the requested ceiling is 20,006,256 loss-bearing targets.
5. Run a bounded T4 CUDA FP16/mixed-FLA smoke at microbatch 4 and verify finite loss/gradients, stable VRAM, W&B identity, local save, and remote publication.
6. Rerun the identical bounded command to prove automatic recovery from the newest valid local/remote checkpoint and exact continuation at the next immutable block.
7. Run through at least one 250-update validation/local-checkpoint/remote-publication boundary before treating the operational cadence as qualified.
8. Complete the 500M-parent trajectory and run comprehensive `fast` then `full` parent-versus-SFT scorecards. Use the observed instruction-gain/base-retention curve to freeze numerical selection/retention gates before selecting the first 2B-parent SFT output.

## When the 2B run completes

1. Verify final checkpoint, consumed-target-token count, dataset identity, WSD completion, and absence of unresolved non-finite events.
2. Run the same frozen `eval_core_v1` fast/full suites plus teacher-forced confidence/rank and free-generation diagnostics.
3. Build a same-model 100M / 500M / 2B scorecard covering loss, perplexity, BPB, top-k accuracy, calibration, cluster slices, generation behavior, throughput, overflow behavior, and memory telemetry. The abandoned 1B profile is not a datapoint.
4. Fit/update the local data-scaling curve. Treat the 500M backend-migration boundary explicitly; the 2B run is the clean FLA-from-update-1 reference.
5. After the 500M SFT lane has passed its operational gates, publish a new 2B-parent SFT bundle using exactly 4% of the verified completed 2B parent counter and switch the same `launch_sft.py` lane to `--tokens 2B`.
6. Keep architecture, template, objective, data split/mixture, microbatch, and cadence fixed for the first 500M-parent versus 2B-parent SFT comparison unless a later ADR explicitly changes one.
7. Use the 2B evidence to decide whether the 20M data-scaling study is complete and what parameter scale follows.

## Explicitly frozen for the 2B pretraining experiment

- model geometry and 8-layer `gdn2_hybrid` architecture;
- context length 2,048;
- GPT-2 token IDs and current source/cluster policy;
- seed-17 fresh initialization policy;
- hybrid Muon + AdamW optimizer geometry;
- microbatch 4;
- saved `gdn_chunk_size=32` and qualified mixed FLA CUDA execution;
- one-pass WSD schedule semantics;
- finite prebuilt Kaggle dataset transport rather than live source streaming.

Do not combine the 2B scaling point with a new architecture, longer context, tokenizer change, data-mixture redesign, optimizer redesign, or continuation from the 500M terminal checkpoint. Those would destroy the intended same-model scaling comparison.

## Open decisions

- exact numerical SFT checkpoint-selection and base-retention gates after the 500M qualification evidence;
- whether the 20M data-scaling study is complete at approximately 96.9 source tokens per parameter after the 2B result;
- exact next model size and geometry;
- whether a future larger model should keep the exact current GDN-2/MHA layer pattern or run controlled architecture ablations;
- which external zero-shot tasks enter the first stable public scorecard after intrinsic evaluation is accepted.

## Primary references

- [`../decisions/0023-run-2b-20m-probe-via-vps-kaggle-dataset.md`](../decisions/0023-run-2b-20m-probe-via-vps-kaggle-dataset.md)
- [`../decisions/0032-scale-sft-budget-with-pretraining-and-qualify-on-500m-first.md`](../decisions/0032-scale-sft-budget-with-pretraining-and-qualify-on-500m-first.md)
- [`../decisions/0033-use-comprehensive-post-sft-qualification-and-pretraining-cadence.md`](../decisions/0033-use-comprehensive-post-sft-qualification-and-pretraining-cadence.md)
- [`../reference/post_training_sft.md`](../reference/post_training_sft.md)
- [`../runbooks/sft_s0_runbook.md`](../runbooks/sft_s0_runbook.md)
- [`../runbooks/20m_2b_runbook.md`](../runbooks/20m_2b_runbook.md)
- [`status.md`](status.md)
