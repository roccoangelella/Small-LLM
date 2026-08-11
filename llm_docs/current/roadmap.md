---
status: current
last_reviewed: 2026-08-11
---

# Current roadmap

## Immediate gate — complete the fresh 20M / 2B scaling run

1. Keep the 2B experiment isolated under `20m-2b-dataset-001` / `20m-2b-data-001`.
2. Preserve the verified finite dataset identity and resume only from checkpoints whose manifests match that dataset.
3. Continue the fresh seed-17 trajectory with the qualified mixed FLA backend and the frozen one-pass WSD schedule.
4. Preserve the 250-update validation/local-checkpoint/verified-remote-publication cadence.
5. When the run completes, verify the final checkpoint, consumed-target-token count, dataset identity, WSD completion, and absence of unresolved non-finite events.

## Completed evidence gate — 500M-parent SFT qualification

The first frozen S0 qualification on the completed 20M/500M parent is complete. Canonical evidence is [`../evidence/20m/20m_500m_sft_full_qualification_2026-08-11.md`](../evidence/20m/20m_500m_sft_full_qualification_2026-08-11.md).

Observed result:

- masked SFT validation/test loss improved strongly;
- unchanged `eval_core_v1` regressed modestly and broadly;
- deterministic instruction behavior remained 0/30 passed;
- EOS termination remained 0%;
- runaway generation remained 100%;
- mean trigram repetition improved, but no instruction category acquired a passing case.

The evidence gate is therefore complete, but the recipe-selection gate is not. The next SFT decision must explicitly determine whether to revise the S0 data/objective/training recipe, change the behavior-selection criterion, or run a controlled follow-up before promoting an SFT recipe to the 2B parent. Do not infer success from held-out SFT loss alone.

## When the 2B run completes

1. Verify final checkpoint, consumed-target-token count, dataset identity, WSD completion, and absence of unresolved non-finite events.
2. Run the same frozen `eval_core_v1` fast/full suites plus teacher-forced confidence/rank and free-generation diagnostics.
3. Build a same-model 100M / 500M / 2B scorecard covering loss, perplexity, BPB, top-k accuracy, calibration, cluster slices, generation behavior, throughput, overflow behavior, and memory telemetry. The abandoned 1B profile is not a datapoint.
4. Fit/update the local data-scaling curve. Treat the 500M backend-migration boundary explicitly; the 2B run is the clean FLA-from-update-1 reference.
5. Keep the 4%-of-parent SFT scaling rule available for the 2B parent, but do not assume the unchanged S0 recipe has qualified behaviorally; resolve the open SFT recipe-selection decision first.
6. For the first accepted 500M-parent versus 2B-parent SFT comparison, keep architecture, template, objective, data split/mixture, microbatch, and cadence fixed unless a later ADR explicitly changes one.
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

- how to respond to the failed behavioral 500M-parent S0 qualification before the first 2B-parent SFT run;
- exact numerical SFT checkpoint-selection and base-retention gates, noting that the current 500M result has 0% instruction-behavior pass rate despite lower held-out SFT loss;
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
