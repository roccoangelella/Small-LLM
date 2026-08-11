---
status: current
last_reviewed: 2026-08-11
---

# Current roadmap

## Active lane — complete the fresh 20M / 2B Kaggle scaling run

1. Keep the existing 20M / 2B experiment isolated under `20m-2b-dataset-001` / `20m-2b-data-001`.
2. Preserve its verified block-16 finite dataset identity and resume only from checkpoints whose manifests match that dataset.
3. Continue the fresh seed-17 trajectory with the qualified mixed FLA backend and the frozen one-pass WSD schedule.
4. Preserve the 250-update validation/local-checkpoint/verified-remote-publication cadence.
5. When the run completes, verify the final checkpoint, consumed-target-token count, dataset identity, WSD completion, and absence of unresolved non-finite events.

This lane remains the canonical same-20M data-scaling comparison. Do not retrofit its new Modal batch geometry into the already-running Kaggle trajectory.

## Authorized parallel lane — 100M / 2B on Modal

ADR 0041 authorizes the first approximately-100M-parameter / 2B-token pretraining trajectory on Modal before the 20M / 2B Kaggle run completes.

1. Derive `modal-2b-b64-dataset-001` byte-for-byte from the verified `20m-2b-dataset-001` corpus with `python -m dataset.reblock`; do not redownload or retokenize source data.
2. Keep context 2,048 and the exact stored sequence order/splits, but use 64 sequences per prepared optimizer block and a 32 MiB target shard size.
3. Upload the completed derived directory to the read-only `small-llm-data` Modal Volume.
4. Launch `100M` / `2B` on `H100` with the canonical `modal/launch.py` path.
5. Before optimizer step 1, probe real forward/backward execution at microbatch 16, 32, 48, and 64; reject OOM/non-finite/>90%-reserved-memory candidates and freeze the fastest safe measured candidate.
6. Keep FP16 autocast + FP32 master parameters, `gdn2_hybrid`, saved chunk 32 / FLA internal chunk 64, hybrid Muon + AdamW, seed 17, and manifest-derived one-pass WSD.
7. Keep checkpointing and validation every 250 successful optimizer updates plus the final checkpoint; checkpoints are durably stored in `small-llm-runs` and W&B remains online under stable run ID `100m-2b-data-001` with exact resume.
8. The reblocked 2B stream contains 976,560 training sequences, yielding 15,259 optimizer updates with a final 48-sequence block. Token-space WSD boundaries remain 100,007,936 warmup, 1,499,987,968 stable, and 399,998,976 decay tokens.

The optimizer batch change from 16 to 64 sequences is intentional for this new trajectory and is not treated as a same-batch continuation of the 20M scaling study.

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

## When the 20M / 2B run completes

1. Verify final checkpoint, consumed-target-token count, dataset identity, WSD completion, and absence of unresolved non-finite events.
2. Run the same frozen `eval_core_v1` fast/full suites plus teacher-forced confidence/rank and free-generation diagnostics.
3. Build a same-model 100M / 500M / 2B scorecard covering loss, perplexity, BPB, top-k accuracy, calibration, cluster slices, generation behavior, throughput, overflow behavior, and memory telemetry. The abandoned 1B profile is not a datapoint.
4. Fit/update the local data-scaling curve. Treat the 500M backend-migration boundary explicitly; the 2B run is the clean FLA-from-update-1 reference.
5. Keep the 4%-of-parent SFT scaling rule available for the 2B parent, but do not assume the unchanged S0 recipe has qualified behaviorally; resolve the open SFT recipe-selection decision first.
6. For the first accepted 500M-parent versus 2B-parent SFT comparison, keep architecture, template, objective, data split/mixture, microbatch, and cadence fixed unless a later ADR explicitly changes one.
7. Use the completed 20M evidence together with the separately authorized 100M / 2B Modal trajectory to decide the next scaling/architecture experiment; do not conflate their different optimizer-batch geometries.

## Explicitly frozen for the active 20M / 2B Kaggle experiment

- model geometry and 8-layer `gdn2_hybrid` architecture;
- context length 2,048;
- GPT-2 token IDs and current source/cluster policy;
- seed-17 fresh initialization policy;
- hybrid Muon + AdamW optimizer geometry;
- microbatch 4;
- 16-sequence optimizer block;
- saved `gdn_chunk_size=32` and qualified mixed FLA CUDA execution;
- one-pass WSD schedule semantics;
- finite prebuilt Kaggle dataset transport rather than live source streaming.

Do not combine that 20M scaling point with a new architecture, longer context, tokenizer change, data-mixture redesign, optimizer redesign, or continuation from the 500M terminal checkpoint. Those would destroy the intended same-model scaling comparison.

## Open decisions

- how to respond to the failed behavioral 500M-parent S0 qualification before the first 2B-parent SFT run;
- exact numerical SFT checkpoint-selection and base-retention gates, noting that the current 500M result has 0% instruction-behavior pass rate despite lower held-out SFT loss;
- whether the 20M data-scaling study is complete at approximately 96.9 source tokens per parameter after the 2B result;
- the model geometry after the authorized 100M `substantive` trajectory;
- whether a future larger model should keep the exact current GDN-2/MHA layer pattern or run controlled architecture ablations;
- whether fused SDPA/FlashAttention replacement of the current explicit MHA path should be qualified before the model size after 100M;
- which external zero-shot tasks enter the first stable public scorecard after intrinsic evaluation is accepted.

## Primary references

- [`../decisions/0023-run-2b-20m-probe-via-vps-kaggle-dataset.md`](../decisions/0023-run-2b-20m-probe-via-vps-kaggle-dataset.md)
- [`../decisions/0039-use-modal-for-future-gpu-training.md`](../decisions/0039-use-modal-for-future-gpu-training.md)
- [`../decisions/0041-use-block64-modal-corpus-and-probe-microbatch-16-32-48-64.md`](../decisions/0041-use-block64-modal-corpus-and-probe-microbatch-16-32-48-64.md)
- [`../decisions/0032-scale-sft-budget-with-pretraining-and-qualify-on-500m-first.md`](../decisions/0032-scale-sft-budget-with-pretraining-and-qualify-on-500m-first.md)
- [`../decisions/0033-use-comprehensive-post-sft-qualification-and-pretraining-cadence.md`](../decisions/0033-use-comprehensive-post-sft-qualification-and-pretraining-cadence.md)
- [`../reference/post_training_sft.md`](../reference/post_training_sft.md)
- [`../runbooks/sft_s0_runbook.md`](../runbooks/sft_s0_runbook.md)
- [`../runbooks/20m_2b_runbook.md`](../runbooks/20m_2b_runbook.md)
- [`../runbooks/modal_training_launcher.md`](../runbooks/modal_training_launcher.md)
- [`status.md`](status.md)
