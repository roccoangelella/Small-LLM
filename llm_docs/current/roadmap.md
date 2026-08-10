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

## When the 2B run completes

1. Verify final checkpoint, consumed-target-token count, dataset identity, WSD completion, and absence of unresolved non-finite events.
2. Run the same frozen `eval_core_v1` fast/full suites plus teacher-forced confidence/rank and free-generation diagnostics.
3. Build a same-model 100M / 500M / 2B scorecard covering loss, perplexity, BPB, top-k accuracy, calibration, cluster slices, generation behavior, throughput, overflow behavior, and memory telemetry. The abandoned 1B profile is not a datapoint.
4. Fit/update the local data-scaling curve. Treat the 500M backend-migration boundary explicitly; the 2B run is the clean FLA-from-update-1 reference.
5. Use the 2B evidence to decide whether the 20M data-scaling study is complete and move to the next parameter scale and/or post-pretraining pipeline.

## Explicitly frozen for the 2B experiment

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

## Open decisions after the 2B evidence

- whether the 20M data-scaling study is complete at approximately 96.9 source tokens per parameter;
- exact next model size and geometry;
- whether the next engineering stage should be a larger pretraining run, a slim SFT/post-training qualification on the 20M model, or both in parallel;
- whether a future larger model should keep the exact current GDN-2/MHA layer pattern or run controlled architecture ablations;
- which external zero-shot tasks enter the first stable public scorecard after intrinsic evaluation is accepted.

## Primary references

- [`../decisions/0023-run-2b-20m-probe-via-vps-kaggle-dataset.md`](../decisions/0023-run-2b-20m-probe-via-vps-kaggle-dataset.md)
- [`../runbooks/20m_2b_runbook.md`](../runbooks/20m_2b_runbook.md)
- [`status.md`](status.md)
