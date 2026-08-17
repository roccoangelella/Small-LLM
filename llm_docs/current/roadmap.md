---
status: current
last_reviewed: 2026-08-17
---

# Current roadmap

## Completed scaling gates

The fresh 20M/2B data-scaling run is complete at `step-00061066` / 2,001,000,448 consumed targets. The 100M/2B Modal run is complete at final `step-00015267` / 2,001,000,448 consumed targets. The frozen `eval_core_v1` comparison against the 20M/500M endpoint is recorded in [`../evidence/scaling/20m_500m_20m_2b_100m_2b_full_eval_2026-08-13.md`](../evidence/scaling/20m_500m_20m_2b_100m_2b_full_eval_2026-08-13.md).

The intrinsic scaling result is clear: 20M still gains from 500M→2B, but unevenly; 100M/2B improves all retained clusters and all context-position buckets relative to 20M/2B. Treat 20M as capacity-constrained by the 2B endpoint unless later evidence overturns that interpretation.

## Active scaling gate — step-12,500 matched-decay probe

ADR 0090 supersedes ADR 0071's instruction to run `100m-10b-data-001` uncapped through all 76,294 updates. The current priority is to pause/terminate that full trajectory after preserving its existing state and use the exact Beam-local `step-00012500` checkpoint for a controlled counterfactual.

Run the temporary `100m-10b-decay-probe-step12500` branch with `beam/decay_probe_12500.py`. It must preserve the step-12,500 model, optimizer, scaler, RNG, and data cursor and change only the LR schedule. The probe uses the exact historical block-64 2B WSD schedule:

```text
peak LR:          3e-4
warmup tokens:    100,007,936
stable tokens:  1,499,987,968
decay tokens:     399,998,976
minimum LR ratio: 0.1
```

At step 12,500 / 1,638,400,000 consumed targets this schedule is already about 9.60% into cooldown, so the branch starts near LR `2.939e-4` and decays to `3e-5`. Run 2,759 additional block-64 updates, ending at step 15,259. The data remain the 10B corpus from the exact checkpoint cursor; the frozen 16-block validation prefix, microbatch 4, FP16, GDN-2 backend, and hybrid Muon+AdamW recipe remain unchanged.

The launcher fails closed if `step-00012500` is absent and CPU-stages/verifies the required dataset window before GPU allocation. It uses separate W&B and remote checkpoint namespaces so the original `100m-10b-data-001` state is not mutated.

After the probe completes, compare its validation trajectory and full `eval_core_v1` result against the completed 100M/2B endpoint. Do not reauthorize the remaining 10B stable-phase compute until that comparison is recorded.

## Completed engineering lane — 100M / 10B data path

The deterministic corpus is complete and verified in HF and Beam. Preserve these invariants during any 10B-corpus consumption:

- pinned ClimbMix source, tokenizer, cluster policy, and exact mixture;
- approximately-1-GiB immutable HF dataset shards;
- upload + independent remote hash verification before durable cursor/READY publication;
- frozen 16-block validation prefix;
- CPU staging establishes the checkpoint-aligned current+successor lead window before GPU allocation;
- the supported single-GPU Beam worker consumes exact block order and fails closed rather than skipping or reordering;
- single-GPU Beam topology with microbatch 4 and the frozen 64-sequence optimizer block;
- model checkpoints in the HF model repository and dataset shards in the HF dataset Storage Bucket.

The original full-run 76,294-update / 10,000,007,168-target ADR-0057 WSD contract remains historical/reproducible, but it is no longer authorized to continue automatically while ADR 0090 is the active scaling gate.

Operational full-run history: [`../runbooks/100m_10b_beam.md`](../runbooks/100m_10b_beam.md).

## Post-training lane

The first 20M/500M S0 SFT is behaviorally failed despite lower held-out SFT loss. The next SFT work must change or re-qualify the recipe explicitly; do not promote S0 unchanged by inertia. A 2B-parent or 100M-parent SFT run is not automatically authorized by the completed pretraining scaling comparison.

## Open decisions

- Does the step-12,500 matched-decay probe materially beat the completed 100M/2B endpoint after equivalent cooldown?
- If it does, what extended-training scheduler/horizon should replace the original fixed-fraction 10B WSD plan?
- If it does not, should the next scaling axis be model capacity, data quality/mixture, architecture, or post-training rather than more fixed-100M tokens?
- What controlled SFT recipe follows the failed S0 qualification?
- Which external standardized zero-shot tasks enter the first public scorecard?

## Frozen boundaries still in force

- New finite scaling trajectories start from fresh initialization unless a later ADR says otherwise; ADR 0090 is an explicit diagnostic fork exception.
- Context remains 2,048 for these comparisons.
- Production CUDA GDN-2 uses `fla-core==0.5.2`, saved chunk 32 / FLA internal chunk 64.
- Kaggle DDP evaluation does not change the single-GPU Beam training topology.
- New dataset durability uses HF Storage Buckets, not Google Drive.
- Stable model artifacts use the `models/...` namespace; live exact-resume checkpoints use `run/...`.
- Canonical qualitative comparison settings come from ADR 0025, not software sampling defaults.
