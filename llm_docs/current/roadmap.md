---
status: current
last_reviewed: 2026-08-17
---

# Current roadmap

## Completed scaling gates

The fresh 20M/2B data-scaling run is complete at `step-00061066` / 2,001,000,448 consumed targets. The 100M/2B Modal run is complete at final `step-00015267` / 2,001,000,448 consumed targets. The frozen `eval_core_v1` comparison against the 20M/500M endpoint is recorded in [`../evidence/scaling/20m_500m_20m_2b_100m_2b_full_eval_2026-08-13.md`](../evidence/scaling/20m_500m_20m_2b_100m_2b_full_eval_2026-08-13.md).

The intrinsic scaling result is clear: 20M still gains from 500M→2B, but unevenly; 100M/2B improves all retained clusters and all context-position buckets relative to 20M/2B. Treat 20M as capacity-constrained by the 2B endpoint unless later evidence overturns that interpretation.

## Active scaling trajectory — aggressive step-15,500 continuation through 10B

ADR 0093 supersedes ADR 0092 and the original long flat-`3e-4` WSD trajectory. The main continuation must fork the exact uncooled `100m-10b-data-001/checkpoints/step-00015500` state and use `beam/aggressive_wsqd_10b_from_15500.py` under the separate run ID `100m-10b-aggressive-wsqd-from-step15500`.

Preserve the step-15,500 model, optimizer, scaler, RNG, data cursor, exact 10B corpus order, frozen 16-block validation prefix, microbatch 4, FP16, GDN-2, and hybrid Muon+AdamW. Change only the LR scheduler.

The accepted schedule is:

```text
source step:                 15,500
source targets:              2,031,616,000
source LR:                   3.0e-4

phase 1:                     cosine settle
settle span:                 300,023,808 targets / 2,289 updates
settle end step:             17,789
settle end targets:          2,331,639,808
settle end LR:               1.5e-4

phase 2:                     inverse-square-root base
formula:                     1.5e-4 * sqrt(2,331,639,808 / committed_targets)
cooldown start step:         73,242
cooldown start targets:      9,599,975,424
LR at cooldown start:        ~7.39243e-5

phase 3:                     linear terminal cooldown
cooldown span:               400,031,744 targets / 3,052 updates
terminal LR ratio:           0.05
final LR:                    1.5e-5
final step:                  76,294
final targets:               10,000,007,168
```

The first approximately 300M targets deliberately reduce LR faster than pure inverse-square-root decay. Around 500 updates after the fork this schedule is near `2.83e-4`, close to the successful diagnostic cooldown at the same point. It then stops the fast settle at `1.5e-4` and transitions into a long decreasing base so the remaining fresh data are not trained at near-terminal LR.

The launcher must fail closed if the exact original step-15,500 source is unavailable, CPU-stage and verify the checkpoint-aligned dataset window before GPU allocation, and keep local/W&B/HF checkpoint namespaces separate from both the original run and the cooldown probe.

## Active diagnostic — 400M cooldown fork

ADR 0091's `100m-10b-decay-probe-step15500` branch remains useful evidence and should be allowed to finish at step 18,552. Its early validation loss already fell below the original flat-LR trajectory within roughly 500 cooldown updates, motivating the more aggressive main schedule.

Do not promote the cooled step-18,552 endpoint into the long 10B run and do not reheat it. The 10B continuation always starts from the original uncooled step-15,500 checkpoint. After the probe finishes, run the frozen full evaluation and retain the result as schedule evidence.

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

The original 76,294-update / 10,000,007,168-target ADR-0057 WSD contract remains historical/reproducible, but it is no longer authorized as the main continuation schedule under ADR 0093.

Operational full-run history: [`../runbooks/100m_10b_beam.md`](../runbooks/100m_10b_beam.md).

## Post-training lane

The first 20M/500M S0 SFT is behaviorally failed despite lower held-out SFT loss. The next SFT work must change or re-qualify the recipe explicitly; do not promote S0 unchanged by inertia. A 2B-parent or 100M-parent SFT run is not automatically authorized by the completed pretraining scaling comparison.

## Open decisions

- How does the completed step-15,500 400M cooldown probe compare with the completed 100M/2B endpoint under frozen `eval_core_v1`?
- Does the aggressive 10B continuation retain its advantage after the 300M settling phase and through the long inverse-square-root middle?
- Which pre-terminal-cooldown checkpoint should be retained as the continuation anchor if training is later extended beyond 10B?
- What controlled SFT recipe follows the failed S0 qualification?
- Which external standardized zero-shot tasks enter the first public scorecard?

## Frozen boundaries still in force

- New finite scaling trajectories start from fresh initialization unless a later ADR says otherwise; ADRs 0091 and 0093 are explicit continuation/diagnostic exceptions.
- Context remains 2,048 for these comparisons.
- Production CUDA GDN-2 uses `fla-core==0.5.2`, saved chunk 32 / FLA internal chunk 64.
- Kaggle DDP evaluation does not change the single-GPU Beam training topology.
- New dataset durability uses HF Storage Buckets, not Google Drive.
- Stable model artifacts use the `models/...` namespace; live exact-resume checkpoints use `run/...`.
- Canonical qualitative comparison settings come from ADR 0025, not software sampling defaults.
