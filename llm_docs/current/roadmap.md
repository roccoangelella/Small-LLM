---
status: current
last_reviewed: 2026-08-14
---

# Current roadmap

## Completed scaling gates

The fresh 20M/2B data-scaling run is complete at `step-00061066` / 2,001,000,448 consumed targets. The 100M/2B Modal run is complete at final `step-00015267` / 2,001,000,448 consumed targets. The frozen `eval_core_v1` comparison against the 20M/500M endpoint is recorded in [`../evidence/scaling/20m_500m_20m_2b_100m_2b_full_eval_2026-08-13.md`](../evidence/scaling/20m_500m_20m_2b_100m_2b_full_eval_2026-08-13.md).

The intrinsic scaling result is clear: 20M still gains from 500M→2B, but unevenly; 100M/2B improves all retained clusters and all context-position buckets relative to 20M/2B. Treat 20M as capacity-constrained by the 2B endpoint unless later evidence overturns that interpretation.

## Active scaling trajectory — full 100M / 10B

The exact ADR-0025 comparison is complete and ADR 0071 closes the launch gate.
The fresh `100m-10b-data-001` Beam trajectory is active from source commit
`1f9dff920ecc45ce2fdb43fd875514a18391273d`. After repeated RTX5090 worker and
startup failures, the current supported RTX4090 segment resumed exactly from
the independently verified HF `step-00003000`; keep it running through all
76,294 updates without `--max-steps-this-session`. The earlier exact step-250
infrastructure-only resume from launch source `42b0376` remains recorded
checkpoint ancestry.

Keep these immediate checks:

1. Preserve exact resume from source commit `1f9dff920ecc45ce2fdb43fd875514a18391273d` and microbatch 4.
   RTX4090 is the current infrastructure failover; `42b0376` is the recorded
   one-time resume parent only.
2. Watch finite loss, gradient, throughput, and overflow telemetry in W&B.
3. Confirm the next Beam-local durability boundary and rolling HF publication
   at step 4,500.
4. Capture `step-00038000` / 4,980,736,000 targets for concurrent Kaggle
   qualification before the rolling HF pointer advances to step 38,500.
5. Do not pause or terminate Beam based on that intermediate result; qualify the
   terminal 10B endpoint after completion.

## Completed engineering lane — 100M / 10B data path

The deterministic corpus is complete and verified in HF and Beam. Preserve these invariants during consumption:

- pinned ClimbMix source, tokenizer, cluster policy, and exact mixture;
- approximately-1-GiB immutable HF dataset shards;
- upload + independent remote hash verification before durable cursor/READY publication;
- frozen 16-block validation prefix;
- CPU producer/stager establishes the checkpoint-aligned current+successor lead window before H100 allocation;
- the supported single-GPU Beam worker consumes exact block order and fails closed rather than skipping or
  reordering if preseeded Beam bytes are missing;
- single-GPU Beam topology with qualified microbatch 4 and the frozen
  64-sequence optimizer block;
- exact 76,294-update / 10,000,007,168-target horizon and ADR-0057 WSD schedule;
- model checkpoints in the HF model repository and dataset shards in the HF dataset Storage Bucket.

Operational runbook: [`../runbooks/100m_10b_beam.md`](../runbooks/100m_10b_beam.md).

## Post-training lane

The first 20M/500M S0 SFT is behaviorally failed despite lower held-out SFT loss. The next SFT work must change or re-qualify the recipe explicitly; do not promote S0 unchanged by inertia. A 2B-parent or 100M-parent SFT run is not automatically authorized by the completed pretraining scaling comparison.

## Open decisions

- What does the concurrent approximately-5B checkpoint show relative to the
  completed 100M/2B endpoint?
- What controlled SFT recipe follows the failed S0 qualification?
- After the fixed-100M data-scaling lane is resolved, should the next axis be model size, architecture, data quality/mixture, or post-training?
- Which external standardized zero-shot tasks enter the first public scorecard?

## Frozen boundaries still in force

- New finite scaling trajectories start from fresh initialization unless a later ADR says otherwise.
- Context remains 2,048 for these comparisons.
- Production CUDA GDN-2 uses `fla-core==0.5.2`, saved chunk 32 / FLA internal chunk 64.
- Kaggle DDP evaluation does not change the single-GPU Beam training topology.
- New dataset durability uses HF Storage Buckets, not Google Drive.
- Stable model artifacts use the `models/...` namespace; live exact-resume checkpoints use `run/...`.
- Canonical qualitative comparison settings come from ADR 0025, not software sampling defaults.
