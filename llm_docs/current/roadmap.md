---
status: current
last_reviewed: 2026-08-13
---

# Current roadmap

## Completed scaling gates

The fresh 20M/2B data-scaling run is complete at `step-00061066` / 2,001,000,448 consumed targets. The 100M/2B Modal run is complete at final `step-00015267` / 2,001,000,448 consumed targets. The frozen `eval_core_v1` comparison against the 20M/500M endpoint is recorded in [`../evidence/scaling/20m_500m_20m_2b_100m_2b_full_eval_2026-08-13.md`](../evidence/scaling/20m_500m_20m_2b_100m_2b_full_eval_2026-08-13.md).

The intrinsic scaling result is clear: 20M still gains from 500M→2B, but unevenly; 100M/2B improves all retained clusters and all context-position buckets relative to 20M/2B. Treat 20M as capacity-constrained by the 2B endpoint unless later evidence overturns that interpretation.

## Immediate scientific gate — exact behavioral qualification

Before representing ADR 0050's fresh-100M/10B launch trigger as satisfied:

1. Run the exact ADR-0025 canonical full qualitative protocol on the completed comparison endpoints: greedy decoding, seed 17, one sample, **global `max_new_tokens=32`**, no repetition penalty or decoding correction.
2. Keep the already-completed `eval_core_v1` intrinsic comparison as the quantitative base; do not rerun it merely to change generation length.
3. Keep sampled 100M/2B behavior separate from greedy behavior. The sampled run answered `Paris`; the greedy run answered `France`.
4. Judge the behavioral gate using the exact frozen prompt output together with termination/repetition and the intrinsic gains. Do not substitute loss/perplexity alone for the ADR-0050 gate.
5. Record an explicit gate decision before dispatching the fresh 100M/10B H100 scientific training run.

The current three full-eval JSON prompt sections are mutually comparable at `temperature=0/top_p=1/top_k=0`, but `trainer.eval_suite` used native per-case budgets and therefore did not reproduce ADR 0025's global 32-token cap.

## Parallel engineering lane — 100M / 10B data path

Technical qualification of the ADR-0058 incremental producer/consumer path may continue before the scientific H100 gate closes. Preserve these invariants:

- pinned ClimbMix source, tokenizer, cluster policy, and exact mixture;
- approximately-1-GiB immutable HF dataset shards;
- upload + independent remote hash verification before durable cursor/READY publication;
- frozen 16-block validation prefix;
- CPU producer/stager establishes the checkpoint-aligned current+successor lead window before H100 allocation;
- H100 consumes exact block order and waits rather than skips/reorders when the frontier is behind;
- single-H100 Modal topology;
- exact 76,294-update / 10,000,007,168-target horizon and ADR-0057 WSD schedule;
- model checkpoints in the HF model repository and dataset shards in the HF dataset Storage Bucket.

Operational runbook: [`../runbooks/100m_10b_incremental_modal.md`](../runbooks/100m_10b_incremental_modal.md).

## Post-training lane

The first 20M/500M S0 SFT is behaviorally failed despite lower held-out SFT loss. The next SFT work must change or re-qualify the recipe explicitly; do not promote S0 unchanged by inertia. A 2B-parent or 100M-parent SFT run is not automatically authorized by the completed pretraining scaling comparison.

## Open decisions

- Does the exact ADR-0025 100M/2B behavioral result satisfy ADR 0050 strongly enough to launch fresh 100M/10B training?
- If the 100M/10B run launches, does the approximately-5B intermediate checkpoint show enough behavioral growth to continue to the 10B endpoint?
- What controlled SFT recipe follows the failed S0 qualification?
- After the fixed-100M data-scaling lane is resolved, should the next axis be model size, architecture, data quality/mixture, or post-training?
- Which external standardized zero-shot tasks enter the first public scorecard?

## Frozen boundaries still in force

- New finite scaling trajectories start from fresh initialization unless a later ADR says otherwise.
- Context remains 2,048 for these comparisons.
- Production CUDA GDN-2 uses `fla-core==0.5.2`, saved chunk 32 / FLA internal chunk 64.
- Kaggle DDP does not imply Modal DDP; Modal remains one H100.
- New dataset durability uses HF Storage Buckets, not Google Drive.
- Stable model artifacts use the `models/...` namespace; live exact-resume checkpoints use `run/...`.
- Canonical qualitative comparison settings come from ADR 0025, not software sampling defaults.
