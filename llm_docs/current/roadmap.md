---
status: current
last_reviewed: 2026-08-19
---

# Current roadmap

## Completed scaling gates

The fresh 20M/2B data-scaling run is complete at `step-00061066` / 2,001,000,448 consumed targets. The 100M/2B Modal run is complete at final `step-00015267` / 2,001,000,448 consumed targets. The frozen `eval_core_v1` comparison against the 20M/500M endpoint is recorded in [`../evidence/scaling/20m_500m_20m_2b_100m_2b_full_eval_2026-08-13.md`](../evidence/scaling/20m_500m_20m_2b_100m_2b_full_eval_2026-08-13.md).

The intrinsic scaling result is clear: 20M still gains from 500M→2B, but unevenly; 100M/2B improves all retained clusters and all context-position buckets relative to 20M/2B. Treat 20M as capacity-constrained by the 2B endpoint unless later evidence overturns that interpretation.

## Active scaling trajectory — deep-decay step-15,500 continuation through 10B

ADR 0099 supersedes ADR 0095's Beam execution choice while retaining its complete scientific schedule; ADR 0095 had already superseded ADR 0094, ADR 0093, ADR 0092, and the original long flat-`3e-4` WSD trajectory. The main continuation must fork the exact uncooled `100m-10b-data-001/checkpoints/step-00015500` state under the separate run ID `100m-10b-deep-decay-from-step15500` and execute through `python kaggle/launch.py deep-decay --model 100M --tokens 10B` on two Kaggle Tesla T4 GPUs.

Preserve the step-15,500 model, optimizer, scaler, RNG, data cursor, exact 10B corpus order, frozen 16-block validation prefix, global 64-sequence optimizer block, FP16, GDN-2, and hybrid Muon+AdamW. The source checkpoint used execution microbatch 4; Kaggle rewrites only that execution-slicing field to microbatch 2 because microbatch 4 OOMed on the 100M/T4 path while microbatch 2 completed 250 real updates with material headroom. Kaggle DDP splits each ordered optimizer block 32/32 across the two ranks, giving sixteen local microbatches per rank while preserving one 64-sequence optimizer update.

The accepted schedule is:

```text
source step:                 15,500
source targets:              2,031,616,000
source LR:                   3.0e-4

phase 1:                     cosine settle
settle span:                 300,023,808 targets / 2,289 updates
settle end step:             17,789
settle end targets:          2,331,639,808
settle end LR:               1.0e-4

phase 2:                     calibrated power-law base
formula:                     1.0e-4 * (2,331,639,808 / committed_targets)^p
p:                           ~1.6270515945
cooldown start step:         73,242
cooldown start targets:      9,599,975,424
LR at cooldown start:        1.0e-5

phase 3:                     linear terminal cooldown
cooldown span:               400,031,744 targets / 3,052 updates
cooldown start LR:           1.0e-5
final LR:                    5.0e-6
final step:                  76,294
final targets:               10,000,007,168
```

The calibrated phase-2 exponent is chosen from the exact `1e-4 -> 1e-5` endpoint requirement rather than from a generic inverse-square-root default. This is materially steeper than the approximately `0.5` power used by standard WSqD-like continuation schedules and is intentionally project-specific evidence-driven experimentation. Approximate phase-2 landmarks are `8.26e-5` at step 20,000, `5.75e-5` at step 25,000, `4.27e-5` at step 30,000, `2.68e-5` at step 40,000, `1.86e-5` at step 50,000, `1.38e-5` at step 60,000, and `1.08e-5` at step 70,000.

The launcher must fail closed if neither a manifest-verified Kaggle deep-decay continuation checkpoint nor the exact original step-15,500 source is available. It must CPU-stage and verify the checkpoint-aligned dataset window before training and keep local/W&B/HF checkpoint namespaces separate from the original run and all superseded continuation branches. Kaggle publishes the live continuation to HF every 250 successful updates. The first bounded 250-update dual-T4 segment is the live block-64 pretraining execution gate before relying on long unattended notebook segments.

Canonical procedure: [`../runbooks/100m_10b_deep_decay_kaggle.md`](../runbooks/100m_10b_deep_decay_kaggle.md).

## Active diagnostics — earlier post-15,500 branches

ADR 0091's `100m-10b-decay-probe-step15500` branch remains useful schedule evidence. The ADR-0093 `100m-10b-aggressive-wsqd-from-step15500` branch showed validation loss falling during its fast settle and then rising after transition to the gentler long phase. ADR 0094 is also historical schedule evidence but is no longer the authorized main trajectory.

Do not promote any diagnostic/older continuation checkpoint into the new 10B run and do not reheat a cooled model. The active continuation always starts from the original uncooled step-15,500 checkpoint unless it is resuming its own manifest-verified Kaggle deep-decay namespace.

## Completed engineering lane — 100M / 10B data path

The deterministic corpus is complete and verified in HF and Beam. Preserve these invariants during any 10B-corpus consumption:

- pinned ClimbMix source, tokenizer, cluster policy, and exact mixture;
- approximately-1-GiB immutable HF dataset shards;
- upload + independent remote hash verification before durable cursor/READY publication;
- frozen 16-block validation prefix;
- CPU staging establishes the checkpoint-aligned current+successor lead window before GPU work;
- the active Kaggle worker consumes exact block order and fails closed rather than skipping or reordering;
- Kaggle exact-batch DDP keeps the frozen 64-sequence global optimizer block, split 32/32 across two T4 ranks, with execution microbatch 2;
- model checkpoints remain in the HF model repository and dataset shards remain in the HF dataset Storage Bucket.

The original 76,294-update / 10,000,007,168-target ADR-0057 WSD contract remains historical/reproducible, but it is no longer authorized as the main continuation schedule under ADR 0099/0095.

Historical Beam full-run procedure: [`../runbooks/100m_10b_beam.md`](../runbooks/100m_10b_beam.md). Active deep-decay procedure: [`../runbooks/100m_10b_deep_decay_kaggle.md`](../runbooks/100m_10b_deep_decay_kaggle.md).

## Post-training lane

The 100M/2B R-SFT R0 12,306-row trajectory is complete at `step-00000361` under run ID `100m-2b-rsft-r0-12306-001`; it is the current accepted R-SFT chat artifact. The earlier atomic pilot, 10-epoch repeat probe, and textual pilot are historical experiment identities only and their Hugging Face run namespaces have been deleted.

The immediate post-training work is evaluation/behavioral inspection of this completed checkpoint, not another same-corpus retrain. Use the registered `chat.py --model_params 100M --num_tokens 2B --r-sft` path or an explicit matching `--run-id`. Any qualification result should be recorded as new evidence without mutating the completed trajectory.

The larger R-SFT corpus is now an active quota-limited lane under ADR 0106. Preserve the 1,122 historical accepted batches and the v1 curation for the completed model, but use expansion curation v2 (8,473 keepers) for all new work. The keeper-only resume starts from 4,009 still-valid old keep rewrites and adapts only the 4,464 missing keepers with the selected Gemini Variant-D compressor. GemRouter must remain hard Gemini-only (`GEMROUTER_NVIDIA_ENABLED=false`, backend order exactly `gemini-api`, fallback disabled). Continue resumably across daily quota windows; after every v2 keeper has an accepted rewrite, deduplicate normalized prompts against the 7,683-row baseline and one another, freeze the exact expanded corpus, build/verify the 90/10 atomic bundle, and assign a new run identity rather than resuming `100m-2b-rsft-r0-12306-001`.

The first 20M/500M S0 behavioral failure remains historical evidence and does not override the now-completed 100M/2B S0→R-SFT trajectory.

## Open decisions

- How does the completed 12,306-row R-SFT R0 checkpoint behave under direct chat and the next frozen reasoning qualification suite, especially on atomic `<think>`/`<answer>` protocol use and generalization beyond the training templates?
- What exact final row count remains after all 8,473 curation-v2 keepers are adapted and normalized-prompt collisions are removed?
- Does the deep-decay 10B continuation keep validation loss falling after step 17,789, or does the much steeper long-phase power law under-train later fresh data?
- How does the completed step-15,500 400M cooldown probe compare with the completed 100M/2B endpoint under frozen `eval_core_v1`?
- Which pre-terminal-cooldown checkpoint should be retained as the continuation anchor if training is later extended beyond 10B?
- What controlled SFT recipe follows the failed S0 qualification?
- Which external standardized zero-shot tasks enter the first public scorecard?

## Frozen boundaries still in force

- New finite scaling trajectories start from fresh initialization unless a later ADR says otherwise; ADRs 0091 and 0099 are explicit continuation/diagnostic exceptions.
- Context remains 2,048 for these comparisons.
- Production CUDA GDN-2 uses `fla-core==0.5.2`, saved chunk 32 / FLA internal chunk 64.
- Kaggle dual-T4 execution may replace single-GPU Beam only when the exact 64-sequence global optimizer update and topology-neutral checkpoint semantics are preserved as in ADR 0099.
- New dataset durability uses HF Storage Buckets, not Google Drive.
- Stable model artifacts use the `models/...` namespace; live exact-resume checkpoints use `run/...`.
- Canonical qualitative comparison settings come from ADR 0025, not software sampling defaults.
