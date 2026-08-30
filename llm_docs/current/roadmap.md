---
status: current
last_reviewed: 2026-08-30
---

# Current roadmap

## Completed scaling gates

The fresh 20M/2B data-scaling run is complete at `step-00061066` / 2,001,000,448 consumed targets. The 100M/2B Modal run is complete at final `step-00015267` / 2,001,000,448 consumed targets. The frozen `eval_core_v1` comparison against the 20M/500M endpoint is recorded in [`../evidence/scaling/20m_500m_20m_2b_100m_2b_full_eval_2026-08-13.md`](../evidence/scaling/20m_500m_20m_2b_100m_2b_full_eval_2026-08-13.md).

The intrinsic scaling result is clear: 20M still gains from 500M→2B, but unevenly; 100M/2B improves all retained clusters and all context-position buckets relative to 20M/2B. Treat 20M as capacity-constrained by the 2B endpoint unless later evidence overturns that interpretation.

## Active scaling trajectory — deep-decay step-15,500 continuation through 10B

ADR 0114 supersedes ADR 0099's Kaggle execution choice while retaining ADR
0095's complete scientific schedule. ADR 0095 had already superseded ADR 0094,
ADR 0093, ADR 0092, and the original long flat-`3e-4` WSD trajectory. The main
continuation keeps the separate run ID
`100m-10b-deep-decay-from-step15500` and executes through
`modal run --detach modal/launch.py --action deep-decay --model 100M --tokens
10B` on one exact Modal H100.

Preserve the step-15,500 model ancestry, optimizer, scaler, RNG, data cursor,
exact 10B corpus order, frozen 16-block validation prefix, global 64-sequence
optimizer block, FP16, GDN-2, and hybrid Muon+AdamW. Modal resumes the newest
manifest-verified checkpoint in its own continuation namespace, currently
`step-00061500`; only an empty continuation namespace may fall back to the
exact original step 15,500. One H100 rewrites only execution slicing to
microbatch 16, giving four ordered accumulations per unchanged optimizer
update. For the prior two-T4 state, byte-identical rank-zero CUDA RNG bytes
become the single live device state while the original two-rank tree remains
in the hidden provider-migration backup.

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

The launcher must fail closed if neither a manifest-verified deep-decay
continuation checkpoint nor the exact original step-15,500 source is available.
It CPU-stages and verifies the checkpoint-aligned dataset window before H100
allocation and keeps local/W&B/HF checkpoint namespaces separate from the
original run and all superseded continuation branches. Modal publishes the live
continuation to HF every 250 successful updates and at a segment boundary. The
previous app stopped after the step-61,500 quota incident; this was a
durability-backend failure, not a training failure. ADR 0132 has now moved the
verified step-61,500 exact-resume tree to the mutable checkpoint Storage Bucket,
whose read-back `latest.json` resolves step 61,500. The historical best remains
step 59,250 at validation loss `2.8437069645151496`; those checkpoint bytes are
not currently retained, so the dedicated best-model repo remains absent rather
than pointing at a worse checkpoint. The previous detached app is no longer live,
and a new Modal invocation is blocked by the workspace spend limit. Once that
account-level blocker is cleared, the same launcher resumes from Bucket step
61,500 and future strict validation improvements populate the dedicated best repo.

Canonical procedure: [`../runbooks/100m_10b_deep_decay_modal.md`](../runbooks/100m_10b_deep_decay_modal.md).

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
- the active Modal worker consumes exact block order and fails closed rather than skipping or reordering;
- Modal keeps the frozen 64-sequence global optimizer block on one exact H100, with execution microbatch 16 and four ordered slices;
- rolling exact-resume `latest` checkpoints remain in the HF checkpoint Storage Bucket, strict validation-loss `best` remains in a dedicated recreate-on-improvement HF model repository, and dataset shards remain in the HF dataset Storage Bucket.

The original 76,294-update / 10,000,007,168-target ADR-0057 WSD contract remains historical/reproducible, but it is no longer authorized as the main continuation schedule under ADR 0099/0095.

Historical Beam full-run procedure: [`../runbooks/100m_10b_beam.md`](../runbooks/100m_10b_beam.md). Historical Kaggle deep-decay procedure: [`../runbooks/100m_10b_deep_decay_kaggle.md`](../runbooks/100m_10b_deep_decay_kaggle.md). Active deep-decay procedure: [`../runbooks/100m_10b_deep_decay_modal.md`](../runbooks/100m_10b_deep_decay_modal.md).

## Post-training lane

The 100M/2B R-SFT R0 12,306-row trajectory is complete at `step-00000361` under run ID `100m-2b-rsft-r0-12306-001`; it is the current accepted R-SFT chat artifact. The earlier atomic pilot, 10-epoch repeat probe, and textual pilot are historical experiment identities only and their Hugging Face run namespaces have been deleted.

The immediate post-training work is evaluation/behavioral inspection of this completed checkpoint, not another same-corpus retrain. Use the registered `chat.py --model_params 100M --num_tokens 2B --r-sft` path or an explicit matching `--run-id`. Any qualification result should be recorded as new evidence without mutating the completed trajectory.

The larger R-SFT corpus is complete under ADR 0106 and promoted to the standard Kaggle training input under ADR 0108. It contains 16,716 rows at SHA-256 `d13052b6fc33108ec65511b790a75f6473144855059b16b55167b046f787c405` (7,683 unchanged Superior rows, 8,403 unique simplified Superior rows, 630 Gemini anchors), with 70 collision exclusions. The verified 90/10 atomic bundle has 417 train blocks and 13,420,823 train targets. ADR 0111 permits explicit exact production repeats via `--num-epochs`; two epochs are 834 steps and default to `100m-2b-rsft-r0-16716-e2-001`, while one epoch retains `100m-2b-rsft-r0-16716-001`. Never resume `100m-2b-rsft-r0-12306-001` with this corpus. The intermediate 12,306-row corpus file is retired from the current tree, while its trained checkpoint remains the accepted chat/eval model until replacement training is qualified.

The first 20M/500M S0 behavioral failure remains historical evidence and does not override the now-completed 100M/2B S0→R-SFT trajectory.

## Open decisions

- How does the completed 12,306-row R-SFT R0 checkpoint behave under direct chat and the next frozen reasoning qualification suite, especially on atomic `<think>`/`<answer>` protocol use and generalization beyond the training templates?
- Does the deep-decay 10B continuation keep validation loss falling after step 17,789, or does the much steeper long-phase power law under-train later fresh data?
- How does the completed step-15,500 400M cooldown probe compare with the completed 100M/2B endpoint under frozen `eval_core_v1`?
- Which pre-terminal-cooldown checkpoint should be retained as the continuation anchor if training is later extended beyond 10B?
- What controlled SFT recipe follows the failed S0 qualification?
- Which external standardized zero-shot tasks enter the first public scorecard?

## Frozen boundaries still in force

- New finite scaling trajectories start from fresh initialization unless a later ADR says otherwise; ADRs 0091, 0099, and 0114 are explicit continuation/diagnostic exceptions.
- Context remains 2,048 for these comparisons.
- Production CUDA GDN-2 uses `fla-core==0.5.2`, saved chunk 32 / FLA internal chunk 64.
- The ADR-0114 Modal lane may change only execution slicing/topology while preserving the exact 64-sequence optimizer update and checkpointed scientific state.
- New dataset durability uses HF Storage Buckets, not Google Drive.
- Stable model artifacts use the `models/...` namespace; live exact-resume checkpoints use `run/...`.
- Canonical qualitative comparison settings come from ADR 0025, not software sampling defaults.
