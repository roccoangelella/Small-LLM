---
status: current
last_reviewed: 2026-08-11
---

# Current project status

This file is high-freshness working memory. Detailed measurements and investigation chronology live in evidence/reference/archive and are linked rather than duplicated here.

## Completed 20M / 100M pretraining

```text
W&B run ID: 20m-100m-data-004
optimizer updates: 3,053
consumed training target tokens: 100,018,176
final validation loss: 4.252758495143203
final validation perplexity: 70.29906475797992
final checkpoint: step-00003053
```

The trajectory remained trainable but suffered a large late-run throughput collapse in the old adaptive PyTorch GDN-2 backend. Historical operating records are archived under `../archive/20m_100m/`; measured incidents/results are under `../evidence/20m_100m/`.

## Completed 20M / 500M pretraining

```text
W&B run ID: 20m-500m-data-001
final checkpoint: step-00015264
consumed training target tokens: 500,156,416
architecture: gdn2_hybrid
d_model: 256
d_ff: 704
layers: 8
context: 2,048
precision: FP16 autocast with FP32 master parameters
```

The accepted 500M checkpoint chain used the adaptive backend through step 4000 and mixed FLA for the accepted continuation. Treat throughput and very fine loss-curve comparisons across that boundary accordingly.

The frozen post-pretraining qualitative suite showed materially stronger answer-shaped/schema continuation than the earlier smoke-scale evidence, including stable Q/A and Alice/Ben surface structure, while factual/arithmetic answering and open-ended semantic stability remained weak. Under ADR 0027 this is enough evidence to keep the approximately-20M model fixed through the already-authorized 2B probe, not evidence for unlimited token scaling.

Canonical qualitative evidence: [`../evidence/20m/20m_500m_post_pretraining_full_suite_2026-08-10.md`](../evidence/20m/20m_500m_post_pretraining_full_suite_2026-08-10.md)

## Qualified GDN-2 production backend

Production CUDA execution is **mixed FLA on `fla-core==0.5.2`** under FP32 master parameters plus CUDA FP16 autocast.

```text
GPU qualified: Tesla T4 / SM75
PyTorch: 2.10.0+cu128
CUDA runtime: 12.8
Triton: 3.6.0
fla-core: 0.5.2
saved/configured gdn_chunk_size: 32
FLA internal runtime chunk: 64
```

The corrected deterministic qualification passed the requested synthetic decay sweep and the exact real step-4000 next-block forward/backward gate. Warmed true-block throughput measured 22,765.80 target tok/s for mixed FLA versus 1,964.75 target tok/s for the adaptive FP32 recurrence. The adaptive backend remains the correctness/reference fallback; full-FP32 FLA remains diagnostic/fallback.

Current contract: [`../reference/gdn2_fla_backend.md`](../reference/gdn2_fla_backend.md)

## Authorized next experiment — fresh 20M / 2B

ADR 0023 authorizes a new independent approximately-2B-token data-scaling trajectory. ADR 0027 adds the completed 500M qualitative result as justification for keeping model size fixed through this point.

```text
profile: 20m-2b-data-scaling-v1
dataset run ID: 20m-2b-dataset-001
W&B run ID: 20m-2b-data-001
target accepted source tokens: 2,000,000,000
minimum: 1,800,000,000
maximum: 2,200,000,000
producer durable checkpoint cadence: 80,000,000 source tokens
fresh initialization seed: 17
training microbatch: 4
training durability / validation / remote publication cadence: 250 updates
```

The 2B trajectory is fresh relative to 500M and the superseded 1B setup. It starts from seed 17 and uses the qualified mixed FLA CUDA backend from optimizer update 1. The exact optimizer-update count and WSD boundaries come from the completed verified manifest rather than a nominal hard-coded count.

Data path remains:

```text
pinned Nemotron-ClimbMix source
  -> VPS deterministic finite build
  -> immutable uint16 shards + verified Google Drive mirror
  -> private Kaggle publication + round-trip byte verification
  -> attached Kaggle dataset
  -> T4 training from Kaggle-local input
```

Current operational state: **the unified 2B publication/training launch surface is prepared on `main`; dataset production and optimizer update 1 have not yet been accepted as completed evidence.** Under ADR 0037 the finite dataset identity/geometry is now centralized in `dataset.qualification` (`--profile 20m-2b`) rather than repeated across per-budget producer/report wrappers.

Direct Kaggle execution via `python kaggle/launch.py ...` now explicitly adds the repository root to `sys.path` before profile resolution, so the consolidated launcher can import the top-level `dataset` package even when Python does not implicitly expose the checkout root.

## Frozen decisions still in force for the 2B probe

- Keep the pinned source revision, GPT-2 token IDs, and programming-cluster-11 exclusion policy.
- Keep context length 2,048 and the existing 20M `gdn2_hybrid` geometry.
- Keep saved/model `gdn_chunk_size=32`; CUDA FLA executes internal chunk 64.
- Keep adaptive PyTorch as correctness/reference fallback; do not change learned decay solely for runtime behavior.
- Use microbatch 4 on the qualified T4 path.
- Preserve fail-closed FP16 scaler/atomic-block behavior.
- Preserve `eval_core_v1` plus free-generation and teacher-forced confidence/rank diagnostics.
- New finite scaling trajectories start fresh unless a later ADR explicitly authorizes continuation.
- Revisit additional fixed-size token scaling and model enlargement after the frozen 2B comparison; SFT qualification is now authorized in parallel under ADR 0032.

## Authorized parallel SFT qualification

ADR 0032 authorizes the completed 20M/500M checkpoint as the SFT qualification parent and freezes the 4%-of-parent token scaling rule. ADR 0033 freezes the comprehensive post-SFT scorecard and the pretraining-equivalent T4 microbatch/cadence defaults.

The operational implementation is now present on `main` behind one canonical human-facing launcher:

```text
kaggle/launch_sft.py
  prepare
  publish
  train
  eval
  profiles
```

Current 500M-parent contract:

```text
parent checkpoint namespace: 20m-500m-dataset-001
verified parent consumed targets: 500,156,416
requested SFT loss-bearing targets: 20,006,256
overall mixture: 85% instruction / 15% ClimbMix replay
instruction allocation: 75 / 10 / 7.5 / 7.5
identity split: 95 / 2.5 / 2.5
optimizer target block: ~32,768 active targets
microbatch: 4
checkpoint / validation / publication cadence: 250 updates
```

The source preparation uses a pinned SmolTalk revision and source-independent prompt-family grouping, so aliases across source labels cannot cross the frozen identity split. Bundle verification binds the source provenance, split manifests, build reports, shard checksums, and target totals.

The SFT bundle can now be privately published to Kaggle and round-trip verified before training. Training independently recomputes the 4% budget from the verified parent checkpoint, uses fresh SFT optimizer/scheduler/scaler state, and automatically resumes from the newest valid local or remote SFT checkpoint while binding parent/data/template/objective identities.

The comprehensive evaluator compares parent and tuned checkpoints on unchanged `eval_core_v1`, base qualitative prompts, masked SFT validation/test loss, deterministic instruction behavior, EOS/runaway/leak/repetition diagnostics, per-category behavior, and parent-minus-tuned deltas without one arbitrary master score.

**Implementation status is operational but not yet accepted GPU evidence.** The remaining gate is repository test execution, private bundle publication/round-trip, bounded T4 FP16/mixed-FLA smoke, intentional exact resume, 250-update boundary validation, and fast/full post-SFT qualification on the 500M parent.

Current implementation reference: [`../reference/post_training_sft.md`](../reference/post_training_sft.md)
Current runbook: [`../runbooks/sft_s0_runbook.md`](../runbooks/sft_s0_runbook.md)

## Current source of truth

- Roadmap: [`roadmap.md`](roadmap.md)
- Durable decisions: [`../decisions/README.md`](../decisions/README.md)
- 2B decision: [`../decisions/0023-run-2b-20m-probe-via-vps-kaggle-dataset.md`](../decisions/0023-run-2b-20m-probe-via-vps-kaggle-dataset.md)
- 500M scaling interpretation: [`../decisions/0027-use-500m-schema-gains-to-justify-fixed-20m-token-scaling-through-2b.md`](../decisions/0027-use-500m-schema-gains-to-justify-fixed-20m-token-scaling-through-2b.md)
- Unified Kaggle runtime decision: [`../decisions/0030-consolidate-kaggle-profile-wrappers-behind-one-runtime.md`](../decisions/0030-consolidate-kaggle-profile-wrappers-behind-one-runtime.md)
- Dataset tooling consolidation: [`../decisions/0037-consolidate-dataset-profile-tools-and-retire-one-off-qualification-code.md`](../decisions/0037-consolidate-dataset-profile-tools-and-retire-one-off-qualification-code.md)
- Memory-governance decision: [`../decisions/0031-govern-project-memory-with-progressive-disclosure.md`](../decisions/0031-govern-project-memory-with-progressive-disclosure.md)
- SFT scaling decision: [`../decisions/0032-scale-sft-budget-with-pretraining-and-qualify-on-500m-first.md`](../decisions/0032-scale-sft-budget-with-pretraining-and-qualify-on-500m-first.md)
- SFT scorecard/cadence decision: [`../decisions/0033-use-comprehensive-post-sft-qualification-and-pretraining-cadence.md`](../decisions/0033-use-comprehensive-post-sft-qualification-and-pretraining-cadence.md)
- 2B runbook: [`../runbooks/20m_2b_runbook.md`](../runbooks/20m_2b_runbook.md)
- SFT runbook: [`../runbooks/sft_s0_runbook.md`](../runbooks/sft_s0_runbook.md)
- Dataset contract: [`../reference/dataset_and_tokenization.md`](../reference/dataset_and_tokenization.md)
- FLA backend contract: [`../reference/gdn2_fla_backend.md`](../reference/gdn2_fla_backend.md)
