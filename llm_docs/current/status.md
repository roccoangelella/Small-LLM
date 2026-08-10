---
status: current
last_reviewed: 2026-08-10
---

# Current project status

## Completed 20M / 100M pretraining experiment

The approximately-20M-parameter GDN-2 hybrid completed its fixed approximately-100M-token pretraining schedule.

Canonical final evidence:

```text
W&B run ID: 20m-100m-data-004
optimizer updates: 3,053
consumed training target tokens: 100,018,176
final validation loss: 4.252758495143203
final validation perplexity: 70.29906475797992
final checkpoint: step-00003053
```

The run stayed trainable but suffered a large late-run throughput collapse in the old adaptive PyTorch GDN-2 backend. That runtime problem motivated the later FLA qualification work.

## Completed 20M / 500M pretraining experiment

The independent seed-17 500M trajectory has completed.

Latest final checkpoint metadata supplied by the completed run/prompt-suite handoff:

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

The 500M trajectory is not a clean single-backend curve: the accepted checkpoint chain reached step 4000 under the adaptive backend, then production continuation was authorized with the mathematically compatible mixed FLA GDN-2 CUDA backend. Interpret throughput and very fine-grained loss-curve discontinuities with that migration boundary in mind.

### Canonical full-suite qualitative result

The validation-selected `best` checkpoint (`step-00015264`) completed the frozen deterministic full qualitative suite on CUDA/FP16 with greedy decoding and a 32-token cap.

The model clearly learned English surface structure, local lexical associations, dialogue/Q&A formatting, and visible text schemas, but open-ended capability remains weak at this scale. The run showed pervasive semantic drift, tautological restatement, and greedy repetition; the `Germany |` structured relation collapsed to repeated `Rome |`; and under a strict direct-answer reading none of the 12 simple factual/arithmetic probes contained the expected answer (`0 / 12`). Several Q&A cases reproduced the `Question: ... Answer:` format instead of supplying the requested fact.

The qualitative result is nevertheless materially encouraging for continued data scaling. Compared with the earlier smoke-scale evidence, the 500M checkpoint more consistently produces answer-shaped continuations, preserves the Alice/Ben speaker schema, and emits a plausible sentiment class before later degeneration. The early checkpoint had already shown some Q/A surface-format imitation, so the important longitudinal signal is stronger schema continuation rather than the first appearance of Q/A syntax.

Project interpretation under ADR 0027: the combination of continued validation improvement and stronger conditional text-schema behavior is sufficient evidence to keep the approximately-20M model size fixed through the already-authorized 2B token probe. This is not treated as proof of unlimited gains from token scaling; the 2B point is the next controlled test for continued improvement versus diminishing returns or capacity saturation.

Canonical evidence: [`../evidence/20m/20m_500m_post_pretraining_full_suite_2026-08-10.md`](../evidence/20m/20m_500m_post_pretraining_full_suite_2026-08-10.md)

## Qualified GDN-2 production backend

The production CUDA backend is **mixed FLA on `fla-core==0.5.2`** under the trainer contract of FP32 master parameters plus CUDA FP16 autocast.

Qualified T4 stack:

```text
GPU: Tesla T4 / SM75
PyTorch: 2.10.0+cu128
CUDA runtime: 12.8
Triton: 3.6.0
fla-core: 0.5.2
saved/configured gdn_chunk_size: 32
FLA internal runtime chunk: 64
```

Corrected qualification established that all requested synthetic constant-decay rows pass against the finite FP32 adaptive oracle, the exact real step-4000 next-block forward/backward gate passes with finite parity-matching gradients, and warmed true-block throughput measured 22,765.80 target tok/s for mixed FLA versus 1,964.75 target tok/s for the adaptive FP32 recurrence. Full-FP32 FLA remains a diagnostic/fallback mode.

## Authorized next experiment — fresh 20M / 2B run

ADR 0023 supersedes the unrun 1B plan and authorizes a new independent approximately-2B-token data-scaling trajectory for the same 20M model. ADR 0027 records the 500M qualitative evidence as additional justification for keeping model size fixed through this 2B point.

Fixed identities:

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

The 2B trajectory does **not** continue the 500M checkpoint and has no dependency on an experimental 1B checkpoint. It starts from fresh seed-17 initialization and uses qualified mixed FLA on CUDA from optimizer update 1.

At 20,637,592 learned parameters, the nominal point is approximately 96.9 accepted source tokens per parameter. A full-block estimate is approximately 61.0k optimizer updates; the exact update count and WSD boundaries are derived from the completed verified manifest.

### Dataset transport

The selected operational path remains:

```text
pinned Nemotron-ClimbMix source
        ↓
VPS deterministic production build
        ↓
local immutable uint16 shards + verified Google Drive mirror
        ↓
private Kaggle publication + round-trip byte verification
        ↓
attached Kaggle dataset
        ↓
GPU training from Kaggle-local input
```

Do not stream the source corpus live during GPU training. The complete 2B prepared payload is still modest at roughly 4 GB of raw uint16 token IDs before validation/EOD/manifest overhead, while prebuilding preserves deterministic manifest/block identity and removes source-network variability from the T4 critical path.

Current operational state: **the 2B launch surface is prepared in the active change set; dataset production and optimizer update 1 have not yet been accepted as completed evidence. The 1B setup was superseded before any build or training result existed.**

## Frozen/accepted decisions still in force

- Keep the existing pinned source revision, GPT-2 token IDs, and programming-cluster-11 exclusion policy.
- Preserve context length 2,048 for this 20M scaling series.
- Preserve checkpoint/model config `gdn_chunk_size=32`; CUDA FLA executes its fixed internal chunk 64.
- Keep the adaptive PyTorch backend as the correctness/reference fallback.
- Do not clip/bound learned GDN-2 decay solely for backend runtime behavior.
- Use microbatch 4 on the qualified T4 training path.
- Let FP16 loss scaling calibrate down to scale 1.0 before failing an otherwise atomic block.
- Preserve `eval_core_v1` plus free-generation and teacher-forced confidence/rank diagnostics.
- New finite scaling trajectories start fresh rather than continuing through a previous run's terminal WSD decay unless a later ADR explicitly changes that rule.
- Keep the approximately-20M model size fixed through the planned 2B probe; revisit further fixed-size token scaling only after the frozen 2B comparison.

## Current source of truth

- 500M qualitative evidence: [`../evidence/20m/20m_500m_post_pretraining_full_suite_2026-08-10.md`](../evidence/20m/20m_500m_post_pretraining_full_suite_2026-08-10.md)
- 500M scaling interpretation: [`../decisions/0027-use-500m-schema-gains-to-justify-fixed-20m-token-scaling-through-2b.md`](../decisions/0027-use-500m-schema-gains-to-justify-fixed-20m-token-scaling-through-2b.md)
- 2B decision: [`../decisions/0023-run-2b-20m-probe-via-vps-kaggle-dataset.md`](../decisions/0023-run-2b-20m-probe-via-vps-kaggle-dataset.md)
- 2B runbook: [`../runbooks/20m_2b_runbook.md`](../runbooks/20m_2b_runbook.md)
- Dataset contract: [`../reference/dataset_and_tokenization.md`](../reference/dataset_and_tokenization.md)
- FLA consolidated handoff: [`gdn2_fla_investigation_handoff.md`](gdn2_fla_investigation_handoff.md)
- FLA qualification: [`gdn2_fla_qualification.md`](gdn2_fla_qualification.md)
- Durable decisions: [`../decisions/README.md`](../decisions/README.md)
