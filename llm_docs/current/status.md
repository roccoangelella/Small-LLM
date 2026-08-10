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

The 500M trajectory is **not** a clean single-backend curve: the accepted checkpoint chain reached step 4000 under the adaptive backend, then production continuation was authorized with the mathematically compatible mixed FLA GDN-2 CUDA backend. Interpret throughput and any very fine-grained loss-curve discontinuity with that migration boundary in mind.

The final frozen post-pretraining evaluation/scorecard should remain the source of truth for model-quality comparison; do not infer a final validation metric from prompt-suite metadata that was not explicitly labeled as held-out validation loss.

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

Corrected qualification established:

- all requested synthetic constant-decay rows pass against the finite FP32 adaptive oracle;
- the exact real step-4000 next-block forward/backward gate passes with finite, parity-matching gradients;
- warmed true-block throughput measured 22,765.80 target tok/s for mixed FLA versus 1,964.75 target tok/s for the adaptive FP32 recurrence;
- full-FP32 FLA is retained as a diagnostic/fallback mode, not the selected production path.

Detailed evidence remains under the August 8 GDN-2/FLA qualification documents and ADR 0021.

## Authorized next experiment — fresh 20M / 1B run

ADR 0022 authorizes a new independent approximately-1B-token data-scaling trajectory for the same 20M model.

Fixed identities:

```text
profile: 20m-1b-data-scaling-v1
dataset run ID: 20m-1b-dataset-001
W&B run ID: 20m-1b-data-001
target accepted source tokens: 1,000,000,000
minimum: 900,000,000
maximum: 1,100,000,000
producer durable checkpoint cadence: 40,000,000 source tokens
fresh initialization seed: 17
training microbatch: 4
training durability / validation / remote publication cadence: 250 updates
```

The 1B trajectory does **not** continue the 500M checkpoint. It starts from fresh seed-17 initialization and uses qualified mixed FLA on CUDA from optimizer update 1.

### Dataset transport

The selected operational path is:

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

Do not stream the source corpus live during GPU training. The complete 1B prepared payload is small enough for this path, while prebuilding preserves deterministic manifest/block identity and removes source-network variability from the T4 critical path.

Current operational state: **the 1B launch surface is being prepared; dataset production and optimizer update 1 have not yet been accepted as completed evidence.**

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

## Current source of truth

- 1B decision: [`../decisions/0022-run-1b-20m-probe-via-vps-kaggle-dataset.md`](../decisions/0022-run-1b-20m-probe-via-vps-kaggle-dataset.md)
- 1B runbook: [`../runbooks/20m_1b_runbook.md`](../runbooks/20m_1b_runbook.md)
- Dataset contract: [`../reference/dataset_and_tokenization.md`](../reference/dataset_and_tokenization.md)
- FLA consolidated handoff: [`gdn2_fla_investigation_handoff.md`](gdn2_fla_investigation_handoff.md)
- FLA qualification: [`gdn2_fla_qualification.md`](gdn2_fla_qualification.md)
- Durable decisions: [`../decisions/README.md`](../decisions/README.md)
