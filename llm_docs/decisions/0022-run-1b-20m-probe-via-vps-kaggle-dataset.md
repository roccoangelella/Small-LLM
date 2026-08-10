---
status: accepted
date: 2026-08-10
supersedes: 0008
---

# 0022 — Run a fresh 20M / 1B probe from a VPS-built Kaggle dataset

## Context and problem statement

The approximately-20M-parameter GDN-2 hybrid has now completed the 500M-token pretraining trajectory, and the qualified mixed FLA GDN-2 backend has made the same-model training path substantially faster on Tesla T4. The user authorized another same-model data-scaling point at approximately 1B accepted source tokens.

The immediate operational choice is whether the Kaggle GPU job should stream source data while training or consume a fully prepared finite dataset built outside Kaggle.

The existing production dataset system already provides a deterministic pinned-source reader, exact token-mixture scheduler, context+1 packing, immutable uint16 shards, Google Drive durability, manifest hashing, resumable production state, private Kaggle publication, and round-trip verification. At one billion stored token IDs, the raw uint16 payload is only on the order of two gigabytes before small manifest/validation overhead, so a complete attached Kaggle dataset remains operationally modest.

The 20M model has 20,637,592 learned parameters. A nominal 1B source-token target is approximately 48.5 source tokens per parameter. Recent small-model scaling work commonly studies substantially larger token/parameter exposure; for example, DataDecide explicitly trains its model ladder, including a 20M model, to a token/parameter ratio of 100. The 1B point should therefore be treated as an additional scaling probe rather than an assumption that the 20M model is data-saturated.

## Considered options

- Stream the pinned Nemotron-ClimbMix source directly during the Kaggle GPU training job.
- Build the deterministic 1B finite dataset on the VPS, mirror it durably to Google Drive, privately publish it to Kaggle, round-trip verify it, and train from the attached Kaggle input.
- Build the 1B finite dataset inside the Kaggle training session before starting optimizer updates.
- Continue the completed 500M checkpoint to 1B total exposure instead of starting a fresh trajectory.

## Decision outcome

Chosen option: **build and verify the complete 1B finite dataset on the VPS, privately publish it to Kaggle, and train a fresh seed-17 20M model from that attached dataset**.

The fixed dataset identity is:

```text
profile: 20m-1b-data-scaling-v1
run ID: 20m-1b-dataset-001
target accepted source tokens: 1,000,000,000
minimum accepted source tokens: 900,000,000
hard maximum accepted source tokens: 1,100,000,000
producer durable checkpoint cadence: 40,000,000 source tokens
context length: 2,048
sequences per optimizer block: 16
target shard size: 8 MiB
remote durability: required
```

The training trajectory is independent of the completed 500M checkpoint:

```text
fresh initialization seed: 17
architecture: gdn2_hybrid
precision: FP16 autocast with FP32 master parameters
training microbatch: 4
saved/configured GDN chunk: 32
CUDA execution: qualified mixed FLA, internal chunk 64
optimizer: hybrid Muon + AdamW
schedule: exact one-pass WSD derived from the completed 1B manifest
training durability / validation / remote publication cadence: 250 updates
W&B run ID: 20m-1b-data-001
```

The fresh run uses FLA from optimizer update 1. It does not inherit the 500M model weights, optimizer state, scheduler state, loss scaler, RNG state, or data cursor.

## Consequences

### Positive

- No Hugging Face/source-network dependency sits on the GPU training critical path.
- Kaggle training sees one immutable, hashed, finite dataset identity and can fail closed if the wrong dataset is attached.
- Exact checkpoint/resume semantics remain tied to a stable block schedule rather than to live source-stream state.
- The already-qualified VPS producer, Drive mirror, Kaggle publication, and round-trip verification machinery are reused instead of introducing a second ingestion path.
- The 1B point extends the same-model data-scaling curve to approximately 48.5 source tokens per parameter while keeping architecture, tokenizer, context, optimizer geometry, and seed policy comparable.
- Starting mixed FLA at update 1 removes the backend-migration discontinuity present in the historical 500M trajectory.

### Negative or limiting

- Training waits for a one-time VPS build and private Kaggle upload before GPU optimization begins.
- The prepared dataset is duplicated across VPS/Drive/Kaggle storage during publication and verification.
- A fresh 1B trajectory is scientifically cleaner for scaling comparison but costs more compute than continuing the existing 500M checkpoint.
- The 1B point is still only an intermediate exposure relative to modern small-model scaling studies and should not be treated as a final token-budget optimum.

## Validation

This decision is operationally satisfied only when:

1. the 1B producer completes under the fixed identity and full local verification passes;
2. every immutable shard referenced by the production cursor is verified in the Google Drive mirror;
3. private Kaggle publication completes and a fresh Kaggle download is byte-identical to the VPS tree after excluding Kaggle transport artifacts;
4. anonymous access to the private dataset is denied;
5. the Kaggle launcher finds exactly one attached `20m-1b-dataset-001` dataset and rejects 100M/500M identities;
6. the exact one-pass WSD trainer plan is derived from the completed manifest rather than guessed from the nominal source-token target;
7. the fresh seed-17 run starts at microbatch 4 with mixed FLA on CUDA and completes the normal 250-update validation/checkpoint/verified-publication gates;
8. the frozen post-pretraining evaluation bundle is run on the final checkpoint for direct comparison with earlier 20M scaling points.

## Links

- [`../runbooks/20m_1b_runbook.md`](../runbooks/20m_1b_runbook.md)
- [`../reference/dataset_and_tokenization.md`](../reference/dataset_and_tokenization.md)
- [`../current/status.md`](../current/status.md)
- [`0008-run-500m-final-20m-data-scaling-probe.md`](0008-run-500m-final-20m-data-scaling-probe.md)
- DataDecide: https://github.com/allenai/DataDecide
