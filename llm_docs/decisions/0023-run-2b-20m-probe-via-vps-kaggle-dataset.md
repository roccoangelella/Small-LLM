---
status: accepted
date: 2026-08-10
supersedes: 0022
---

# 0023 — Run the next 20M data-scaling probe at 2B tokens via a VPS-built Kaggle dataset

## Context and problem statement

ADR 0022 authorized a fresh approximately-1B-token probe for the approximately-20M-parameter GDN-2 hybrid after the qualified FLA backend made long runs substantially faster. Before that 1B experiment was built or trained, the user changed the desired data budget to approximately 2B accepted source tokens.

The model has 20,637,592 learned parameters. A 2B accepted-source-token point is approximately 96.9 source tokens per parameter, making it a deliberately strong overtraining/data-scaling probe of this fixed small model before parameter scaling.

The data-transport question remains whether Kaggle should stream the source corpus live or consume a finite dataset built elsewhere. The existing producer already provides deterministic source-range ingestion, cluster filtering, exact mixture scheduling, context+1 packing, immutable shards, Google Drive durability, and private Kaggle round-trip verification. At 2B tokens the uint16 payload remains only roughly 4 GB before validation/EOD/manifest overhead, so prebuilding does not create a material storage obstacle.

## Considered options

- Keep the unrun 1B experiment authorized by ADR 0022.
- Run a fresh approximately-2B-token experiment and stream source data live during Kaggle training.
- Run a fresh approximately-2B-token experiment using a finite dataset built and verified on the VPS, then attached to Kaggle.
- Continue the completed 500M checkpoint instead of starting a fresh trajectory.

## Decision outcome

Chosen option: **replace the unrun 1B plan with a fresh approximately-2B-token probe, using a VPS-built, verified, privately published Kaggle dataset**.

The fixed dataset envelope is:

```text
profile: 20m-2b-data-scaling-v1
run ID: 20m-2b-dataset-001
target accepted source tokens: 2,000,000,000
minimum: 1,800,000,000
maximum: 2,200,000,000
producer checkpoint cadence: 80,000,000 source tokens
context: 2,048
sequences per block: 16
target shard size: 8 MiB
remote durability: required
```

Training starts from the existing seed-17 initialization policy, not from the 500M checkpoint. It uses microbatch 4, FP16 autocast with FP32 master parameters, hybrid Muon + AdamW, a one-pass WSD schedule derived from the completed 2B manifest, and the qualified mixed `fla-core==0.5.2` GDN-2 backend from update 1. Validation, local checkpointing, and verified remote checkpoint publication remain every 250 successful optimizer updates.

The abandoned 1B setup is not an experimental datapoint and should not remain an active launch target.

## Consequences

### Positive

- The fixed 20M model is characterized at approximately 96.9 source tokens per parameter, giving a much stronger upper data-scaling point than 500M.
- The experiment remains directly comparable to earlier fresh seed-17 finite-data runs rather than mixing in continuation after a terminal WSD decay.
- Source/network ingestion remains off the GPU critical path.
- Dataset identity, exact schedule, checkpoint cursor, and resume behavior remain deterministic and fail closed.
- The approximately-4-GB prepared token payload remains practical for private Kaggle attachment.

### Negative or limiting

- The run roughly doubles the training work and prepared data of the abandoned 1B plan.
- It spends additional compute on a deliberately overtrained small model rather than moving immediately to a larger parameter count.
- The 80M producer checkpoint cadence increases replay between producer durability points relative to smaller profiles, although it keeps roughly the same number of production checkpoints across the build.
- The 2B result characterizes this model/recipe/data mixture; it does not isolate architecture quality from parameter capacity.

## Validation

The decision is fulfilled when:

1. the 2B dataset reaches the fixed envelope and passes local plus Drive verification;
2. the private Kaggle publication passes fresh round-trip byte-identity verification and anonymous-access denial;
3. the fresh seed-17 20M model completes the exact finite 2B WSD plan with the qualified mixed FLA backend and 250-update durability boundaries;
4. the frozen post-pretraining evaluation is run so 100M, 500M, and 2B checkpoints can be compared directly.

## Links

- [`../runbooks/20m_2b_runbook.md`](../runbooks/20m_2b_runbook.md)
- [`../current/status.md`](../current/status.md)
- [`../current/roadmap.md`](../current/roadmap.md)
- [`../reference/dataset_and_tokenization.md`](../reference/dataset_and_tokenization.md)
- [`0022-run-1b-20m-probe-via-vps-kaggle-dataset.md`](0022-run-1b-20m-probe-via-vps-kaggle-dataset.md)
