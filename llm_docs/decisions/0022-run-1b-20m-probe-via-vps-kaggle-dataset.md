---
status: superseded
date: 2026-08-10
supersedes: 0008
superseded_by: 0023
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

This decision was superseded before any 1B dataset build or training run began. ADR 0023 replaces the target with 2B accepted source tokens while preserving the VPS-build/private-Kaggle transport choice.

The fixed dataset identity was:

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

The intended training trajectory was independent of the completed 500M checkpoint and would have used seed 17, microbatch 4, mixed FLA from update 1, hybrid Muon + AdamW, an exact one-pass WSD schedule, and 250-update durability.

## Consequences

### Positive

- The data-transport rationale remains useful and is retained by ADR 0023.
- The superseded record makes clear that no 1B experimental result exists.

### Negative or limiting

- The exact 1B dataset/training identities in this ADR are historical only and must not be launched as the current experiment.

## Validation

Supersession is complete when active status, roadmap, runbook indexes, and launch surfaces point only to the 2B experiment and no 1B result is treated as an experimental datapoint.

## Links

- [`0023-run-2b-20m-probe-via-vps-kaggle-dataset.md`](0023-run-2b-20m-probe-via-vps-kaggle-dataset.md)
- [`../reference/dataset_and_tokenization.md`](../reference/dataset_and_tokenization.md)
- [`../current/status.md`](../current/status.md)
