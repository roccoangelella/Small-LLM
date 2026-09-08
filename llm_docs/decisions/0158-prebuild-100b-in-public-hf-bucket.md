---
status: accepted
date: 2026-09-08
supersedes: null
---

# 0158 — Prebuild the 100B corpus in a public HF bucket

## Context and problem statement

ADR 0153 selects 100M/100B and names the existing 10B dataset machinery as the starting point. Building the corpus inside a paid provider session wastes the small provider budget and makes Modal's 24-hour CPU-function lifetime relevant even though dataset construction does not need a GPU.

## Considered options

- Produce shards concurrently inside each provider, as in the original 10B launch.
- Build provider-specific copies.
- Build one complete corpus outside the providers and publish it through the existing HF shard protocol.

## Decision outcome

Chosen option: **prebuild one complete public-HF 100B corpus using the existing 10B schema-v2 incremental producer**, because it preserves the qualified data semantics while removing dataset production from paid GPU-provider sessions.

The frozen dataset identity is `100b-b64-dataset-001`. It keeps context 2048, 64 sequences per optimizer block, approximately-1-GiB immutable shards, the existing deterministic split and mixture scheduler, 500M-source-token durability cadence, 16 frozen validation blocks, verified upload/read-back, bounded local eviction, and a terminal READY frontier. The nominal horizon is 762,940 blocks / 100,000,071,680 target tokens.

The bucket must be public and dedicated to this corpus. Modal and Beam do not launch concurrent producers for this profile. Both must verify public visibility, `ready.json`, the completed frontier, full train/validation coverage, and the checkpoint-aligned current-plus-successor shard window before GPU dispatch.

The production lanes remain those authorized by ADR 0153: Modal H100 and Beam RTX 4090. Every 100B provider invocation requires an explicit positive session-step budget.

## Consequences

### Positive

- Dataset construction consumes no Modal or Beam GPU allocation.
- Both providers consume exactly the same immutable corpus identity.
- Existing resume, SHA-256, READY-frontier, and rolling-cache code is reused.
- Provider-local dataset storage remains bounded around current plus successor.

### Negative or limiting

- Training cannot start until the complete public frontier is published.
- Deleting the public bucket before all training/checkpoint recovery is complete makes later shard resume impossible.
- Bucket visibility and identity must remain unchanged for the trajectory.

## Validation

Repository tests must prove the exact 100B horizon and profile arguments, public-bucket visibility checks, completed-frontier gates, common Modal/Beam profile resolution, provider restrictions, and explicit session budgets. A live launch still requires the completed remote bucket and provider credentials.

## Links

- `0153-select-100m-100b-as-next-pretraining-scale-target.md`
- `../reference/100m_10b_incremental_dataset.md`
- `../../dataset/README.md`
