---
status: accepted
date: 2026-09-06
owners: [Small-LLM]
---

# ADR 0153: select 100M/100B as the next pretraining scale target

## Decision

The next major pretraining scaling experiment will keep the model at approximately 100M parameters and increase the nominal pretraining horizon to 100B target tokens.

For this stage, do not increase the model parameter count. The purpose is to test how far the current approximately-100M architecture can be pushed with substantially heavier overtraining before moving to a larger model.

The production GPU execution lanes for this 100M/100B trajectory are restricted to:

- Beam RTX 4090;
- Modal H100.

Kaggle dual-T4 is explicitly excluded for the 100M/100B trajectory because its expected wall-clock time is not operationally acceptable at this token horizon.

## Context

The completed 100M/10B trajectory reached 10,000,007,168 consumed target tokens. It improved materially over 100M/2B, but its late marginal improvement was much smaller and is confounded by an aggressively reduced terminal learning rate. The project therefore does not treat the 10B endpoint as clean evidence that the 100M model has exhausted useful data scaling.

Contemporary small-model practice also includes very large token-to-parameter ratios, so a 100B-token experiment at approximately 100M parameters is scientifically defensible as an overtraining study rather than a compute-optimal Chinchilla allocation.

## Implementation state

This ADR authorizes the scale target and provider boundary only. It does not yet freeze the 100B dataset profile, learning-rate schedule, checkpoint cadence, or exact launch command.

The existing 10B infrastructure is the starting point for design:

- schema-v2 little-endian uint16 packed tokens;
- block64/context-2048 optimizer geometry;
- approximately-1-GiB immutable Hugging Face dataset shards;
- incremental producer with a monotonic READY frontier;
- CPU-before-GPU current-plus-successor staging;
- one-shard-ahead online prefetch;
- exact checkpoint-aligned resume;
- Hugging Face Storage Bucket durability.

A dedicated 100B profile and schedule must be reviewed before any production launcher is wired.

## Open design questions

1. Freeze the 100B dataset/run contract and confirm whether the existing incremental frontier can be extended directly or needs retention/frontier changes for roughly 200 GB of packed token bytes.
2. Select an LR policy suited to roughly 762,940 optimizer updates without reproducing the 10B trajectory's very-low-LR tail too early.
3. Qualify the provider-specific 100B launch surfaces for Beam RTX 4090 and Modal H100 while keeping Kaggle unavailable for this profile.
