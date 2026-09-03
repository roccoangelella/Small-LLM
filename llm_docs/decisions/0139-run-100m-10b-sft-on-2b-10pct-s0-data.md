---
status: accepted
date: 2026-09-03
supersedes: 0138
---

# 0139 — Run 100M/10B SFT on the 100M/2B 10% S0 data

## Context and problem statement

ADR 0138 registered the completed 100M/10B pretraining endpoint as a first-class SFT parent but left the scientific SFT recipe fail-closed. The immediate concern is that scaling the historical 4% or 10% SFT policy directly from the 10,000,007,168-token parent would require roughly 400M or 1B SFT train targets, which likely exceeds the finite no-silent-repeat S0 instruction data budget.

The project already has an accepted and privately published 100M/2B 10% S0 corpus. Its train horizon is the accepted equal-SFT-token budget for this comparison: 200,100,044 requested train targets, with the published train split identity bound by the existing manifest and publication checks.

The desired experiment is not to maximize 100M/10B SFT performance with a newly scaled data recipe. It is to isolate how much the stronger 100M/10B pretraining parent improves downstream SFT behavior when the architecture, SFT data, SFT token count, and SFT schedule are kept as close as possible to the 100M/2B 10% S0 run.

## Considered options

- Keep 100M/10B SFT blocked until a full data-capacity audit chooses a new 4%/10% replacement recipe.
- Scale the old 4% or 10% policy against the 10B parent and risk exhausting or repeating finite instruction data.
- Reuse the exact 100M/2B 10% S0 dataset and absolute token budget for the 100M/10B parent, producing an apples-to-apples same-SFT-data comparison.

## Decision outcome

Wire the Kaggle SFT path so `--model 100M --tokens 10B` uses the completed 100M/10B final parent with the exact 100M/2B 10% S0 training dataset and absolute SFT train-token budget.

The experiment identity is:

```text
parent run:       100m-10b-deep-decay-from-step15500
parent pointer:   latest
parent transport: hf_storage_bucket
parent targets:   10,000,007,168
SFT dataset:      small-llm-100m-2b-sft-s0-10pct-001
SFT targets:      200,100,044 requested train targets
SFT run id:       100m-10b-sft-s0-2b10pct-data-001
LR schedule:      same peak-through-3000 policy as the accepted 100M/2B 10% run
peak LR:          3e-5
```

This replaces the pending recipe state from ADR 0138 for the narrow same-data experiment only. It does not approve a scaled 4% or 10% 100M/10B SFT recipe.

## Consequences

The 100M/10B SFT profile must reject `--sft-fraction` overrides, because this experiment is defined by an absolute corpus/token budget, not by a percentage of the 10B parent.

The Kaggle launcher must bind to the already-published 100M/2B 10% S0 dataset identity and fail closed if that dataset is missing or fails manifest/publication checks. It must not rebuild a larger 10B-derived SFT bundle for this experiment.

The parent checkpoint must resolve through the verified Hugging Face Storage Bucket `latest` path for the final 100M/10B endpoint, not through the 2B model-repository stable artifact path or any validation-loss `best` pointer.

Evaluation of this run should be interpreted as a controlled comparison of pretraining quality under fixed SFT data, not as evidence that 200M SFT targets is the optimal post-training budget for 100M/10B.
