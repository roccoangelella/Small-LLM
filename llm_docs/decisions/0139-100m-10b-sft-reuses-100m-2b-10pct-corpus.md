---
id: 0139
title: 100M/10B SFT reuses the 100M/2B 10% S0 corpus
status: accepted
date: 2026-09-03
---

# 0139. 100M/10B SFT reuses the 100M/2B 10% S0 corpus

## Context and problem statement

The completed 100M/10B parent consumes 10,000,007,168 target tokens. Applying the historical percentage-shaped SFT policy directly to that parent would require about 400,000,286 SFT train targets at 4% or about 1,000,000,716 SFT train targets at 10%.

The current S0 data path has only been proven at the already-published 100M/2B 10% bundle horizon: 200,100,044 requested train targets, with the accepted private Kaggle bundle identity recorded by its published split manifests. Rebuilding a larger 100M/10B percentage-shaped corpus would mix two variables at once: parent pretraining quality and available SFT data scale.

The immediate experiment should isolate the value of the stronger 100M/10B pretraining endpoint under the same post-training data exposure used by the 100M/2B 10% SFT run.

## Considered options

1. Apply the old 4% rule to the 10B parent, creating a roughly 400M-target SFT run.
2. Apply the old 10% rule to the 10B parent, creating a roughly 1B-target SFT run.
3. Reuse the exact already-published 100M/2B 10% S0 corpus and train the 100M/10B parent on the same SFT tokens.
4. Wait for a larger instruction-data audit before doing any 100M/10B SFT.

## Decision outcome

Accept option 3.

The canonical 100M/10B SFT profile is `100m-10b-sft-s0-2b10pct-data-001`. It uses the completed parent run `100m-10b-deep-decay-from-step15500` resolved from the verified Hugging Face Storage Bucket `latest` pointer, not the validation-loss `best` pointer.

The SFT data is the same published S0 10% corpus used for the 100M/2B run: `small-llm-100m-2b-sft-s0-10pct-001`. The train target budget remains the absolute 100M/2B 10% horizon of 200,100,044 requested loss-bearing targets, not 4% or 10% of the 10B parent. In the 10B profile this is represented as the exact fraction `200100044 / 10000007168` only to satisfy existing trainer budget plumbing.

The training schedule remains the accepted 100M/2B 10% peak-through-3000 schedule with LR 3e-5, microbatch 2, cadence 250, and 2xT4 DDP execution. The profile disables `--sft-fraction` overrides so the experiment cannot silently drift into a percentage-shaped 10B SFT run.

## Consequences

This creates an apples-to-apples comparison: same learned-parameter count, same architecture, same SFT dataset, same SFT target horizon, same SFT schedule, but different pretraining horizon and parent checkpoint quality.

The experiment is not a new general 2% SFT policy. The apparent percentage is only the absolute 200,100,044-token corpus expressed against the exact 10B parent counter.

A later ADR is still required before launching a true percentage-shaped 100M/10B SFT run, adding controlled repetition, or expanding the instruction corpus beyond the accepted S0 bundle.
