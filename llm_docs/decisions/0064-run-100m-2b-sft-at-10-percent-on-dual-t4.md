---
status: accepted
date: 2026-08-13
---

# ADR 0064: Run 100M/2B SFT at 10% on Kaggle dual T4

## Context

The completed 100M/2B pretraining endpoint has 2,001,000,448 consumed training targets. The first 20M/500M S0 SFT experiment used the earlier 4%-of-parent budget and failed behavioral qualification despite improving masked SFT likelihood. The next experiment should test whether substantially more supervised signal helps the larger 100M parent without changing the source stratification, so budget and model capacity are the intended experimental changes rather than a simultaneous mixture rewrite.

Kaggle exposes two Tesla T4 GPUs, and ADR 0056 already establishes exact-batch two-T4 DDP as the Kaggle production topology. SFT should use both GPUs rather than leaving one idle, while preserving one global optimizer update per immutable SFT block and rank-zero-only external side effects.

## Decision

For the 100M/2B SFT profile only:

- request SFT train loss-bearing target tokens equal to 10% of the verified parent consumed-token counter;
- with the completed parent count 2,001,000,448, request exactly 200,100,044 SFT target tokens (integer floor);
- keep the overall target mixture unchanged at 85% filtered instruction / 15% frozen ClimbMix replay;
- keep the instruction-source stratification unchanged at 75% `smol-magpie-ultra-short`, 10% `smol-contraints`, 7.5% `smollm-rewrite-30k`, and 7.5% `smol-summarize-20k`;
- keep the existing identity-safe 95/2.5/2.5 train/validation/test split, decontamination, template, assistant-only instruction loss, optimizer target-block size, seed, and qualification suite;
- launch Kaggle training as two-process NCCL DDP across exactly two Tesla T4 GPUs, with the global SFT block split across ranks and DDP loss scaling preserving the serial global-token objective;
- keep W&B, checkpoint publication, evaluation side effects, and final user-facing summary rank-zero-only; checkpoints remain topology-neutral raw-model snapshots and automatic verified resume remains required.

The earlier 4% rule remains historical/default behavior for existing 20M SFT profiles. This ADR is a profile-specific override for 100M/2B rather than a retroactive rewrite of completed experiments.

## Consequences

The new 100M/2B bundle must have a distinct identity from any 4% bundle and must be rejected if its requested target count does not equal 200,100,044 for the verified completed parent. The canonical launcher must expose the 100M/2B profile and report the 10% budget in dry-run output.

The Kaggle SFT execution shim must support variable SFT block sequence counts rather than assuming the fixed 16-sequence pretraining block. Both ranks must execute the same number of DDP synchronization points, including on a short final block, without adding loss-bearing padding targets.

Behavioral qualification remains mandatory; increasing SFT volume is an experiment, not evidence in advance that the SFT recipe is promoted.
