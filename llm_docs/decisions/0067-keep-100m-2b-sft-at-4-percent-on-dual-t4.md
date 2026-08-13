---
status: accepted
date: 2026-08-13
supersedes: 0066
---

# ADR 0067: Keep 100M/2B SFT at 4% on Kaggle dual T4

The 100M/2B SFT run keeps the established 4%-of-parent budget. With the verified parent count of 2,001,000,448 training targets, the requested SFT horizon is 80,040,017 loss-bearing targets.

The SFT mixture and stratification stay unchanged: 85% filtered instruction targets, 15% frozen ClimbMix replay, with instruction targets split 75% `smol-magpie-ultra-short`, 10% `smol-contraints`, 7.5% `smollm-rewrite-30k`, and 7.5% `smol-summarize-20k`.

The two-Tesla-T4 Kaggle DDP implementation remains selected, including exact global-token loss scaling, synchronized exact resume, and rank-zero-only W&B, checkpoint, evaluation, and publication side effects.

The canonical run identity is `100m-2b-sft-s0-001`. This decision supersedes ADR 0066's temporary 10% budget choice while retaining its dual-T4 execution work.
