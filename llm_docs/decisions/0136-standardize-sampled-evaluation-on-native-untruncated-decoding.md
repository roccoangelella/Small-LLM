---
status: accepted
date: 2026-09-03
supersedes: 0059
---

# 0136 — Standardize sampled evaluation on native untruncated decoding

## Context and problem statement

The project has accumulated two sampled decoding configurations in evaluation artifacts:

- legacy generic sampler defaults: `temperature=0.8`, `top_p=0.95`, `top_k=50`;
- ADR 0059 cross-checkpoint sampled comparison: `temperature=1.0`, `top_p=0.9`, `top_k=20`.

This created avoidable protocol drift, including the final 100M/10B sampled qualification being generated under legacy defaults rather than the ADR 0059 comparison settings.

The project now wants one simple, neutral sampled protocol that interferes as little as possible with the model's learned next-token distribution and is easy to remember and verify.

## Decision outcome

Effective immediately, the **standard sampled qualitative/model-comparison evaluation protocol** is:

```text
temperature: 1.0
top_p: 1.0
top_k: 0
seed: 17
samples_per_prompt: 1
questions_only: false
```

Interpretation:

- `temperature=1.0` preserves the learned logit scale;
- `top_p=1.0` disables nucleus truncation;
- `top_k=0` means top-k truncation is disabled;
- sampling therefore draws from the model distribution without temperature sharpening or top-p/top-k filtering.

Use each prompt's native sampled-test generation budget unless a separately frozen suite explicitly defines another budget.

This protocol supersedes ADR 0059's `temperature=1.0`, `top_p=0.9`, `top_k=20` sampled configuration for all **future standard cross-checkpoint sampled evaluations**, including pretrained and ordinary SFT model comparisons.

It does **not** alter independently defined task-specific stochastic protocols such as R-SFT reasoning pass@1/majority-style evaluation, which must retain their own explicitly frozen settings unless separately changed.

## Historical-result handling

Do not rewrite old sampled results as if they used this protocol.

Historical artifacts remain valid evidence under the decoding settings they actually recorded:

- `0.8 / 0.95 / 50` results remain legacy sampled diagnostics;
- `1.0 / 0.9 / 20` results remain ADR 0059-era sampled comparisons;
- only results generated after this decision with `1.0 / 1.0 / 0` are canonical under the new sampled protocol.

Cross-checkpoint sampled claims must compare matching decoding configurations. If a checkpoint lacks a `1.0 / 1.0 / 0` result, rerun the sampled qualitative test rather than numerically comparing incompatible historical runs.

## Operational consequence

Future evaluation commands, wrappers, runbooks, and default presets intended to produce the project's standard sampled comparison should explicitly resolve to:

```text
--temperature 1.0 --top-p 1.0 --top-k 0 --seed 17 --samples-per-prompt 1
```

The JSON output must record these decoding fields so protocol identity can be verified from the artifact itself.

## Superseded decision

- ADR 0059 — `temperature=1.0`, `top_p=0.9`, `top_k=20` for supplementary sampled cross-checkpoint evaluation.
