---
status: accepted
date: 2026-09-06
supersedes: null
---

# ADR 0156 — Use small random bias-free router initialization without jitter

Date: 2026-09-06
Branch scope: `moe-8e-top1`

## Context and problem statement

No separate context was recorded at the time; section added for the template.

## Considered options

No alternative was recorded at the time; section added for the template.

## Decision outcome

For the first 8-expert Top-1 MoE experiment:

- initialize each per-layer router projection with small random weights;
- do not use a router bias term;
- do not add router jitter/noise during training;
- keep routing deterministic given model state, RNG-initialized weights, and input tokens;
- retain FP32 router logits/softmax, Switch-style Top-1 selected-probability output scaling, dropless dispatch, no load-balancing objective/bias, and router z-loss coefficient `1e-4` from the previously accepted MoE experiment contract.

The small random router initialization breaks exact symmetry between experts without imposing an explicit expert-utilization target. Jitter/noise remains a future ablation if the first run exhibits pathological routing collapse or poor specialization.

## Consequences

This decision is intentionally recorded on the experimental `moe-8e-top1` branch only and does not modify `main`.
