---
status: accepted
date: 2026-08-06
supersedes: immediate 20M all-attention baseline proposal
---

# 0003 — Defer architecture baselines until larger models

## Context and problem statement

A matched all-attention baseline is scientifically useful, but the current goal is to understand the main GDN-2 hybrid's data-scaling behavior while the approximately-20M model moves from 10M to 100M training tokens. Running another mixer now would split limited time and compute before the main learning curve and evaluation system are established.

## Considered options

- Train the matched all-attention baseline immediately at 20M parameters.
- Explore additional attention mechanisms during the current run.
- Keep the main architecture fixed and revisit controlled baselines at larger model versions.

## Decision outcome

Chosen option: **keep the main GDN-2 hybrid fixed during the current stage and revisit architecture baselines when larger model versions are reached**.

The evaluator remains architecture-agnostic so this timing decision does not block later matched comparisons.

## Consequences

### Positive

- The current experiment isolates data scaling instead of changing model and data simultaneously.
- Engineering effort goes into evaluation quality and the main training path.
- Later baselines can reuse the same frozen scorecard.

### Negative or limiting

- The project cannot yet claim that GDN-2 beats a matched Transformer.
- Architecture attribution is deferred; current results describe the chosen system only.

## Validation

Revisit this decision after the 100M-token result is evaluated and a larger model geometry is explicitly authorized.

## Links

- [`../current/roadmap.md`](../current/roadmap.md)
- [`../reference/model_architecture.md`](../reference/model_architecture.md)
