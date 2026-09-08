# ADR 0153 — Add a controlled top-k sampling benchmark

Date: 2026-09-08
Status: accepted

## Decision

Add an additional sampled Base Prompt benchmark for both pretrained and SFT evaluation paths, keeping the current sampled temperature and nucleus settings while applying a tighter positive top-k cutoff:

- `temperature=1.0`
- `top_p=1.0`
- `top_k=15`
- `seed=17`
- output view name: `sampled_topk15`

The existing canonical sampled benchmark remains unchanged at `temperature=1.0`, `top_p=1.0`, `top_k=0`, `seed=17` for comparability with prior runs.

Important semantic clarification: in the current evaluator, `top_k=0` means top-k filtering is disabled and sampling can draw from the full vocabulary distribution. `top_k=15` instead restricts each decoding step to the 15 highest-logit candidates before sampling, preserving stochastic decoding while aggressively cutting the low-probability tail.

The same `sampled_topk15` Base Prompt view must be produced for pretrained checkpoints and for both parent and SFT sides of post-SFT qualification. The GemRouter Base Prompt semantic judge must score greedy, canonical sampled, and `sampled_topk15` views.

## Rationale

The current full-distribution sampled results are substantially below greedy results. For a 100M-parameter model, the hypothesis is that the long probability tail contains too many weak candidates, so unconstrained `top_k=0` sampling destroys otherwise usable local knowledge. A top-15 view is deliberately more restrictive than top-50 and gives a stronger diagnostic of whether sampled degradation is caused by tail noise rather than missing knowledge.

This benchmark is additive. It does not redefine the canonical sampled contract.
