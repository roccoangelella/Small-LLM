---
status: accepted
date: 2026-09-10
supersedes: null
owners: [Small-LLM]
---

# ADR 0170: use the completed 20M dense backbone as the initial ultra-small MoE base

## Context and problem statement

ADR 0168 opened an investigation into an ultra-small many-expert MoE intended for a 100B-token pretraining trajectory. The remaining architecture should be scientifically interpretable against an already trained dense Small-LLM geometry rather than introducing unnecessary changes to depth, residual width, mixer schedule, attention geometry, and GDN-2 geometry at the same time as sparsity.

The completed approximately-20M dense model is an unusually clean base for this experiment: it already uses `d_model=256`, 8 decoder layers, 6 GDN-2 layers plus 2 gated full-MHA layers, `d_ff=704`, and 4 attention/GDN heads of dimension 64. This matches the scale under discussion for the ultra-small MoE while keeping the shared backbone substantially smaller than the 100M geometry.

## Considered options

- Base the first ultra-small MoE on the completed 20M dense geometry.
- Base it on the completed 100M dense geometry (`d_model=512`, 20 layers, `d_ff=1408`), which would make a many-expert model much larger in both active and total parameters.
- Invent a new intermediate or narrower backbone specifically for the MoE, sacrificing direct comparability with an already completed dense trajectory.

## Decision outcome

Chosen option: **use the completed 20M dense geometry as the architectural base for the first ultra-small MoE experiment.**

Unless later architecture decisions explicitly override a component, the comparison baseline is therefore:

- `d_model=256`;
- 8 decoder layers;
- `[GDN-2, GDN-2, GDN-2, gated full MHA] x 2`;
- 6 GDN-2 layers and 2 gated full-MHA layers;
- 4 heads x 64 dimensions for attention and GDN key/value geometry;
- dense reference `d_ff=704`;
- context length 2048;
- the same pre-norm residual layout, RMSNorm, RoPE/MHA semantics, GDN-2 recurrence semantics, dropout=0, and tied embedding/output design as the completed 20M model.

The MoE-specific design is not frozen by this ADR. In particular, expert count, Top-K, per-expert hidden width, shared experts, router score function, gate normalization, balancing method (including Quantile Balancing), capacity/drop policy, stabilization terms, and execution kernel remain open.

The custom reduced vocabulary remains part of the ADR-0168 investigation and is a deliberate non-MoE architectural change; its exact tokenizer contract is not frozen here.

## Consequences

### Positive

- The experiment can isolate sparse-FFN effects while reusing a completed dense backbone with measured behavior.
- `d_model=256` and 8 layers keep the shared backbone and many-expert parameter bank small enough for constrained GPUs.
- The dense `d_ff=704` provides a natural active-compute reference when choosing Top-K and expert width.

### Negative or limiting

- This is not a parameter-for-parameter comparison against the 100M dense trajectory.
- A custom tokenizer will still prevent perfect one-variable comparability with the historical 20M model unless a same-tokenizer dense control is also trained or evaluated.
- The architecture remains incomplete until the MoE subsystem and tokenizer are separately frozen.

## Validation

Before production wiring, compute exact active/total parameter counts for the final MoE geometry and verify that all non-MoE backbone dimensions and layer semantics match the completed 20M reference except for explicitly accepted deviations.

## Links

- [`0168-investigate-ultra-small-sparse-moe-for-100b-tokens.md`](0168-investigate-ultra-small-sparse-moe-for-100b-tokens.md)
- [`../reference/model_architecture.md`](../reference/model_architecture.md)
- [`../reference/model_geometry.md`](../reference/model_geometry.md)
