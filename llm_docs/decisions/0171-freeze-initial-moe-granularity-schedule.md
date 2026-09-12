---
status: accepted
date: 2026-09-10
supersedes: null
owners: [Small-LLM]
---

# ADR 0171: freeze the initial MoE granularity experiment and matched follow-ups

## Context and problem statement

The first ultra-small MoE experiment is based on the completed 20M dense Small-LLM backbone. The routed FFN keeps the dense SwiGLU structure, uses MoE in all eight decoder blocks, and the router contract already selected sigmoid scoring, biased Top-K selection, post-selection normalization using the unbiased scores, and Quantile Balancing.

The remaining first-order architecture choice is the routed-expert granularity. To keep the initial experiment scientifically interpretable against the dense reference, the active FFN width should initially match the dense `d_ff=704` budget while using many experts.

## Considered options

No alternative was recorded at the time; section added for the template.

## Decision outcome

The initial MoE experiment is frozen as:

- `E = 64` routed experts per MoE layer;
- `Top-K = 2`;
- expert SwiGLU hidden width `h = 352`;
- therefore `K * h = 2 * 352 = 704`, matching the dense reference FFN active hidden width;
- no shared experts unless a later ADR explicitly changes that choice.

Two matched follow-up experiments are also recorded now so they are not lost:

1. `E = 128`, `Top-K = 4`, `h = 176`;
2. `E = 256`, `Top-K = 8`, `h = 88`.

These three configurations preserve both:

- active FFN hidden-width budget: `K * h = 704`;
- total routed hidden-width capacity per layer: `E * h = 22,528`.

Thus the intended comparison primarily varies routing granularity / number of simultaneously active experts while holding the active expert budget and total expert-bank capacity constant, apart from the small increase in router parameters as `E` grows.

## Consequences

### Positive

- The first run starts from the least fragmented of the serious sparse candidates, avoiding extremely small expert GEMMs initially.
- `64E / Top-2 / h352` is directly matched to the dense 20M FFN active width.
- The planned `128E / Top-4 / h176` and `256E / Top-8 / h88` experiments form a controlled granularity series rather than ad-hoc Top-K changes.
- The experiment schedule explicitly preserves `K*h` and `E*h`, making quality and systems comparisons much easier to interpret.

### Negative or limiting

- Matching hidden-width and expert parameter budgets does not make the functions mathematically equivalent: routing and weighted expert mixtures change optimization and representation dynamics.
- Increasing `E` and `K` creates more router work, dispatch overhead, and smaller expert GEMMs even when theoretical expert FLOPs stay approximately matched.
- The later Top-4 and Top-8 configurations are planned experiments, not yet authorized implementations or training runs.

## Validation

Before wiring the initial 64E/Top-2/h352 implementation, the project protocol still requires an open-ended understanding check covering the routing path, the meaning of the matched `K*h=704` budget, and the role of Quantile Balancing. The grouped/fused execution path must also be qualified on target hardware before a long production run.

## Links

- `0168-investigate-ultra-small-sparse-moe-for-100b-tokens.md`
- `0170-freeze-first-ultra-small-moe-routing-contract.md`
- `../reference/model_geometry.md`
