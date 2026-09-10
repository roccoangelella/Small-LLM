---
status: accepted
date: 2026-09-10
supersedes: null
owners: [Small-LLM]
---

# ADR 0169: preserve the dense reference architecture wherever MoE sparsity does not require a change

## Context and problem statement

ADR 0168 authorizes investigation of an ultra-small, many-expert MoE targeted at a 100B-token training budget, but deliberately leaves the final geometry open. The next design goal is scientific interpretability: avoid changing unrelated architectural variables while introducing sparsity, so that differences can be attributed as cleanly as possible to the tokenizer/vocabulary and MoE FFN mechanism rather than to a wholesale backbone redesign.

The completed dense Small-LLM family already supplies validated hybrid GDN-2/gated-MHA blocks, normalization, attention geometry, residual layout, SwiGLU FFNs, tied embeddings, context length, and positional encoding. The 20M dense geometry is also the natural small-scale reference candidate because it uses `d_model=256`, eight decoder layers and `d_ff=704`, matching the shared-backbone scale considered in the ultra-small MoE proposal.

## Considered options

- Redesign the entire backbone jointly with the MoE in order to minimize active parameters as aggressively as possible.
- Preserve only a few high-level motifs while freely changing width, depth, attention/GDN geometry and normalization.
- Preserve the dense reference architecture and numerical geometry wherever sparsity does not technically require a change, and make every departure explicit.

## Decision outcome

Chosen option: **preserve the dense reference architecture and parameters wherever possible; changes must be limited to components needed for the ultra-small sparse experiment and separately justified.**

This is a control principle, not yet the final MoE geometry. In particular:

- the established dense hybrid block structure, residual/pre-norm layout, normalization semantics, GDN-2 and gated-MHA definitions, positional encoding, bias/dropout policy, and context semantics should remain unchanged unless a later accepted decision identifies a concrete incompatibility;
- the custom vocabulary/tokenizer remains an intentional experimental change under ADR 0168 and must be accounted for separately;
- the dense FFN provides the reference active-compute geometry for sizing routed experts; expert count, Top-K, expert granularity, shared experts and routing remain to be selected explicitly;
- the completed 20M dense geometry is the default candidate reference for the ultra-small model because it is the smallest completed geometry with the proposed `d_model=256` / 8-layer backbone, but the exact final parameter table is to be frozen in a subsequent architecture decision before implementation;
- no implementation is authorized by this ADR alone.

## Consequences

### Positive

- The experiment has a clean architectural counterfactual instead of confounding MoE sparsity with a new mixer/backbone design.
- Existing GDN-2, attention, normalization and residual behavior can be reused and compared directly.
- Expert granularity can be defined relative to the known dense SwiGLU width rather than chosen as an unrelated hidden dimension.

### Negative or limiting

- The architecture may not be the absolute smallest possible model because width/depth reductions are no longer free optimization knobs.
- A custom tokenizer still creates a non-MoE confound relative to historical GPT-2-tokenized dense checkpoints; tokenizer-independent corpus accounting remains required.
- If the final many-expert kernel imposes geometry constraints, any necessary departure from the dense reference will require a separate explicit decision.

## Validation

Before wiring, freeze a component-by-component table marking each field as identical to the dense reference, intentionally changed for tokenization, or MoE-specific. Recompute exact active and total parameter counts from code and verify that the selected expert granularity preserves the intended dense-reference active FFN budget.

## Links

- [`0168-investigate-ultra-small-sparse-moe-for-100b-tokens.md`](0168-investigate-ultra-small-sparse-moe-for-100b-tokens.md)
- [`../reference/model_architecture.md`](../reference/model_architecture.md)
- [`../reference/model_geometry.md`](../reference/model_geometry.md)
