---
status: accepted
date: 2026-09-10
supersedes: []
superseded_by: []
---

# 0173 — Wire the accepted 64-expert Top-2 geometry with a batched expert GEMM

## Context

The owner fixed the architecture on 2026-09-09: **64 routed experts per block, Top-2, no shared
expert**, on a base of 8 layers, width 256, SwiGLU experts of width 352 and an 8,000-entry tied
vocabulary. The code did not implement it and could not: `MoEModelConfig` raised on anything other
than 8 experts and Top-1, which was the correct fail-closed contract for the frozen M0 pilot and a
hard stop for the decided geometry.

Two facts shaped the implementation. First, the 2026-09-10 A/B on a Modal A10 measured that a
training update is dominated by kernel launches, not arithmetic
(`../evidence/moe_execution_ab_modal_a10_2026-09-10.md`). Second, the M0 dispatch executes one
expert at a time; at 8 experts over 20 layers that is 5,120 expert invocations per update, and the
accepted geometry would issue 16,384 across 8 layers — more groups, each smaller, in precisely the
regime the measurement says is launch-bound. Wiring the geometry without addressing the loop would
hand the pilot a known bottleneck, and ADR 0168 on `main` already requires the many-expert path to
be benchmarked before any long sparse run.

## Decision

Introduce **configuration version 3**, which opens exactly two axes — expert count and `top_k` —
and decouples expert width from the dense FFN width. Version 2 keeps every one of its constraints,
so no M0 checkpoint or identity moves.

Implement the geometry as:

- **`SwitchTopKRouter`** — softmax scores, `topk` selection over score plus balancing bias, and
  mixture weights renormalised over the selected experts. `SwitchTop1Router` subclasses it pinned to
  k = 1 and drops the trailing axis, so the M0 return contract is unchanged; at k = 1 `topk(1)` and
  `argmax` select the same expert, including the tie-break.
- **Stacked expert weights** — one parameter per projection, shaped `[experts, in, out]`, instead of
  one module per expert.
- **`DroplessTopKMoE`** with two interchangeable dispatches over those same parameters: the
  production `dropless_padded_batched_gemm`, which sorts tokens by expert, writes them into a dense
  `[experts, capacity, d_model]` buffer and evaluates every expert in three `bmm` calls; and
  `dropless_grouped_by_expert`, the per-expert loop kept as the measurement control.
- **Muon accepts stacked weights.** `_muon_group_step` now views every parameter as a batch of
  `[m, n]` matrices — a plain weight is one, a stacked expert weight is as many as it has experts —
  buckets them by shape and runs one batched Newton–Schulz per bucket. The per-matrix arithmetic is
  unchanged.

Recombination is deterministic by construction: the permutation is inverted with `index_copy_` onto
unique destinations and a token's k contributions are summed along a dedicated axis, so no atomics
are used and repeated runs are bit-identical.

## Consequences

- The parameter count is an independent check and it passes exactly: **144,025,496 stored,
  9,938,840 active per token**, matching the arithmetic in the 2026-09-09 architecture options
  document, which was computed by a separate script.
- Padding is the cost of batching. The buffer is sized by the largest expert load in the batch, so
  wasted arithmetic is proportional to routing imbalance. With a near-uniform router it is small;
  under collapse it approaches a factor of `num_experts`. This is a measurable quantity, not an
  assumption, and the loop dispatch exists to price it.
- Checkpoints of a version-3 model are not loadable as version 2 and vice versa: the expert
  parameter names and shapes differ by design.
- **The router score function remains an open owner decision.** The `main` history records two
  accepted and mutually inconsistent contracts for the same experiment — sigmoid scores, and
  `sqrt(softplus(z))` scores — and neither file is on any branch tip. This ADR implements softmax,
  which is what this branch already used, and confines the disagreement to the score: selection,
  normalisation, dispatch, shapes and every systems measurement are identical under all three.
- Load-balancing stays off (`balancing_step_size = 0`). With elementwise scores and Top-k selection
  the unselected router rows still receive gradient through the softmax, but expert balance is not
  guaranteed; that is a training question for the pilot, not an execution one.

## Verification

Thirteen CPU tests in `tests/test_moe_accepted_geometry.py`: the accepted configuration matches the
document; version 2 still rejects each opened axis; version 3 rejects incoherent routing; the
parameter counts equal the independently computed totals; the Top-1 router is value- and
shape-identical to the frozen one; Top-k weights are normalised and counts equal assignments; the
selection bias steers selection without biasing weights; the batched dispatch matches a
one-expert-at-a-time oracle at k = 1, 2 and 3 and under deliberate routing collapse; the two
dispatches agree on identical weights; output is deterministic across calls; gradients reach every
expert and the router; stacked weights are routed to Muon and orthogonalised slice by slice; and a
full model step leaves every parameter finite.
