---
status: accepted
date: 2026-09-10
supersedes: []
superseded_by: []
---

# 0174 — Adopt the owner-accepted MoE router numerical contract

## Context and problem statement

ADR 0173 wired the accepted 64-expert Top-2 geometry but deliberately left the router score
function open, because the `main` history carried two accepted decisions on it and neither was
reachable from any branch tip. Reading both in full settles that: they are **sequential, not
contradictory**. The routing ADR (`433039d`, 08:53) chose sigmoid affinities; the numerical ADR
(`e0621b6`, 10:43) is later and more specific and replaces the score with `sqrt(softplus(z))`,
keeping every other clause. There is one accepted contract, and Edo confirmed on 2026-09-10, with
Rocco present, that it is adopted in full.

Both source ADRs were subsequently removed from `main`'s tree under `main`'s own ADR 0172
(experiment memory belongs on the dedicated branch) and never landed on that branch, so they are
currently reachable only through deleted history. That is a process defect the owners know about;
this ADR records the content so the contract has a live home.

## Considered options

- Treat the two source ADRs as contradictory and ask the owners to choose. Rejected after reading
  both: the later one is more specific and supersedes only the score, so there was nothing to choose.
- Keep the softmax scoring already implemented on this branch and defer the change. Rejected: the
  owners confirmed the accepted contract in full, and the score function is not a free parameter.
- Adopt the contract but invent the missing Quantile Balancing estimator. Rejected: the routing ADR
  lists that estimator among the things still to be specified, so an invented controller would have
  looked accepted while being unreviewed.
- Adopt everything specified and fail closed on the estimator until its owner supplies it. Chosen,
  and the owner supplied it the same day: the per-expert score quantile at rate K/E.

## Decision outcome

Adopt the accepted contract for configuration version 3:

- **Affinity `s = sqrt(softplus(z))`**, elementwise, with `z = W_r x`. No cross-expert
  normalisation before selection.
- **Selection `TopK(s + b)`**, mixture weights from the **unbiased** `s`, renormalised over the
  selected set. The balancing bias steers selection only and never scales an expert's output.
- **No classical softmax z-loss.** There is no partition function to regularise, so the router
  returns an exact zero and the configuration refuses a non-zero coefficient.
- **Router matrix initialised `Normal(0, 0.02^2)`**, balancing bias initialised to zero.
- **Router and balancing arithmetic in FP32**, unchanged from version 2.
- **Router matrix on AdamW with `weight_decay = 0`** and no router-specific learning-rate
  multiplier. Version 2 keeps the router in the decaying group; only version 3 moves it.
- **Vocabulary 8,192.** Edo settled the open "8k" question on 2026-09-10. Parameter counts become
  144,074,648 stored and 9,987,992 active per token, the 2026-09-09 document's figures plus the
  192 x 256 extra tied entries.

Two guards follow from the mathematics rather than from the source text, and are enforced:

- **`sqrt_softplus` requires `top_k >= 2`.** With elementwise affinities and k = 1 the single
  mixture weight renormalises to exactly 1, so the router would receive no gradient from the
  language-model loss. The accepted geometry is Top-2, so this only forbids an incoherent
  combination.
**Quantile Balancing** was the one clause whose estimator no reachable document specified: the
routing ADR lists "QB estimator/update semantics" among the things a design review must still fix,
and the numerical ADR gives only the schedule. Rocco supplied the estimator on 2026-09-10 — *the
quantile of the per-expert scores at rate K/E* — and it is implemented exactly:

- For each expert, over **all tokens of one logical optimizer step**, take the score threshold at
  the target selection rate `K/E`; equivalently the `round(N*K/E)`-th largest of that expert's own
  affinities.
- Set the bias to the **mean-centred negative** of that threshold. An expert whose affinities sit
  high is pushed down until it is selected at the target rate.
- The bias is **recomputed from absolute scores each step, never accumulated**: there is no step
  size and no learning rate. It is applied from the following optimizer step.
- Exactness across gradient accumulation is preserved by fixing the target rank from the step's
  token count before the microbatch loop and carrying forward exactly that many top scores per
  expert; an element of the final top-M of the union survives every intermediate top-M. The cost is
  one `[experts, M]` FP32 frontier per layer — about 1 MiB per layer at the accepted geometry.

## Consequences

- The accepted geometry trains **with Quantile Balancing on**. The version-2 sign-based controller
  remains available and unchanged, but it is not the accepted mechanism for version 3.
- The controller is non-gradient and stateless between steps, so it cannot fight the language-model
  objective through the loss; it only moves the selection threshold. It also cannot recover from a
  collapse *within* a step, only between steps.
- The entropy telemetry for `sqrt_softplus` is computed over affinities normalised to sum to one.
  That is a diagnostic convenience, not a probability the model uses; version 2's entropy is
  unchanged.
- No version-2 behaviour moves: the frozen M0 identity keeps softmax scoring, its z-loss, its router
  initialisation and its decaying router group.

## Verification

Seven CPU tests in `tests/test_moe_accepted_geometry.py`: the affinity equals elementwise
`sqrt(softplus)` and its scores do not sum to one; selection follows `TopK(s + b)` while weights come
from unbiased `s`; the bias steers selection without biasing weights; `top_k = 1` is refused at both
the router and the configuration; a non-zero z-loss coefficient is refused; quantile balancing is
refused; the router matrix lands in the no-decay AdamW group under version 3 and in the decaying
group under version 2; and a full forward, backward and optimizer step leaves every parameter finite
with a zero z-loss and a live router gradient.

Four further tests cover Quantile Balancing: the committed bias equals the mean-centred negative of
the `K/E` score quantile computed independently in the test; the bias is bit-identical whether the
step arrives in 1, 4 or 9 microbatches; applying it moves per-expert selection counts closer to the
target rate on deliberately skewed routing; and committing without an open step raises rather than
writing a silent partial bias.
