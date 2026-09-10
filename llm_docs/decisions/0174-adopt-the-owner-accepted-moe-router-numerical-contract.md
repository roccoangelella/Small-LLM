---
status: accepted
date: 2026-09-10
supersedes: []
superseded_by: []
---

# 0174 — Adopt the owner-accepted MoE router numerical contract

## Context

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

## Decision

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
- **Quantile Balancing fails closed.** It is accepted by both source ADRs, but the routing ADR
  itself lists "QB estimator/update semantics" among the things a design review must still specify,
  and the numerical ADR fixes only the *schedule* — aggregate statistics over the entire logical
  optimizer step, one fixed bias across gradient-accumulation microbatches, compute the exact
  quantile after the complete step, apply the mean-centred bias on the next step. The estimator
  itself is not specified anywhere reachable. Selecting `load_balancing="quantile"` therefore raises
  rather than running an invented controller. **This is the one open item of the contract.**

## Consequences

- Training the accepted geometry today runs **without load balancing**. The existing sign-based
  controller from version 2 remains available but is not the accepted mechanism, and expert collapse
  is unmitigated until QB is specified. This is a real risk for a long run and is why the failure is
  loud rather than silent.
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
