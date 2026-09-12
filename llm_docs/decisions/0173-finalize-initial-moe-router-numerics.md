---
status: accepted
date: 2026-09-10
supersedes: null
owners: [Small-LLM]
---

# ADR 0173: finalize initial MoE router numerics

## Context and problem statement

ADR 0170-0172 define the first ultra-small MoE experiment on the completed 20M dense backbone. The initial sparse geometry is 64 routed experts per MoE layer, Top-2, expert hidden width 352, no shared experts, dropless routing, FP32 router arithmetic, and exact Quantile Balancing (QB). ADR 0172 had intentionally left several numerical details open and had provisionally recorded a per-microbatch QB update cadence pending review.

The final pre-implementation review compared the relevant 2026 designs, especially Kimi K3's step-level Quantile Balancing and DeepSeek-V4's Sqrt(Softplus) affinity scoring. The user then accepted the choices below.

## Considered options

No alternative was recorded at the time; section added for the template.

## Decision outcome

1. **Quantile Balancing is accumulated over the complete logical optimizer step, not updated after each microbatch.** Every microbatch contributing gradients to optimizer step `t` uses the same balancing bias `b_t`. Routing statistics/margins are accumulated across those microbatches, the exact per-expert quantile is computed after the complete logical step, the resulting bias is mean-centered, and the new `b_{t+1}` is used only by the next optimizer step. This supersedes ADR 0172's provisional per-microbatch direction.

2. **Router affinity scores use Sqrt(Softplus), not sigmoid.** For router logits `z = W_r x`, use

   `s = sqrt(softplus(z))`.

   Expert selection is `TopK(s + b)`. The balancing bias affects selection only. After Top-K selection, mixture coefficients are computed from the underlying unbiased positive scores `s` and normalized across the selected experts. No balancing bias enters the mixture weights.

3. **No classical softmax-router z-loss is used initially.** The selected score function does not use a softmax partition function. Stability is instead diagnosed directly from router logits/scores, gradients, mixture weights, and QB-bias telemetry. A later explicit experiment may add another regularizer if evidence demands it.

4. **Router weights use random-normal initialization with `std = 0.02`, mean zero.** The chosen distribution is `W_r ~ Normal(0, 0.02^2)`. The non-gradient QB bias `b` starts at exactly zero.

   Rationale: DeepSeek-V4 is the closest current score-function reference because it explicitly replaces sigmoid affinity scoring with Sqrt(Softplus), and its released Transformers configuration uses `initializer_range = 0.02` with normal initialization for the router weight. The released Kimi K3 text implementation is an important counterexample: its custom MoE gate resets the router matrix with Kaiming-uniform rather than the global 0.02 normal initializer. That choice is fan-in-scaled and accompanies sigmoid scoring, so it is not copied blindly. For Small-LLM's `d_model=256`, `std=0.02` also matches the project's existing dense GPT-style normal initializer and gives an estimated initial router-logit standard deviation of about `sqrt(256)*0.02 = 0.32` for unit-RMS normalized inputs, keeping Sqrt(Softplus) in a non-saturated, gently symmetry-breaking regime.

5. **Router arithmetic remains FP32.** Router logits, Sqrt(Softplus) scores, bias-adjusted selection values, Top-K decisions, selected mixture weights, and QB arithmetic are all computed in FP32. Expert compute follows the common mixed-precision trainer policy.

6. **The trainable router matrix is optimized by AdamW with zero weight decay.** Initially use the same instantaneous LR schedule as the common training recipe, i.e. no router-specific LR multiplier (`1.0x`) unless qualification telemetry justifies a later change.

7. **Initial training-policy reference remains the original 20M recipe.** Use the existing hybrid Muon+AdamW model optimization policy as the conservative baseline, with peak LR `3e-4`, WSD scheduling, max-grad-norm `1.0`, and minimum-LR ratio `0.1`. Router parameters remain in their separate AdamW/zero-WD group. Exact warmup/stable/decay token horizons for the eventual long MoE run are not frozen by this ADR and must be scaled/tuned deliberately rather than copied as fixed 20M percentages.

8. **The sparse execution backend remains deferred to a later systems decision with Edo.** A correctness/reference implementation does not authorize a long production run; the production grouped/fused backend still requires target-hardware qualification.

9. **Tokenizer implementation remains deferred.** The current architectural target is an approximately 8k-token vocabulary, but tokenizer algorithm/training/integration are outside the present MoE-routing freeze.

## Evidence and nuance

- Kimi K3's QB design pools routing statistics across the full training/optimizer step (including accumulation) and applies the new bias on the next step. This is the behavior adopted here because it preserves one routing policy across all microbatches that form a single optimizer gradient and supplies a larger, less noisy quantile sample.
- DeepSeek-V4 changes its MoE affinity activation from sigmoid to Sqrt(Softplus), while retaining the principle that the load-balancing correction affects Top-K selection but not the expert-combination weights. This motivates the chosen score path.
- The public Kimi K3 implementation initializes its custom gate with Kaiming-uniform, even though the global model config exposes `initializer_range=0.02`. Therefore `0.02` should not be described as Kimi K3's router initializer. It is selected here primarily because it is the direct DeepSeek-V4/Sqrt(Softplus) reference and matches Small-LLM's established normal initializer.

## Validation and telemetry

Retain ADR 0172's W&B observability contract: per-layer and aggregate load dispersion, target-load error, utilization entropy, zero-load experts, router-logit/score statistics, selected-weight and Top-K margin statistics, QB-bias and bias-update statistics, router gradient/parameter norms, loss/LR/overflow/throughput/memory metrics, and periodic compact per-expert histograms/vectors. Non-finite router values or any nonzero token-drop count are fail-closed conditions.

## Implementation gate

This ADR freezes design choices but does **not** authorize wiring yet. Before code changes, the user must explain in their own words: (a) the difference between the trainable router matrix `W_r` and the non-gradient QB bias `b`; (b) why `b` appears in `TopK(s+b)` but not in the selected-expert mixture coefficients; (c) why step-level QB keeps one routing policy across gradient-accumulation microbatches and applies the new bias only to the next optimizer step; and (d) what Sqrt(Softplus) changes relative to sigmoid, including the fact that it reduces positive-side saturation without making gradients constant.

## References

- Kimi Team, *Kimi K3: Open Frontier Intelligence*, arXiv:2607.24653 (2026).
- DeepSeek-AI, *DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence*, arXiv:2606.19348 (2026).
- `0170-freeze-initial-ultra-small-moe-routing-contract.md`
- `0171-freeze-initial-moe-granularity-schedule.md`
- `0172-record-initial-moe-routing-numerics-and-operations.md`
## Consequences

No consequences were recorded at the time; section added for the template.
