---
status: accepted-with-open-items
date: 2026-09-10
supersedes: null
owners: [Small-LLM]
---

# ADR 0172: record initial MoE routing numerics and operational choices

## Context

ADR 0170 and ADR 0171 freeze the first ultra-small MoE geometry as the completed 20M dense backbone with MoE FFNs in all eight blocks, 64 routed experts per layer, Top-2 routing, expert hidden width 352, no shared experts, and a non-gradient Quantile Balancing controller. The remaining work is to freeze the numerical and training behavior of the router before implementation.

This ADR records the choices accepted on 2026-09-10 and explicitly separates them from items that still require clarification or a later decision.

## Accepted choices

1. **Dropless routing.** There is no expert capacity factor, overflow token dropping, or overflow rerouting in the initial model. Every Top-2 assignment is executed.
2. **Exact Quantile Balancing for the initial implementation.** Do not use the production histogram approximation initially; compute the required per-expert quantile exactly from the routing statistics available to the controller.
3. **Current selected QB cadence: one bias update per microbatch.** Statistics from microbatch `t` produce the bias used by a future microbatch, never retroactively rerouting the same microbatch. This is an explicit departure from Kimi K3's report, which pools the quantile over the whole optimizer/training step and applies the new bias on the next step. Because gradient accumulation may therefore see different routing biases inside one optimizer update, this cadence must be re-confirmed after the user explains the consequence during the mandatory understanding gate; no code is authorized by this ADR alone.
4. **Router initialization family: random normal.** The exact standard deviation remains to be frozen. The dense Small-LLM `normal` initializer currently uses mean 0 and standard deviation 0.02 for ordinary matrix weights, so matching that initializer is the natural reference candidate rather than inventing another scale.
5. **FP32 router path.** Router logits/scores, bias-adjusted selection values, Top-K decisions, selected routing weights, and Quantile Balancing arithmetic should be evaluated in FP32. Expert computation remains governed by the model/trainer mixed-precision policy.
6. **Router optimizer: AdamW with `weight_decay = 0` initially.** The exact router LR/schedule is still open; do not silently inherit a previous MoE experiment's router LR.
7. **Backend implementation is deferred.** The grouped/fused sparse execution backend will be selected later with Edo and is not part of the current architecture freeze. A simple reference backend may be used only for mathematical/correctness qualification; the long production run remains gated on a real target-hardware sparse-backend benchmark.
8. **Tokenizer scope is deferred.** The current initial vocabulary target is approximately 8,000 tokens, but tokenizer algorithm/training/data integration is a later decision. Current work should focus on the MoE itself.
9. **Initial LR-policy direction.** Use the original 20M training policy as the conservative starting reference rather than inventing a novel MoE schedule, while treating the exact long-run LR recipe as provisional and subject to later tuning. The historical full 20M one-pass recipe used hybrid Muon+AdamW, peak LR `3e-4`, WSD scheduling, minimum LR ratio `0.1`, and a warmup/stable/decay plan derived from its finite data budget. The router remains on its separately selected AdamW/zero-WD group.

## W&B observability contract

The user delegated MoE telemetry selection to the implementation design. The initial qualification must expose enough information to diagnose both routing collapse and an over-active balancing controller without logging hundreds of individual scalar series every step.

Every normal logging step should include, per MoE layer and in aggregate where meaningful:

- assignment-load mean, standard deviation, coefficient of variation, minimum/mean and maximum/mean load ratios;
- target load and absolute/relative load error;
- load entropy or equivalent normalized utilization entropy;
- number/fraction of experts receiving zero assignments in the current observation window;
- token-drop count, which must remain exactly zero under dropless routing;
- router-logit mean, standard deviation, RMS, maximum absolute value and finite/non-finite status;
- base-score mean/std plus useful quantiles appropriate to the final score function;
- selected raw-score statistics, normalized mixture-weight statistics, Top-1/Top-2 weight imbalance, and Top-K-versus-(K+1) selection margin/cutoff statistics;
- QB selection-bias mean/std/min/max/max-absolute value, bias-update L2/max-absolute delta, and mean-centering residual;
- router gradient norm and parameter norm;
- train loss, validation loss, LR, optimizer/overflow telemetry, tokens/s, step time and peak/reserved GPU memory from the common trainer.

High-dimensional diagnostics should be sampled periodically rather than emitted as 512 continuously plotted scalar streams. At validation/diagnostic cadence, log per-layer histograms or compact vectors for all 64 expert loads, routing biases, router logits/scores and selected weights, plus rolling dead/underused-expert counts. Once a production sparse backend exists, add dispatch/sort/grouped-GEMM timing and expert-group size distributions.

Qualification should fail closed on non-finite router values or any nonzero token-drop count. Thresholds for load imbalance, dead experts, bias magnitude, routing-score pathologies and routing churn are empirical qualification thresholds, not yet frozen here.

## Still open before implementation

- **Router score function:** retain sigmoid or replace it with `sqrt(softplus(z))`; the latter is under active review and is not decided by this ADR.
- **Router random-normal scale:** whether to match the dense initializer exactly at `Normal(0, 0.02)` or use another justified scale.
- **Router stabilization / z-loss:** no classical softmax-router z-loss coefficient is frozen. The choice depends partly on the final score function.
- **Exact semantics and serialization of the QB bias `b`:** its conceptual role must be explained to and understood by the user before implementation is authorized.
- **Per-microbatch versus whole-optimizer-step QB update:** per-microbatch is the current selected direction, but because it deliberately differs from Kimi K3 and changes routing within a gradient-accumulation window, it must be explicitly re-confirmed after the understanding check.
- **Exact LR schedule for the MoE production trajectory:** the original 20M recipe is the initial reference only; token horizons and any router LR multiplier remain open.

## Implementation gate

No routing/training implementation change is authorized until the user can explain in their own words: (a) how trainable router weights differ from the non-gradient QB bias, (b) where that bias enters selection and where it does not enter mixture weighting, (c) the consequence of updating QB every microbatch rather than once per logical optimizer step, and (d) the behavior of the final chosen score function.

## Links

- `0170-freeze-initial-ultra-small-moe-routing-contract.md`
- `0171-freeze-initial-moe-granularity-schedule.md`
- `../archive/20m_qualification/20m_kaggle_runbook.md`
