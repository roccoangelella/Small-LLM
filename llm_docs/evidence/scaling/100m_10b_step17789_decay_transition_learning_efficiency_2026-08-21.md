---
status: evidence
date: 2026-08-21
---

# 100M/10B validation response at the step-17,789 LR transition

## Observation

During the live `100m-10b-deep-decay-from-step15500` continuation, the LR schedule changes at step 17,789 from the aggressive cosine settle (`3e-4 -> 1e-4` over 2,289 updates) into the long calibrated power-law phase.

A direct W&B history inspection on 2026-08-21 covered entity `rocchissimo936-none`, project `Small-LLM`, primary run ID `100m-10b-deep-decay-from-step15500`, through latest validation checkpoint step 30,250. Exact metrics inspected were `validation/loss`, `validation/perplexity`, `train/learning_rate`, `train/loss`, `trainer/global_step`, `train/consumed_tokens`, and `train/target_tokens`.

The validation curve shows a sharp regime change around the schedule boundary. During the aggressive settle, an OLS fit over validation checkpoints 15,750–17,750 gives approximately `-0.04371` validation loss per 1,000 optimizer steps (`R^2 ~= 0.990`). Immediately after the transition, 18,000–20,000 gives only about `-0.00125` per 1,000 steps (`R^2 ~= 0.379`), roughly 35x slower. Broader post-transition fits remain much shallower: 18,000–26,500 is about `-0.00494` per 1,000 steps and 18,000–30,250 about `-0.00380` per 1,000 steps.

Important checkpoints include:

```text
step 15,750  targets 2.064B  LR 2.942e-4   val loss 3.03999
step 16,750  targets 2.195B  LR 1.856e-4   val loss 2.99824
step 17,500  targets 2.294B  LR 1.078e-4   val loss 2.96108
step 17,750  targets 2.327B  LR 1.0014e-4  val loss 2.95651
step 17,789  targets 2.332B  LR 1.0000e-4  schedule transition; no validation row
step 18,000  targets 2.359B  LR 9.810e-5   val loss 2.95345
step 20,000  targets 2.621B  LR ~8.265e-5  val loss 2.95044
step 24,000  targets 3.146B  LR ~6.143e-5  val loss 2.92291
step 26,500  targets 3.473B  LR ~5.228e-5  val loss 2.91344
step 29,000  targets 3.801B  LR ~4.515e-5  val loss 2.91912
step 30,000  targets 3.932B  LR ~4.273e-5  val loss 2.91222
step 30,250  targets 3.965B  LR 4.2155e-5  val loss 2.91225
```

The post-transition curve is not monotonically improving. From roughly step 26,500 through 30,250, medium-window regression is effectively flat/noisy; a shorter 29,000–30,250 window shows a renewed downward move around `-0.00567` loss per 1,000 steps, but the latest 30,250 validation point is essentially unchanged from 30,000. Therefore the current state should not be described as a robust medium-term steep downward trend.

This remains consistent with the earlier ADR-0093 aggressive-WSqD branch, where validation loss fell during its fast settle and then rose after transition to the substantially gentler inverse-square-root long phase.

## Historical endpoint evidence

The completed `100m-2b-data-001` run provides an important counterexample to the idea that low absolute LR itself is starving learning. Its final WSD cooldown drives LR toward `3e-5`, yet validation loss remains strongly descending at the token-budget endpoint. OLS tail fits are approximately:

```text
steps 12,250–15,267  val-loss slope ~-0.04224 / 1k steps
steps 12,500–15,267  val-loss slope ~-0.04281 / 1k steps
steps 13,250–15,267  val-loss slope ~-0.04123 / 1k steps
steps 14,500–15,267  val-loss slope ~-0.02494 / 1k steps
```

The 12,500–15,267 regression has `R^2 ~= 0.986`. The completed `20m-2b-data-001` tail similarly falls materially, around `-0.0117` loss per 1,000 steps over steps 53,750–61,066 (`R^2 ~= 0.97`). These runs therefore ended because of their token budgets, not because validation loss had naturally saturated.

Crucially, the strongest late improvement in `100m-2b-data-001` occurs while LR is being reduced aggressively. This argues against the simple hypothesis that Small-LLM is generally annealing to an absolute LR that is too small. It instead strengthens the competing hypothesis that **continued/stronger LR decay itself may be beneficial while a flatter or weakly decaying phase is less efficient**.

## Revised working interpretation

Treat the evidence as support for the working conclusion that, for this approximately-100M trajectory, **relaxing the LR decay rate sharply after an aggressive decay phase is associated with a major reduction in validation learning efficiency**.

This is stronger than the earlier formulation that the schedule may simply be reducing LR too early or too strongly. The direct W&B evidence points in the opposite direction for absolute LR: both the 100M/10B settle and the 100M/2B terminal cooldown improve fastest while LR is falling aggressively. A long high-LR or very gently decaying middle phase is therefore increasingly disfavored by project evidence.

The evidence still does **not** isolate the LR-decay derivative as the sole causal variable. Training age, consumed-token count, LR level, and diminishing returns change simultaneously, and no matched counterfactual continuation has yet held `1e-4`, continued a steeper decay, or used another schedule over the identical fresh-data window.

Also do not extrapolate the exact 15,500–17,789 multiplicative decay rate across the full remaining 10B horizon: copying that rate literally would drive LR near zero far too early. The open schedule-design question is instead how to preserve meaningful/possibly stronger decay over the long horizon without exhausting the LR budget prematurely.

## Forward diagnostic

The current 10B run should finish unchanged. Useful LR landmarks remain:

```text
step ~37k   LR ~3e-5
step ~48k   LR ~2e-5
step ~57k   LR ~1.5e-5
step 73,242 LR 1e-5   terminal cooldown begins
step 76,294 LR 5e-6   10B endpoint
```

Step ~37k is especially informative because `~3e-5` is comparable to the successful terminal LR region of `100m-2b-data-001`. If validation improvement strengthens again as the current run approaches that LR, it would support the hypothesis that stronger ongoing decay is useful. If validation remains flat despite another roughly 0.9B fresh targets, ordinary model/data saturation becomes a more plausible explanation for the post-17,789 slowdown.

The ~48k and ~57k landmarks then test LR regimes below those reached by the completed 100M/2B schedule. Neither the current `p ~= 1.627` power law nor a constant/exponential alternative is yet promoted as the project-wide standard.

## Related decisions/evidence

- ADR 0093 — front-loaded settle followed by gentler inverse-square-root continuation; historical evidence that the gentler phase degraded after the fast settle.
- ADR 0095 — current deep-decay scientific schedule (`1e-4 -> 1e-5` calibrated power law, then `5e-6` terminal cooldown).
- ADR 0114 — current one-H100 Modal execution of the unchanged ADR-0095 scientific schedule.
