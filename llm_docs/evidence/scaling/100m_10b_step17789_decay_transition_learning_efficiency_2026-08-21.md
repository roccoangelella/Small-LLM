---
status: evidence
date: 2026-08-21
---

# 100M/10B validation response at the step-17,789 LR transition

## Observation

During the live `100m-10b-deep-decay-from-step15500` continuation, the LR schedule changes at step 17,789 from the aggressive cosine settle (`3e-4 -> 1e-4` over 2,289 updates) into the long calibrated power-law phase.

The validation-loss / perplexity curve discussed on 2026-08-21 shows a clear change in learning rate around that boundary: validation loss falls rapidly through the settling phase, then the rate of improvement becomes materially shallower once the LR decay itself becomes less aggressive. Absolute validation loss still improves overall; the observed degradation is in **learning efficiency / slope**, not a sustained regression in model quality.

This is consistent with the earlier ADR-0093 aggressive-WSqD branch, where validation loss fell during the fast settle and then rose after transition to the substantially gentler inverse-square-root long phase.

## Accepted working interpretation

Treat the evidence as support for the working conclusion that, for this approximately-100M / 10B trajectory, relaxing LR decay too sharply after the initial settle harms learning efficiency. A long high-LR or very gently decaying middle phase is therefore no longer the preferred interpretation of the observed training dynamics.

The evidence does **not** isolate decay derivative as the sole causal variable: LR level, training progress, and diminishing returns change at the same time. It also does not establish the exact current power-law exponent (`p ~= 1.627`) as a universal recipe for future model sizes or token horizons.

The durable lesson to carry into future schedule design, subject to completion of the 10B run, is to prefer meaningful decay through the main training phase rather than reverting to a long flat or weakly decaying plateau. Exact peak LR, decay onset, shape/exponent, and terminal LR remain run-specific calibration questions.

## Endpoint-slope concern and forward diagnostic

A separate concern is that earlier completed trajectories, especially the completed 100M/2B endpoint cited in ADR 0093, ended while validation loss was still falling with no obvious learning slowdown. This suggests that at least some historical endpoints were imposed by token budget rather than by natural saturation of the model/data learning curve.

The current 10B deep-decay run should therefore be allowed to finish unchanged, but its later validation slope should be treated as a direct diagnostic of whether the present schedule anneals too early or too strongly while useful fresh-data learning remains available.

Useful LR landmarks under the accepted schedule are approximately:

```text
step ~37k   LR ~3e-5
step ~48k   LR ~2e-5
step ~57k   LR ~1.5e-5
step 73,242 LR 1e-5   terminal cooldown begins
step 76,294 LR 5e-6   10B endpoint
```

If validation loss is still descending with a clearly non-flat slope around steps ~37k, ~48k, and especially ~57k, treat that as evidence that the schedule may be reducing optimizer mobility substantially before the model has exhausted useful learning from the remaining fresh data. In that case, future fresh pretraining schedules should preserve more LR budget deeper into the token horizon rather than copying the current deep-decay curve literally.

Conversely, if validation loss naturally flattens before or through those landmarks, the current annealing strength gains support. The exact future schedule remains undecided until this evidence exists; in particular, neither the current `p ~= 1.627` power law nor a constant/exponential alternative is yet promoted as the project-wide standard.

## Related decisions/evidence

- ADR 0093 — front-loaded settle followed by gentler inverse-square-root continuation; it also records that the completed 100M/2B run showed no obvious late cooldown slowdown.
- ADR 0095 — current deep-decay scientific schedule (`1e-4 -> 1e-5` calibrated power law, then `5e-6` terminal cooldown).
- ADR 0114 — current one-H100 Modal execution of the unchanged ADR-0095 scientific schedule.
