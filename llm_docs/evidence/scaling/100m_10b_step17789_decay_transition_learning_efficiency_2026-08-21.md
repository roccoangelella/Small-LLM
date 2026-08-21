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

## Related decisions/evidence

- ADR 0093 — front-loaded settle followed by gentler inverse-square-root continuation.
- ADR 0095 — current deep-decay scientific schedule (`1e-4 -> 1e-5` calibrated power law, then `5e-6` terminal cooldown).
- ADR 0114 — current one-H100 Modal execution of the unchanged ADR-0095 scientific schedule.
