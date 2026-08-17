---
status: accepted
date: 2026-08-15
---

# ADR 0090 — Reuse S0 retention stratification and keep R-SFT one-pass

## Context and problem statement

The R0 generator, strict JSON schema, 90/10 reasoning-versus-retention mixture, serialization path, and R-SFT tokenizer are now defined. Before the first production R-SFT run, the remaining training-shape questions include where the 10% retention slice comes from, whether R-SFT should introduce a new trainer/launcher architecture, and whether the first run should repeat the new corpus for multiple epochs.

The overall R-SFT token budget is still under discussion. In particular, a possible 15%-of-parent budget has been raised but is not frozen by this ADR.

## Considered options

1. Build a new retention mixture, a separate R-SFT training stack, and tune the number of epochs independently.
2. Reuse the existing S0 instruction distribution for retention, reuse the qualified SFT trainer/launcher machinery wherever the objective is shared, and keep the first R-SFT run to one pass over its frozen bundle.
3. Omit retention and train only on newly generated reasoning examples.

## Decision outcome

Choose option 2.

- The already-frozen top-level mixture remains 90% R0 reasoning targets and 10% instruction-retention targets.
- The retention 10% is sampled from the same S0 instruction data the model already saw. Preserve the S0 source/topic stratification when taking the smaller retention slice rather than inventing a new retention distribution.
- The R-SFT training entry point should follow the existing simple SFT implementation for the major operational pieces: verified parent loading, immutable tokenized bundle consumption, masked next-token training, WSD trainer integration, checkpoint cadence, exact resume, Hugging Face publication, W&B telemetry, and the established Kaggle/SFT execution structure. R-SFT-specific code should be limited to the genuinely different data/tokenizer/model-transition/stage identity pieces.
- The first R-SFT experiment is one pass over its frozen training bundle. Do not obtain a larger apparent token budget by repeating the same generated examples for multiple epochs.
- The final R-SFT token budget and production examples-per-cell remain open until the live generation pilot measures the actual concise-reasoning token distribution and generation quality.

## Consequences

- Retention directly tests preservation of the instruction behavior already taught in S0 rather than introducing a confounded new instruction mixture.
- R-SFT remains operationally comparable with S0 and avoids a second trainer stack that would make checkpointing or hardware behavior harder to compare.
- One-pass training keeps dataset size and consumed loss-bearing targets aligned, so a chosen R-SFT budget must be backed by actual corpus tokens rather than silent data repetition.
- A proposed 15%-of-pretraining R-SFT budget is not yet an accepted decision; generation volume and empirical token length must be measured first.
