---
status: accepted
date: 2026-08-18
---

# ADR 0091 — Generate a 630-example R0 pilot and run a matched delimiter ablation

## Context and problem statement

The first live R0 teacher-generation pilot needs to be large enough to measure actual target-token lengths and inspect variability across every frozen skill × difficulty cell before the production R-SFT corpus size is chosen. The project also still owes the small textual-versus-atomic reasoning-delimiter ablation required by the reasoning-token decision.

The R0 matrix has seven reasoning skills and three difficulty bands, for 21 cells. Teacher generation is already batched at approximately 10 examples per API call.

## Decision outcome

- Generate **30 Gemini examples per skill × difficulty cell** for the first live pilot.
- This is **630 generated reasoning examples total** (`7 skills × 3 difficulties × 30 examples`).
- Keep generation uniform across all 21 cells and globally shuffled after schema-valid generation, consistent with the existing R0 data policy.
- With the current batch size of 10 examples per teacher request, the planned pilot is **63 Gemini API calls**: three calls per cell.
- Treat 630 examples as a measurement/pilot corpus, not as evidence that the final R-SFT production token budget has been selected. Measure actual serialized target-token statistics before freezing the larger corpus size.

Run the previously planned reasoning-delimiter ablation on Kaggle T4 GPUs. The two arms must be matched on parent S0 checkpoint, underlying reasoning examples, train/held-out partition, optimizer/training schedule, random seed, number of passes, and evaluation procedure. The intended independent variable is only the delimiter/tokenization representation:

1. **Textual arm:** ordinary GPT-2-tokenized textual reasoning/answer delimiters such as `Reasoning:` / `Answer:`; no promoted atomic reasoning-token IDs are used for those boundaries.
2. **Atomic arm:** the three promoted R-SFT control-token IDs are used for reasoning start, reasoning end, and answer start, with the S0→R-SFT model transition initializing only those three promoted rows.

The ablation is deliberately small and is not a second production-scale R-SFT run. Kaggle T4 availability means compute scarcity should not be used to bias one arm toward a smaller training budget. The production choice should prefer atomic tokens if their learning/behavior is comparable or better, because they provide an explicit machine-readable reasoning-control interface; a material degradation would justify retaining textual delimiters or revisiting the atomic cold start.

## Consequences

- The immediate teacher-generation command is parameterized by `examples_per_cell=30`, producing 630 examples and 63 requests at batch size 10.
- The pilot provides a three-batch sample inside every cell, reducing the risk of sizing the production corpus from one unusually short or homogeneous batch.
- The delimiter comparison remains causally interpretable because dataset content and training conditions are held fixed between arms.
- The final production R-SFT token percentage remains intentionally open pending pilot token statistics and quality inspection.
