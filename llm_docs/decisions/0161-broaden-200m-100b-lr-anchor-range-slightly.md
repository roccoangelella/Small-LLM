---
status: accepted
date: 2026-09-08
supersedes: null
owners: [Small-LLM]
---

# ADR 0161: slightly broaden the LR anchor range for the 200M/100B run

## Context and problem statement

The completed 100M/10B deep-decay trajectory provides the project's strongest learning-rate evidence. Its successful schedule used the approximate anchor sequence `3e-4 -> 1e-4 -> 1e-5 -> 5e-6`, and measured validation learning was strongest while LR was being reduced aggressively rather than held high or decayed only gently.

The next major run is approximately 200M parameters trained on 100B target tokens under ADR 0160. Because this horizon is ten times longer than the completed 100M/10B trajectory, simply reusing the same numerical LR anchors unchanged may constrain the useful LR operating range too tightly over a much longer optimization path.

## Considered options

- Reuse the exact 100M/10B numerical LR anchors unchanged.
- Scale all LR values substantially upward/downward for the longer run.
- Keep the successful 100M/10B schedule family and ordering, but widen each numerical LR anchor only modestly.

## Decision outcome

Chosen option: **retain the successful deep-decay / WSqD-style LR structure, but use a slightly broader numerical LR range at every major anchor for the 200M/100B run.**

The intended direction is:

- peak LR: modestly higher than the 100M/10B `3e-4` peak;
- first aggressive-settle endpoint: modestly lower than `1e-4`;
- late long-decay anchor: modestly lower than `1e-5`;
- terminal LR: modestly lower than `5e-6`.

This decision freezes the qualitative direction only. It does **not** yet freeze exact LR values, phase token boundaries, decay exponent, warmup span, or terminal-cooldown span.

The widening must remain small rather than turning into a new high-variance LR experiment. Exact values must be selected explicitly from the 100M evidence and the 200M/100B horizon before implementation.

## Consequences

### Positive

- The new run preserves the empirically successful LR ordering and deep-decay behavior from 100M/10B.
- The wider range gives the 100B trajectory additional headroom both for early optimization and for very-late low-LR refinement.
- The change is deliberately incremental rather than a wholesale scheduler redesign.

### Negative or limiting

- The exact widened anchors are not yet experimentally validated on the 200M geometry.
- A higher peak can increase instability risk, while lower late anchors can waste tokens if reached too early.
- Phase timing and numerical anchors must therefore be designed together rather than independently.

## Validation

Before the production run is wired:

1. choose exact widened LR anchors and phase boundaries;
2. verify that the schedule remains continuous and reaches the exact block-aligned 100B endpoint;
3. run short 200M stability/learning probes that exercise the selected peak and early settle behavior;
4. confirm finite gradients, acceptable clipping/overflow behavior, and improving validation loss;
5. preserve exact scheduler state in checkpoints for deterministic resume.

## Links

- `llm_docs/decisions/0160-select-200m-100b-as-next-pretraining-run.md`
- `llm_docs/decisions/0095-decay-1e-4-to-1e-5-then-5e-6.md`
- `llm_docs/evidence/scaling/100m_10b_step17789_decay_transition_learning_efficiency_2026-08-21.md`
- `llm_docs/reference/optimizer_strategy.md`
