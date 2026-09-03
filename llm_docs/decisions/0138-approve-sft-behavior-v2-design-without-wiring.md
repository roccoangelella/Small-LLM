---
status: accepted
date: 2026-09-03
---

# 0138 — Approve SFT Behavior v2 design without wiring

## Decision

Approve the proposed SFT Behavior v2 evaluation design described in the 2026-09-03 planning discussion, but do not implement or wire it yet.

The approved design direction includes:

- a substantially enlarged mechanically scored SFT behavior benchmark;
- paired underlying tasks across capability-only and progressively constrained variants so capability failures can be separated from instruction-following failures;
- explicit constraint-level scoring rather than only aggregate pass/fail;
- preserved legacy SFT behavior v1 results for longitudinal comparison;
- separate diagnostic and held-out qualification partitions;
- greedy decoding as the primary qualification protocol, with sampled robustness kept secondary;
- code-verifiable scoring as the primary evaluation method rather than an LLM judge;
- statistical paired comparison between checkpoints rather than relying only on raw pass-rate deltas.

The detailed implementation remains intentionally unwired. No evaluator code, dataset artifact, launcher surface, or training/decontamination path is changed by this ADR.

## Rationale

The existing 30-case SFT behavior suite spreads a small number of probes across many categories, which is useful for smoke testing but too sparse for strong diagnosis. The approved v2 direction is intended to distinguish missing base capability from failures of instruction compliance and to provide per-constraint evidence before changing future SFT recipes.

## Consequences

- Future implementation work may realize this approved design in a dedicated phase.
- Until that phase is explicitly authorized, current SFT evaluation behavior remains unchanged.
- Historical v1 metrics must remain available and must not be silently redefined.
