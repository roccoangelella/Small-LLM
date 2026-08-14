---
status: accepted
last_reviewed: 2026-08-14
---

# ADR 0080 — Use target reasoning depth without hard trace ceilings

## Decision

Reasoning-SFT data generation will use explicit target reasoning depth as a teacher-generation control, but the requested depth is a structural target rather than a hard output-length ceiling.

For each synthetic reasoning example, the generation specification passed to the teacher should include the intended reasoning band and an approximate number of necessary dependent inference steps. Gemini should be instructed to produce a concise, complete reasoning path matching that target when possible.

Do not truncate or reject an otherwise valid reasoning trajectory solely because it requires more steps than the nominal band. A solution may exceed the requested target when the additional steps are genuinely necessary for a complete and correct derivation. Conversely, verbose restatement, redundant branches, filler, and repeated verification do not count as additional reasoning depth.

The difficulty label remains metadata describing intended structural complexity, not a token-count bucket. Training examples remain globally shuffled within any given reasoning-SFT stage.

## Rationale

The project wants difficulty to represent dependent inference depth rather than response length. Recent reasoning-distillation work warns both against indiscriminately long teacher traces that exceed student capacity and against naive truncation that removes necessary reasoning. Asking the teacher for a target number of essential steps gives the dataset generator direct control over reasoning depth while preserving complete solutions.

## Deferred

This ADR does not decide whether the full reasoning-depth range should be taught in one mixed R-SFT corpus or in multiple qualified R-SFT stages. That sequencing remains a separate evidence-backed decision.
