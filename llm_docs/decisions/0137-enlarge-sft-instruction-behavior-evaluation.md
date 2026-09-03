---
status: accepted
date: 2026-09-03
supersedes: null
---

# 0137 — Enlarge SFT instruction-behavior evaluation

## Context and problem statement

The current `post_training/sft/behavior_eval.py` suite contains 30 deterministic cases spread across many categories. The canonical 100M/2B 10% peak-through-3000 SFT model improved generation health and SFT held-out likelihood but achieved only 1/30 strict behavior passes, making the existing suite too sparse for reliable per-category diagnosis or for separating inability to solve a task from inability to obey its requested output constraints.

Recent 2026 instruction-following evaluation work motivates granular evaluation by constraint type, count, position, difficulty, and multi-turn behavior rather than treating compliance as one scalar capability.

## Considered options

- Keep the current 30-case suite as the sole SFT behavior qualification.
- Add more ad-hoc prompts without changing the diagnostic structure.
- Build a larger frozen successor designed around repeated observations, skill/constraint categories, and mechanically verifiable scoring where possible.

The exact number of cases, task families, constraint taxonomy, difficulty levels, diagnostic/qualification split, sampled-decoding protocol, external benchmark integration, and promotion thresholds remain proposed until separately approved.

## Decision outcome

Enlarge the project's SFT instruction-behavior test suite beyond the current 30-case probe before using subsequent SFT experiments to make recipe decisions.

The enlarged suite must be designed as a diagnostic instruction-following benchmark rather than merely adding more ad-hoc prompts. It should provide enough repeated observations to report meaningful per-skill and per-constraint behavior, preserve mechanically verifiable scoring where possible, and distinguish underlying task capability from instruction-compliance failure.

## Consequences

Do not treat the current 30-case strict pass rate as sufficiently granular for choosing the next SFT data recipe. Preserve it for historical comparability, but design a larger frozen successor before the next round of SFT recipe selection.

This ADR records only the accepted direction to enlarge the suite. The detailed Small-LLM v2 suite design remains a separate decision.
