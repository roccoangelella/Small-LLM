---
status: accepted
date: 2026-09-04
owners: [Small-LLM]
supersedes: []
implements:
  - 0140-wire-evaluation-v2-and-retire-fixed-length-qualitative-protocol
---

# ADR 0149: require genuinely unique Base Prompt v2 cases

## Decision

Correct the active Base Prompt v2 definition so that its advertised 120-prompt full suite is composed of 120 genuinely distinct prompt texts and 120 distinct case IDs.

The full suite remains structurally unchanged at the protocol level:

- 100 mechanically scored prompts;
- five scored families: factual, arithmetic, extraction, classification, transformation;
- exactly 20 scored prompts per family;
- 20 readable qualitative continuations;
- native per-case generation budgets;
- greedy decoding at `temperature=0`, `top_p=1`, `top_k=0`;
- sampled decoding at `temperature=1`, `top_p=1`, `top_k=0`;
- the existing Base Prompt v2 scoring contract remains unchanged by this correction.

The previous implementation was defective because it constructed nominally larger families by cycling small template lists. In particular, several 20-case families contained only five distinct prompt texts, and the qualitative set likewise repeated five prompts four times. Renaming a repeated prompt or changing its sampling seed does not create an independent semantic benchmark case.

The corrected prompt set is identified in evaluation JSON as:

`base-prompt-v2-unique-120-2026-09-04`

The evaluator now fails closed at module construction if the full set is not exactly 120 cases, if any case ID or prompt text is duplicated, if the 100/20 scored/qualitative split changes, or if any scored family does not contain exactly 20 cases.

Regression tests independently assert the same uniqueness and family-count invariants.

## Comparability consequence

Base Prompt v2 aggregate scores produced by the recycled-template implementation must not be interpreted as statistics over 100 unique scored prompts. They are historical defective Base Prompt v2 evidence and should not be compared directly with scores from the corrected prompt set without explicitly noting the prompt-set change.

This correction does not alter or invalidate `eval_core_v1` results or L20 conditional-likelihood results, because those layers do not use the native Base Prompt v2 case constructor.

## Rationale

Evaluation v2 was approved in ADR 0140 specifically as an expanded native prompt layer with 100 mechanically scored prompts and 20 qualitative continuations. Exact repetition artificially increases nominal sample count without increasing semantic coverage, overweights a small number of questions, and makes greedy repeats fully redundant. The corrected suite restores the intended breadth while keeping the established family structure and decoding protocol.
