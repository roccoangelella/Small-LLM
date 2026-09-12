---
status: accepted
date: null
# Decision date was not recorded; last_reviewed is retained below.
last_reviewed: 2026-08-15
supersedes: null
---

# ADR 0084 — Require consistent, precisely worded deductive teacher problems

## Context and problem statement

### Rationale

Recent test batches were substantially better after tightening the DED family definition, but exposed two residual failure modes: accidental premise inconsistency and ambiguous logical wording. These should be prevented at generation time with a small prompt-level guardrail, while deeper correctness checking remains the responsibility of the verifier/rejection pipeline rather than an increasingly prescriptive teacher prompt.
## Considered options

No alternative was recorded at the time; section added for the template.

## Decision outcome

The Gemini deductive R-SFT generation prompt must explicitly require that all stated premises are mutually consistent unless the problem is intentionally about detecting a contradiction.

Gemini must not resolve an accidental inconsistency by silently ignoring one of the premises. When contradiction detection is intended, the question and answer should make that task explicit.

The prompt must also ask Gemini to use precise wording for quantities, thresholds, exclusivity, and necessary/sufficient conditions so that the intended logical interpretation is unambiguous. Examples include preferring `at least three` over ambiguous `three` when the lower-bound meaning is intended, and using `exactly one of` when exclusivity matters.

This refinement supplements the existing DED prompt contract: open-ended question forms are preferred when natural, answers should state the actual conclusion, problems remain self-contained, and Gemini retains freedom over the natural reasoning depth.

## Consequences

No consequences were recorded at the time; section added for the template.
