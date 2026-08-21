---
status: accepted
date: 2026-08-21
supersedes: null
---

# ADR 0113 — Use Superior Reasoning Stage 2 for R-SFT scaling

## Context and problem statement

ADR 0104 deliberately restricted the first large R-SFT experiment to Superior Reasoning Stage 1 `instruction_following`, and the later expansion exhausted that Stage-1 lane under the accepted R0 policy. The resulting canonical reasoning corpus contains 16,716 rows, but its verified one-pass native bundle reaches only about 0.67% of the 100M/2B pretraining target count. ADR 0112 therefore selected nested 1% / 2% / 4% unique-data scaling experiments.

The Superior Reasoning repository also exposes a much larger Stage 2. That availability was not previously called out in project memory because the first-large-R-SFT decisions were intentionally Stage-1-only. This omission made the available scaling headroom unnecessarily opaque.

## Decision outcome

Use **Superior Reasoning Stage 2 `instruction_following` as the next expansion source** for the ADR-0112 1% / 2% / 4% R-SFT scaling sweep.

The existing 16,716-row Stage-1-expanded corpus remains frozen and forms the base of every larger nested corpus. Stage 2 is additive; it does not rewrite or reinterpret the completed Stage-1 artifact.

Stage-2 examples must pass the same accepted R0 processing contract used for Stage 1:

- strict `<think>...</think>` teacher-output parsing with a non-empty final answer;
- normalized-prompt deduplication, including deduplication against every prompt already present in the frozen Stage-1-expanded corpus;
- `instruction_following` only for this scaling lane;
- the same conservative exclusion of primary mathematical computation/proof and primary programming/code tasks while allowing incidental numbers and ordinary formatting constraints;
- rejection of reserved `<think>`, `</think>`, and `<answer>` marker collisions;
- exact atomic 2,048-token R-SFT serialization validation with no truncation;
- unchanged context-fit Stage-2 examples are retained directly;
- over-context Stage-2 examples are frozen as adaptation candidates and may enter a scaling corpus only after the same explicit curation gate used by the completed Stage-1 expansion;
- only curated `keep` candidates may receive ADR-0103 Variant-D fidelity-first compression;
- every rewrite must pass the exact 2,048-token validator and normalized-prompt deduplication before inclusion;
- teacher traffic remains Gemini-only under the ADR-0106 backend/fallback constraints; no alternate teacher provider is introduced.

The Stage-2 scaling implementation is `post_training/R-SFT/dataset/scale_superior_reasoning.py`. The historical Stage-1 producer remains unchanged so the already-frozen Stage-1 artifact stays directly reproducible.

## Budget semantics

The 1% / 2% / 4% labels refer to **loss-bearing training target tokens relative to the verified 100M/2B parent count of 2,001,000,448 targets**, not raw text tokens, serialized context tokens, bytes, or row counts.

The top-level training mixture remains 90% reasoning / 10% S0 instruction retention. Corpus assembly therefore targets the reasoning side of each requested total budget and mirrors the production builder's deterministic 1% validation + 1% test partition per reasoning group before measuring train targets. Retention remains sampled later from the completed S0 tokenized instruction records.

The corpora must remain nested and one-pass:

```text
16,716-row Stage-1-expanded base ⊂ 1% ⊂ 2% ⊂ 4%
```

Repeated epochs must not be used to manufacture a larger nominal budget.

## First execution

Build the 1% corpus first. Prefer unchanged, already-context-fit Stage-2 instruction rows. If those rows are sufficient to reach the exact projected 1% train-token budget, do not spend Gemini quota merely to enlarge the candidate pool. If they are insufficient, curate the frozen Stage-2 over-context candidates and resume only the curated keepers through the existing Variant-D keeper-only adaptation path until the 1% target can be reached.

## Consequences

- Superior Reasoning remains the primary reasoning-data source while materially increasing unique-data headroom.
- The 1% experiment remains directly interpretable as a scale change rather than a simultaneous policy change.
- Stage-1 reproduction remains immutable.
- Stage-2 over-context examples inherit the same quality/safety and Gemini-only constraints rather than bypassing them for volume.
- Project memory now explicitly records that both Superior Reasoning stages exist and that Stage 2 is the selected scaling source.
