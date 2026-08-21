---
status: accepted
date: 2026-08-21
supersedes: null
---

# ADR 0112 — Scale R-SFT with nested 1% / 2% / 4% Superior-Reasoning corpora

## Context and problem statement

The completed expanded R-SFT corpus contains 16,716 unique normalized reasoning prompts and yields a verified one-pass native bundle with 13,420,823 loss-bearing training targets. Against the completed 100M/2B parent horizon of 2,001,000,448 training targets, this is about 0.67% of pretraining exposure.

The project now wants to measure whether additional unique reasoning supervision continues to improve the approximately-100M model before considering substantially larger R-SFT budgets such as 10% of pretraining.

## Decision outcome

Run a nested unique-data R-SFT scaling sweep at approximately **1%, 2%, and 4% of the 100M/2B pretraining target count**.

Using the verified parent count of 2,001,000,448 targets, the nominal total R-SFT budgets are:

- 1%: approximately 20,010,004 loss-bearing targets;
- 2%: approximately 40,020,009 loss-bearing targets;
- 4%: approximately 80,040,018 loss-bearing targets.

Keep the already-accepted top-level R-SFT mixture at **90% reasoning / 10% completed-S0 instruction retention**, measured in loss-bearing target tokens.

Keep **Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b** as the primary external reasoning-data source. The completed 16,716-row corpus remains the lower-scale seed/baseline and should be retained when constructing larger nested corpora. The 1% corpus must be a subset of the 2% corpus, and the 2% corpus a subset of the 4% corpus, apart from deterministic held-out partitions required for evaluation.

Do not reach the larger budgets by replaying the same examples for extra epochs. The scaling variable is unique supervised data volume; production comparison runs remain one pass over each frozen bundle.

The exact policy for admitting additional Superior-Reasoning stages/domains beyond the already exhausted Stage-1 clean instruction-following pool is intentionally left open for a separate implementation choice. Existing logic-first exclusions of primary math/computation and primary programming/code remain in force unless explicitly superseded.

## Rationale

A 1% → 2% → 4% sweep gives a direct learning-curve measurement while bounding the risk of over-specialization and catastrophic overwrite in the small 100M model. Nested corpora make the comparison interpretable because each larger point adds data rather than changing the entire sample identity. Keeping the Superior-Reasoning family as the primary source preserves teacher/data provenance across the sweep.

## Consequences

- The dataset-production pipeline needs a token-budget-aware nested corpus assembler rather than a row-count-only target.
- The current Stage-1 instruction-following source pool alone cannot supply the full sweep under the existing filter, so the source-expansion policy must be extended within the Superior-Reasoning dataset before 2%/4% production is possible.
- The existing atomic marker contract, 2,048-token fit requirement, deterministic deduplication, 90/10 S0-retention builder, 32,768-target optimizer blocks, one-pass training, and current qualification stack remain applicable.
- A future 10%-of-pretraining R-SFT experiment is not authorized by this ADR; it should depend on the observed 2%→4% scaling curve.
