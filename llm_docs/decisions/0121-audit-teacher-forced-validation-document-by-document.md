---
status: accepted
decided_at: 2026-08-24
---

# ADR 0121 — Audit teacher-forced validation document by document

## Context

Manual inspection of the new teacher-forced held-out confidence report exposed cases where a model can be heavily penalized for disagreeing with uncommon but potentially valid source wording. Examples include `parg coat` in a masonry article and `ballast substances` in flour-milling copy. Distinguishing source-text quality from tokenizer/packing corruption is therefore necessary before interpreting individual low-probability or high-confidence-error examples as model failures.

## Decision

Review the teacher-forced validation evidence **document by document and example by example**, keeping each source document as the unit of manual context rather than loading the whole validation corpus at once.

For each flagged example:

1. recover enough of the underlying source document to establish the real text and document boundary;
2. verify whether the suspicious wording exists in the source rather than being introduced by tokenization, packing, decoding, or display rendering;
3. classify the example as clean/valid, unusual-but-valid, source-quality noise, scrape/formatting contamination, or pipeline corruption;
4. separate model plausibility from exact-ground-truth correctness when interpreting the teacher-forced probability/rank;
5. keep a cumulative audit summary across documents without carrying full document text forward unnecessarily.

## Consequences

- Individual teacher-forced outliers will not be treated automatically as language-model failures.
- Source-quality weaknesses can be quantified separately from model quality.
- Any evidence of actual tokenizer/packing corruption becomes a dataset-pipeline issue and should be escalated separately.
- Manual review stays bounded in context by processing one source document at a time.
