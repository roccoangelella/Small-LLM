---
status: accepted
date: 2026-09-10
supersedes: null
---

# 0174 — Defer tokenizer intra-cluster sampling strategy

## Context and problem statement

### Rationale

The intra-cluster sampling method controls corpus dispersion and possible source-order bias, but several defensible deterministic designs remain. The project will reason about those alternatives separately rather than prematurely binding the tokenizer corpus to one method.
## Considered options

No alternative was recorded at the time; section added for the template.

## Decision outcome

The exact strategy used to sample documents *within* the selected ClimbMix clusters for the 10 GB tokenizer-development corpus remains deliberately **undecided**.

Do not yet freeze any of the following candidate mechanisms:

- reuse of the existing deterministically shuffled ClimbMix work plan;
- document-level hash subsampling;
- source-region size or region-level dispersion policy;
- any particular combination of region randomization and document randomization.

These alternatives will be evaluated and selected later before the tokenizer corpus is materialized.

## Already accepted surrounding constraints

This deferral does not reopen the decisions already made for Gate A:

- target approximately 10,000,000,000 bytes of effective UTF-8 text for tokenizer training;
- persist the detokenized corpus to disk as sharded JSONL rather than treating raw text as ephemeral;
- use only the ClimbMix training side for tokenizer-training data;
- perform a short cleanliness preflight before choosing the source clusters;
- target four selected clusters with an equal high-level balance: two scientific/technical and two discursive, approximately 25% of effective text bytes each;
- remove terminal GPT-2 EOD structurally before text decoding and later insert the new tokenizer's EOD deterministically when producing pretraining shards;
- require lossless GPT-2 decode/re-encode verification;
- defer the final MoE pretraining-corpus stratification policy until after the tokenizer is frozen.

## Consequences

No consequences were recorded at the time; section added for the template.
