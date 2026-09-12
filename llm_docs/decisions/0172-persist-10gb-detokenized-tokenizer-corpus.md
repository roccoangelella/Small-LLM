---
status: accepted
date: 2026-09-10
supersedes: null
---

# 0172 — Persist the initial 10 GB detokenized tokenizer corpus

## Context and problem statement

ADR 0171 selected direct reconstruction from the pinned `nvidia/Nemotron-ClimbMix` GPT-2-tokenized source for tokenizer R&D and an initial approximately-10-GB raw-text sample. ADR 0171 proposed keeping reconstructed text transient.

## Considered options

No alternative was recorded at the time; section added for the template.

## Decision outcome

For the initial tokenizer corpus, materialize and retain approximately **10 GB of detokenized natural-language text on disk** instead of feeding reconstructed text only transiently to the tokenizer trainer. The persisted text is a reusable intermediate artifact and may be deleted later if it is no longer useful.

The source remains the pinned official ClimbMix tokenized JSONL records, before Small-LLM context+1 packing. Each selected document must be reconstructed losslessly from GPT-2 token IDs and retain enough sidecar provenance to reproduce/audit the sample (at minimum stable source identity and cluster identity). The tokenizer-training corpus should not pass through the existing packed 100B `.bin` shards, so no 2049/2048 overlap inversion is needed.

The exact cluster stratification/mixing policy for the later full MoE corpus conversion is **not decided here**. It will be revisited after the tokenizer architecture and vocabulary are frozen.

## Consequences

- Tokenizer candidates can be retrained and compared repeatedly without re-reading and detokenizing ClimbMix every time.
- Roughly 10 GB plus provenance/metadata storage is required locally for the initial corpus.
- The persisted text must be byte-faithful; no normalization, cleanup, replacement decoding, or other text transformation is authorized by this decision.
- The later production conversion remains a separate operation: ClimbMix GPT-2 IDs -> reconstructed text/bytes -> frozen Small-LLM tokenizer -> immutable model-training shards.

## Open details

Before implementation, define the exact byte-count stopping rule, output container/framing, metadata/provenance representation, handling of source GPT-2 EOT tokens if encountered inside source records, train/dev/held-out partitioning for tokenizer evaluation, deterministic sampling rule, and integrity/round-trip checks.

## Links

- `0170-moe-tokenizer-8k-reference-corpus.md`
- `0171-moe-tokenizer-10gb-direct-climbmix-source.md`
