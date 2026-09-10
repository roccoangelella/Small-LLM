---
status: accepted
date: 2026-09-10
supersedes: null
---

# 0173 — Freeze tokenizer-corpus Gate A format, EOD handling, and cluster policy

## Context

ADR 0171 selected direct reconstruction from the pinned official `nvidia/Nemotron-ClimbMix` tokenized source rather than inverting the project's packed 100B `.bin` corpus. ADR 0172 then changed the tokenizer-R&D path so the reconstructed text is persisted locally rather than treated as purely transient.

The remaining Gate A choices are how much text to materialize, how to persist it, how to handle GPT-2 EOD tokens, and how to constrain the source mixture used to learn the tokenizer.

## Decision

The initial tokenizer-training corpus will contain **10,000,000,000 bytes of effective UTF-8 document text**, excluding JSON framing and metadata. Whole documents are atomic: the producer may exceed the target slightly to finish the final accepted document but never truncates one to hit the byte target exactly.

The corpus will be persisted as **uncompressed sharded JSONL**, not as one monolithic JSON value. Each row is one reconstructed document and preserves at least `source_id`, `cluster_id`, `gpt2_token_count`, and `text`. Sharding is chosen for streaming reads, bounded failure/restart scope, checksumability, and avoiding an all-or-nothing 10 GB JSON parse. The exact shard-size target remains an implementation detail to freeze before wiring; approximately 0.5–1 GB of effective text per shard is the intended range.

No redundant copy of the full GPT-2 token-ID list is stored in the tokenizer corpus. The immutable pinned ClimbMix source remains the authority for those IDs.

GPT-2 reconstruction is lossless. For each accepted source document:

1. validate the source token IDs structurally;
2. if token ID `50256` occurs exactly once at the terminal position, remove it before text decoding and record that fact as provenance if useful;
3. if `50256` occurs at any interior position, fail/quarantine that record rather than silently reinterpret it;
4. decode the remaining GPT-2 IDs to bytes/text without normalization or lossy replacement;
5. re-encode the reconstructed text with GPT-2 and require exact equality with the source IDs after terminal-EOD removal.

The stored natural-text document therefore contains no literal GPT-2 `<|endoftext|>` marker. Document boundaries are represented by JSONL record boundaries. Once the new tokenizer is frozen, its own EOD token will be inserted deterministically during the separate large-scale retokenization/packing pipeline.

`tiktoken` package version is not duplicated as a semantic corpus-identity field in the manifest. The repository dependency lock/environment remains responsible for implementation reproducibility; the pinned ClimbMix revision and exact round-trip test define the data identity.

Only the deterministic training side of the existing source split is eligible for the 10 GB tokenizer-training corpus. No separate tokenizer held-out corpus is frozen by this ADR; validation/diagnostic data can be defined when tokenizer candidates are evaluated.

The tokenizer corpus will intentionally use a **small balanced subset of ClimbMix topical clusters rather than the complete production mixture**. The target design is four clusters: two scientific/technical and two discursive/prose-oriented, contributing equal effective UTF-8 bytes (25% each). Exact cluster IDs are not yet frozen. They must be selected after a bounded cleanliness audit among plausible ClimbMix clusters, preferring coherent, high-quality natural-language content and avoiding obviously noisy/web-heavy categories. This tokenizer-corpus balance is separate from, and does not freeze, the later MoE pretraining mixture after the tokenizer is finalized.

## Consequences

- Gate A no longer depends on the project's 2049/2048 packed-shard geometry.
- The 10 GB reconstructed corpus can be reused across BPE/Unigram/MinGram and pretokenizer experiments without repeating GPT-2 detokenization.
- Sharded JSONL preserves document and cluster provenance while remaining easy to stream.
- GPT-2 EOD cannot leak into learned natural-language vocabulary as the literal `<|endoftext|>` string.
- Equal four-cluster byte quotas deliberately control tokenizer-learning frequencies, but the exact clusters must be chosen empirically before corpus production.
- Final MoE corpus stratification remains intentionally unresolved until after tokenizer freeze.

## Next Gate-A decision

Run a bounded candidate-cluster cleanliness audit and freeze the exact two scientific/technical plus two discursive/prose cluster IDs. Then freeze the shard-size/checkpoint geometry and wire the corpus builder only after implementation understanding is demonstrated.
