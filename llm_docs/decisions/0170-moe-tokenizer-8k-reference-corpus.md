---
status: current
date: null
# Decision date was not recorded; last_reviewed is retained below.
last_reviewed: 2026-09-10
supersedes: null
---

# ADR 0170 — MoE tokenizer 8k target and reference corpus

## Context and problem statement

The MoE line is moving away from the inherited GPT-2 50,257-token tokenizer. The project already has a production-scale reference corpus in the Hugging Face Storage Bucket `roccoangelella/small-llm-100b-datasets`; it is currently being uploaded in GPT-2-tokenized binary form and is not yet complete. The corpus was produced from the project's accepted clustered/weighted dataset pipeline.

## Considered options

No alternative was recorded at the time; section added for the template.

## Decision outcome

- Target an **8k-class semantic vocabulary** for the new MoE tokenizer.
- Use the current `roccoangelella/small-llm-100b-datasets` corpus as the tokenizer R&D/testbed so tokenizer design reflects the same data distribution intended for model pretraining.
- Reconstruct text from the stored GPT-2 token IDs before training candidate tokenizers. Detokenization is a transport/reconstruction step only; it must not introduce cleaning or normalization that changes the corpus.
- The incomplete Bucket may be used for prototyping and comparative tokenizer experiments. It is not by itself sufficient grounds to freeze the final tokenizer unless a separately accepted stability criterion demonstrates that additional corpus upload no longer changes the relevant tokenizer metrics/vocabulary materially.

## Important data-layout constraint

The packed binary shards preserve the intended cluster mixture through the producer's deficit scheduler and record aggregate per-cluster statistics in metadata, but they do not retain per-token/per-document cluster labels in the final payload. Exact per-cluster tokenizer diagnostics therefore require either returning to the pinned source records or producing an explicit sidecar/diagnostic sample with cluster attribution.

The packed sequences also use context+1 storage with one-token overlap between adjacent sequences. A text reconstruction path must undo that physical overlap before decoding rather than independently decoding every stored sequence and concatenating the results.

## Open decisions

- Whether `8k` means exactly 8,000 or 8,192 semantic IDs, and whether special tokens count inside that budget.
- Exact special-token inventory.
- Tokenization algorithm, normalization policy, pre-tokenization rules, byte fallback/open-vocabulary guarantee, and cross-word merging policy.
- Size and deterministic construction of the tokenizer-training sample.
- Intrinsic and proxy-language-model acceptance metrics and stability thresholds.

## Consequences

- No model/tokenizer code is changed by this ADR.
- Current MoE-branch model configuration still uses the inherited GPT-2 vocabulary until the tokenizer design is selected and separately approved for implementation.
