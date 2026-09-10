---
status: accepted
date: 2026-09-10
supersedes: null
---

# 0171 — Stream ClimbMix detokenization for tokenizer R&D

## Context

The MoE tokenizer work needs natural-language text derived from the same ClimbMix source used by the project's pretraining corpus. Reconstructing text from the project's packed 100B `.bin` shards would require undoing context+1 packing, one-token sequence overlap, shard boundaries, and inserted EOD markers even though those transformations occur only after the original ClimbMix document records are read.

The canonical source is the pinned `nvidia/Nemotron-ClimbMix` revision already frozen by the project. Its `part_*.tokenized.jsonl` records retain document boundaries, GPT-2 token IDs, `cluster_id`, and stable source identity before project-specific packing.

## Decision

For tokenizer R&D, bypass the project's packed 100B HF bucket and read the pinned original ClimbMix tokenized JSONL records directly.

The initial tokenizer-training corpus target is **10 GB of detokenized raw text**. Documents are selected through the same accepted-source policy and cluster-aware stratification semantics used by the project corpus, then decoded from their original GPT-2 IDs on the fly. Each selected document must satisfy an exact GPT-2 round trip before use:

`original GPT-2 IDs -> lossless decode -> raw text -> GPT-2 encode == original GPT-2 IDs`.

Do **not** persist a complete 10 GB raw-text copy merely as an intermediate artifact. Feed verified decoded documents through a streaming/corpus interface suitable for tokenizer training, while preserving enough deterministic metadata (source identity, cluster identity, pinned revision, selection/sample definition) to reproduce the same 10 GB sample.

The tokenizer itself is trained globally over the selected 10 GB corpus; it is not incrementally frozen batch-by-batch in source traversal order.

For eventual MoE pretraining, do not repeatedly perform GPT-2 detokenization and new-tokenizer encoding at every model run or epoch. Once the new tokenizer is frozen, run a separate large-scale one-time corpus conversion:

`pinned ClimbMix GPT-2 records -> verified decode -> frozen new tokenizer -> immutable new-tokenizer training shards`.

Those newly tokenized shards become the reusable training artifact for the MoE trajectory. There is no requirement to persist the transient natural-language text between decode and re-encode.

## Consequences

- Tokenizer R&D avoids all inverse handling of 2049/2048 packing, optimizer blocks, shard boundaries, and inserted project EOD markers.
- Per-document `cluster_id` and source identity remain available for deterministic stratification and diagnostics.
- Tokenizer-corpus generation can begin independently of completion of the project's currently uploading 100B packed bucket.
- Raw text remains transient, reducing unnecessary storage and eliminating a large duplicate corpus artifact.
- Final MoE training consumes a corpus tokenized directly with the frozen project tokenizer rather than paying GPT-2 decode/new-tokenizer encode cost during training.

## Open decisions

- Exact deterministic construction of the 10 GB sample and its held-out tokenizer evaluation split.
- Whether the 10 GB limit is enforced on UTF-8 raw bytes, another explicit raw-text size measure, or an equivalent reproducible unit; raw UTF-8 bytes are the current intended interpretation.
- How cluster mixture accounting should be defined after changing tokenizers: preserve selection/mix according to original source/GPT-2 token counts versus rebalance according to new-token counts. This must be decided before the final MoE corpus conversion.
- Final tokenizer family, pretokenizer, vocabulary size convention, special-token inventory, and freeze criteria remain governed by the tokenizer decision process and are not settled here.

## Links

- `0170-moe-tokenizer-8k-reference-corpus.md`
- `../../dataset/config.py`
- `../../dataset/src/records.py`
- `../../dataset/src/streaming.py`
