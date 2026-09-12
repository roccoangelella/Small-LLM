---
status: accepted
date: 2026-09-11
supersedes: null
---

# 0176 — Retokenize the 100B MoE corpus before token-count stratification

## Context and problem statement

The MoE line uses the frozen 8,000-ID SuperBPE tokenizer from ADR 0175, while the pinned Nemotron-ClimbMix source is distributed as GPT-2-tokenized JSONL records. The existing production dataset pipeline already provides the desired deterministic cluster filtering/stratification, train/validation split, schema-v2 context+1 packing, immutable sharding, crash-safe resume, incremental READY frontier, and Hugging Face durability/upload behavior.

Those components should be reused rather than reimplemented. However, the existing scheduler and corpus-size accounting use `SourceDocument.source_token_count`, so leaving documents in GPT-2 token space until after scheduling would measure cluster deficits and the 100B corpus budget in the wrong token unit.

## Considered options

No alternative was recorded at the time; section added for the template.

## Decision outcome

For the production MoE corpus, retokenization happens at the document boundary **before** a `SourceDocument` is handed to the existing scheduler.

For each accepted ClimbMix record:

1. Preserve the original numeric `cluster_id`, stable source identity, work-item position, and deterministic train/validation split.
2. Read the original GPT-2 token IDs.
3. Remove a terminal GPT-2 `<|endoftext|>` ID `50256` when present.
4. Reconstruct the document bytes with canonical GPT-2 decoding and require strict valid UTF-8.
5. Encode the reconstructed text with the frozen `tokenizer/superbpe_8000.json` artifact.
6. Construct `SourceDocument` from the resulting SuperBPE source-text IDs.
7. Feed that document into the unchanged deficit scheduler, rolling-mixture checks, packing, sharding, resume, READY-frontier, and Hugging Face publication pipeline.

Consequently, cluster weights, rolling mixture windows, queue accounting, stopping limits, and the nominal 100B corpus target are measured directly in **SuperBPE source tokens**, not inferred from the previous GPT-2 counts.

## Token contract

- Semantic vocabulary: IDs `0..7999` (8,000 classes).
- Ordinary pretraining source text may emit only BPE IDs `0..7991`.
- SuperBPE `<|endoftext|>` is ID `7992` and is inserted by the existing packer as the document boundary marker.
- Reserved/control IDs `7993..7999` are not reachable from ordinary pretraining text. Literal strings such as `<think>` inside source documents remain ordinary text and must not become control-token IDs.
- GPT-2 EOD `50256` must never appear in final SuperBPE shards.
- IDs `8000+` are invalid corpus tokens even though the model stores embeddings padded to 8192 rows.

## Dataset identity and resume safety

The SuperBPE corpus identity includes the source tokenizer identity, output tokenizer identity, frozen tokenizer artifact path/hash, semantic vocabulary size, and EOD identity. These fields participate in the SuperBPE production configuration/schema hashes, so a run cannot be resumed with a different tokenizer contract.

The historical GPT-2 producer identity is deliberately left unchanged so existing GPT-2 producer state remains reproducible/resumable.

The producer fails closed if the checked-in `tokenizer/superbpe_8000.json` no longer matches the frozen artifact identity expected by this MoE branch.

## Frozen 100B production profile

Dedicated run identity: `moe-100b-superbpe-b64-dataset-001`.

- target SuperBPE source tokens: 100,000,000,000
- minimum: 90,000,000,000
- hard maximum: 110,000,000,000
- context length: 2,048
- sequences per block: 64
- target shard size: 1 GiB
- durable source-token checkpoint interval: 500,000,000
- output transport: existing incremental schema-v2 READY frontier plus Hugging Face Storage Bucket durability

A dedicated Modal CPU producer entrypoint is provided for this corpus. It performs no MoE training; it only executes GPT-2 decode -> SuperBPE encode -> stratify -> pack -> shard -> HF publication, with resume through the existing production state machinery.

## Implementation status

Implemented on `moe-8e-top1` through code head `c3863a0c4ba644320c6d28fe8f259e0243b5c87f` before this ADR commit:

- `dataset/superbpe_retokenization.py`: document-boundary GPT-2 -> SuperBPE conversion and frozen tokenizer contract.
- `dataset/production/cli.py` and `dataset/production/policy.py`: tokenizer-aware production identity while preserving the legacy GPT-2 identity.
- `dataset/src/verify.py`: semantic-vocabulary-aware shard verification.
- `dataset/moe_100b.py`: frozen 100B SuperBPE production profile.
- `modal/moe_100b_dataset.py`: resumable CPU producer using the existing HF incremental pipeline.
- `tests/test_superbpe_100b_pipeline.py`: regression contracts for pre-scheduler retokenization, reserved-token safety, EOD replacement, tokenizer-dependent hashes, semantic-vocabulary limits, and the frozen profile.

The targeted tests are present but have not yet been executed in a qualified project environment. The assistant sandbox could not perform a clean checkout because external GitHub resolution was unavailable, and no GitHub Actions workflow/status checks are attached to the branch. No 100B corpus production has been launched yet.

## Consequences

### Branch ownership

This decision and its implementation belong to the MoE development line and are recorded in `llm_docs/` on `moe-8e-top1`. They should not be maintained as project-memory state on `main` unless/until the MoE line is intentionally merged there.
