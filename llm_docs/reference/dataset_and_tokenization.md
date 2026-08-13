# Dataset and tokenization

_Last reviewed: 2026-08-13_

## Tokenizer contract

The project consumes GPT-2 byte-level BPE IDs already embedded in the pinned Nemotron-ClimbMix records.

```text
tokenizer: gpt2
semantic vocabulary: 50,257
EOD: <|endoftext|> / 50256
storage dtype: explicit little-endian uint16
internal padded embedding rows: 50,304
```

IDs 50,257-50,303 are implementation padding only. They are never valid dataset targets or sampled semantic outputs. Accepted source records are not detokenized and retokenized.

## Pinned source and content policy

```text
source: nvidia/Nemotron-ClimbMix
revision: 5eaa64b9c0c85b7f56af01d7dffdb0795816b12b
included source files: root part_*.tokenized.jsonl
retained clusters: 1-10, 12-20
excluded cluster: 11 (explicit software/programming cluster)
validation: deterministic document-level identity hash
```

Describe the resulting corpus as **programming-cluster-excluded**, not guaranteed code-free. There is no production detokenization, language filter, code-density filter, quality classifier, or LLM approval pass.

## Exact mixture contract

The scheduler uses the measured source-token totals of the pinned release, conditioned on cluster 11 being excluded. The approved retained-cluster weight artifact has SHA-256:

```text
76e82e22760adcac59c7294fe9bac11358f5a8b7a26035aae64c3f2e6fa1acb7
```

The completed calibration scanned all 100 pinned source files and measured:

```text
source bytes:                 1,987,970,304,099
records:                      553,315,056
all-cluster source tokens:    356,864,528,972
accepted source tokens:       351,792,454,745
excluded cluster-11 tokens:     5,072,074,227
accepted documents:           544,684,421
```

Mixture accounting is continuous across documents, blocks, shards, checkpoints, and resumes. Whole documents plus EOD are scheduled with exact integer deficit accounting; a long document is not split merely to satisfy a local mixture quota.

## Sequence-packing contract

For context length `L=2048`, each stored sequence has `L+1=2049` IDs:

```text
stored: [t0 ... t2048]
input:  [t0 ... t2047]
target: [t1 ... t2048]
```

Stride is 2,048, so adjacent stored sequences duplicate one physical overlap token while preserving every intended next-token transition. Padding/provenance and deterministic block identifiers are recorded in schema-v2 metadata.

## Immutable shard contract

Finalized shards are immutable and independently verifiable. Active temporary files are never trainer-visible. Manifests bind schema, geometry, block ranges, counts, byte sizes, SHA-256 hashes, source identity, tokenizer identity, and mixture identity. Trainer cursor advancement is legal only across durable verified block boundaries.

Historical finite 20M/Kaggle datasets remain valid and readable under the schema that produced them.

## Current remote durability

ADR 0054 retires Google Drive for **new** dataset production. Trusted new production uses Hugging Face Storage Buckets and verifies remote byte size and SHA-256 by independent read-back before advancing durable producer state or evicting a finalized local shard.

Historical compatibility names remain part of readable schema/checkpoint identity:

```text
drive_manifest.json
drive_file_id
drive_checksums
```

For new HF-backed data these are provider-neutral legacy names only. They do not imply a Google API backend, OAuth flow, or second permanent mirror.

## Finite Kaggle datasets

Completed finite datasets such as the 20M/2B corpus are prebuilt, privately published to Kaggle, round-trip verified, attached read-only to the notebook, and consumed in exact prepared-block order. Kaggle GPU training does not live-stream the 2-TB source corpus.

## Incremental 100M/10B dataset

ADR 0058 removes the requirement that the whole 10B derived corpus exist before training. A CPU producer range-reads the pinned source and publishes approximately-1-GiB immutable HF shards behind a monotonic READY frontier. CPU staging verifies the checkpoint-aligned current+successor window and frozen 16-block validation prefix before an H100 can be allocated. Online training preserves exact block order and waits rather than skips if production falls behind.

Detailed contract: [`100m_10b_incremental_dataset.md`](100m_10b_incremental_dataset.md).

## Ownership

Dataset source reading, mixture scheduling, packing, manifests, remote durability, READY frontier, and rolling-cache logic belong under `dataset/`. Platform-specific orchestration belongs under `kaggle/` or `modal/`; it must not redefine dataset semantics.
