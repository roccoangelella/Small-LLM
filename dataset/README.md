# Nemotron-ClimbMix Production Dataset Pipeline

> [!WARNING]
> **Final Production Cluster Weights Are Open and Unapproved**
> Final production cluster weights for the streaming cache path are currently **open and unapproved**. No production run may start without an approved `--weights-file` mapping signed off by the team.

---

## Overview

This directory contains the Nemotron-ClimbMix corpus pipeline with two supported formats:

1. **Schema-v2 Streaming Cache (`dataset/src/streaming.py`)**: Framework-independent, deterministic first-pass streaming cache that packs validated documents into fixed-geometry sequence blocks, fsyncs each active-shard block before trainer queue visibility, and schedules clusters with exact integer deficit accounting plus rolling-mixture backpressure.
2. **Legacy Monolithic Binary Build (`dataset.main build`)**: Appends GPT-2 token IDs directly to continuous `train.bin` and `validation.bin` files. Retained as legacy/prebuild format.

---

## Commands

### Environment Setup

```bash
uv sync --locked
```

### Stream-Cache Weight Validation (Offline Preflight)

The `stream-cache` command validates a JSON cluster weight mapping and displays sequence geometry configuration. It requires `--weights-file` (there is no production default) and does not perform network calls or start a production run:

```bash
# Validate weights file and confirm stream cache sequence geometry
uv run python -m dataset.main stream-cache \
  --weights-file path/to/approved_weights.json \
  --show-stream-config
```

### Legacy Monolithic Build Commands

Start the legacy monolithic build, resume it, check status, or verify:

```bash
uv run python -m dataset.main build
uv run python -m dataset.main build --resume
uv run python -m dataset.main status
uv run python -m dataset.main verify
```

A bounded connectivity test for the legacy build:

```bash
uv run python -m dataset.main build \
  --target-tokens 10000000 \
  --max-work-items 20 \
  --output-dir /tmp/climbmix-smoke

uv run python -m dataset.main verify \
  --output-dir /tmp/climbmix-smoke \
  --full-scan
```

---

## Schema-v2 Streaming Cache Specification (`dataset/src/streaming.py`)

### 1. Sequence Geometry: Context + 1 Format

Each record in the stream cache contains `context_length` input tokens plus 1 next-token target label (`stored_sequence_tokens = context_length + 1`).
- The record stride is exactly `context_length`: `[A,B,C,D,E]` is followed by `[E,F,G,H,I]` for context 4. The overlap is physically present in both records, but only its first physical appearance contributes to source-token and cluster accounting.
- Tokens are raw little-endian unsigned 16-bit integers (`uint16`).
- Document boundaries receive an appended `<|endoftext|>` token (ID 50256) if not already present.
- Sequence blocks contain `sequences_per_block` stored sequences, yielding a fixed block size:
  $$\text{block\_bytes} = (\text{context\_length} + 1) \times \text{sequences\_per_block} \times 2$$

### 2. Deterministic Concurrency & Integer Deficit Scheduling

- **Parallel Reader (`parallel_read_documents`)**: Active work items advance by one bounded batch per deterministic cycle. Documents are then interleaved across that cycle, and changing the worker count does not change output order.
- **Bounded batches**: `parallel_read_document_batches` has independently configurable source-token, document, and estimated-byte limits. At most one result batch per in-flight work item is retained, so a 256 MiB work item is never materialized as Python objects all at once.
- **Token Deficit Scheduler (`TokenDeficitScheduler`)**: Computes cluster selection strictly using integer arithmetic (`Fraction`) based on:
  $$\text{deficit}(c) = \text{units}[c] \times \text{total\_emitted} - \text{emitted}[c] \times \sum\text{units}$$
  Ties are broken deterministically using SHA-256 hashes of `(seed, counter, cluster_id)`.

### 3. Durability-Before-Trainer Contract

- Prepared blocks are written, flushed, and `fsync`ed in temporary active shard files (`.tmp`) before being enqueued to the trainer queue (`QueueConsumer`).
- Active shards are finalized via `fsync` and atomic rename (`os.replace`) to `train/train-XXXXXX.bin` or `validation/validation-XXXXXX.bin`.
- Backpressure: If the trainer queue reaches `--prepared-block-queue-limit`, the producer blocks until queue space is available. A checkpoint is refused until every durable trainer-visible block has also been acknowledged.

### 4. Shard Layout & Metadata

```text
dataset/output/
├── train/
│   ├── train-000000.bin
│   └── train-000001.bin
├── validation/
│   ├── validation-000000.bin
│   └── validation-000001.bin
├── manifest.json
└── progress.json
```

`manifest.json` tracks shard metadata including checksums, block ID ranges, context lengths, and per-cluster token counts.

### 5. Checkpoint, Drive, and migration contract

- `CheckpointCoordinator` accepts a framework adapter and atomically commits opaque trainer state (model/optimizer/LR/scaler/RNG/metrics) together with caller-supplied pipeline state. `StreamCacheProducer` serializes queues, scheduler, rolling counters, packers, pending prepared sequences, block counters, and finalized shard metadata. It refuses partial optimizer or gradient-accumulation windows.
- `RemoteShardStore` accepts only finalized immutable `.bin` files. The Drive backend is optional; credentials come from `GOOGLE_APPLICATION_CREDENTIALS` or an explicit mounted path, never the repository. Unit tests use `InMemoryDriveStore`.
- A Drive manifest records run ID, logical path, split/index, block range, bytes, sequence/source counts, cluster counts, SHA-256, Drive ID/checksums, verification state/time, schema hash, and configuration hash.
- `TwoPhaseCheckpointPublisher` uploads a versioned checkpoint, verifies its file manifest, then moves `run/<run-id>/latest.json`. It publishes `best.json` only after the configured metric improves. Old history is never removed automatically.
- Migration fetches `latest.json`, validates the embedded checkpoint and Drive manifests, stages the requested immutable prefetch window, verifies SHA-256, and only then installs the cache and checkpoint directories. Arithmetic may not be bitwise-identical across GPU/CUDA environments; serialized logical state must restore exactly.

The streaming library still has no durable remote source-reader cursor or executable `stream-cache build/resume` command. The CLI command above is an offline preflight, not production orchestration.

### 6. Synthetic / Offline Testing Example

You can run offline validation using synthetic test weights:

```bash
# Generate a test weights mapping for accepted clusters (1-10, 12-20)
python3 -c 'import json; print(json.dumps({str(i): 1 for i in list(range(1, 11)) + list(range(12, 21))}))' > /tmp/synth_weights.json

# Validate stream config offline
uv run python -m dataset.main stream-cache --weights-file /tmp/synth_weights.json --show-stream-config
```

Run the unit test suite:

```bash
uv run python -m unittest discover -v
```

---

## Frozen Production Policy

- Source: `nvidia/Nemotron-ClimbMix`
- Revision: `5eaa64b9c0c85b7f56af01d7dffdb0795816b12b`
- Files: root `part_*.tokenized.jsonl` only; `climbmix_small` and every subdirectory are excluded
- Target: 90,000,000,000 accepted source tokens
- Minimum acceptable completed size: 80,000,000,000
- Hard maximum: 100,000,000,000
- Accepted clusters: 1–10 and 12–20
- Excluded cluster: 11, NVIDIA's software/programming cluster
- Seed: `small-llm-climbmix-production-v1`
- Validation probability: 0.001 per document
- Tokenizer: the source GPT-2 token IDs, reused as-is
- End-of-document token: GPT-2 `<|endoftext|>`, ID 50256

---

## License and Limitations

Nemotron-ClimbMix is published by NVIDIA under CC BY-NC 4.0. Keep NVIDIA's attribution and the generated manifest with any derived corpus. The repository and dataset card are at <https://huggingface.co/datasets/nvidia/Nemotron-ClimbMix>.
