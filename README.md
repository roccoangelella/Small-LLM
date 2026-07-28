# Small-LLM

This repository is working toward a small English language model. The pretraining-corpus data pipeline is in `dataset/`.

> [!WARNING]
> **Production Cluster Weights Are Open and Unapproved**
> Final production cluster weights for the streaming cache are currently **open and unapproved**. No production run may start without an approved `--weights-file` JSON weight mapping signed off by the team.

---

## Overview & Architecture

The dataset pipeline provides two formats:

1. **Schema-v1 Deterministic Streaming Cache (`dataset/src/streaming.py`)**:
   A framework-independent, first-pass streaming cache that turns validated source documents into fixed-geometry sequence blocks, fsyncs each active-shard block before exposing it to the trainer queue, and atomically finalizes immutable shards at legal boundaries.
2. **Legacy Monolithic Binary Build (`dataset.main build`)**:
   The original prebuild format that streams byte ranges and appends GPT-2 token IDs directly to continuous `train.bin` and `validation.bin` files. Retained as legacy/prebuild format.

Both paths process NVIDIA Nemotron-ClimbMix at the pinned immutable revision `5eaa64b9c0c85b7f56af01d7dffdb0795816b12b`, keeping accepted clusters 1–10 and 12–20 while strictly excluding cluster 11 (programming/software).

---

## Command Line Interface

Validate a JSON weight mapping and confirm sequence geometry without starting a network run:

```bash
# Stream-cache weight mapping validation (offline/preflight check)
uv run python -m dataset.main stream-cache --weights-file path/to/approved_weights.json --show-stream-config
```

Legacy monolithic prebuild commands:

```bash
uv run python -m dataset.main build
uv run python -m dataset.main build --resume
uv run python -m dataset.main status
uv run python -m dataset.main verify
```

---

## Streaming Cache Architecture & Contracts

- **Context+1 Sequence Geometry**: Every sequence in the cache stores `context_length` input tokens plus 1 target token (`stored_sequence_tokens = context_length + 1`, e.g. 2049 tokens for context length 2048), encoded as raw little-endian uint16 integers.
- **Deterministic Concurrency**: `parallel_read_documents` reads HTTP source ranges concurrently while reordering futures to yield records in exact work-plan order. `TokenDeficitScheduler` uses exact integer arithmetic (`Fraction`) and deterministic SHA-256 tie-breaking.
- **Durability-Before-Trainer Contract**: Sequence blocks are written, flushed, and fsynced in active shard files (`.tmp`) before becoming visible to the trainer queue. Completed shards are separately finalized with `fsync` and atomic rename.
- **Shard Layout**: Output files are written to `train/` and `validation/` subdirectories (`train-XXXXXX.bin`), with full metadata written to `manifest.json`.
- **Resume & Replay Limits**: Checkpoints record `last_durable_block_id`. Replay restores state strictly from the last durable block. **No GPU checkpoint atomicity is claimed**; joint state synchronization remains the responsibility of the trainer loop.

---

## Testing & Verification

Run the local offline unit test suite:

```bash
uv run python -m unittest discover -v
```

See [dataset/README.md](dataset/README.md) for full specifications, checkpoint contracts, and bounded smoke-test execution options.
