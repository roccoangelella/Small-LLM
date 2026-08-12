---
status: current
last_reviewed: 2026-08-12
---

# 100M / 10B incremental dataset and Modal launch state

The 100M / 10B data path is now an incremental producer/consumer pipeline under ADR 0058 rather than a completed-corpus prerequisite.

## Frozen training contract

```text
training run ID: 100m-10b-data-001
dataset profile: modal-10b-b64
dataset run ID: modal-10b-b64-dataset-001
context: 2,048
prepared optimizer block: 64 sequences
nominal training budget: 10,000,000,000 target tokens
exact whole-block horizon: 76,294 updates
exact target prefix: 10,000,007,168 target tokens
shard target: 1 GiB
live validation prefix: 16 deterministic validation blocks
WSD updates: 3,815 warmup / 57,220 stable / 15,259 decay
WSD tokens: 500,039,680 / 7,499,939,840 / 2,000,027,648
minimum LR ratio: 0.1
execution topology: single Modal H100
```

## Incremental data flow

The canonical source remains the pinned `nvidia/Nemotron-ClimbMix` revision. A cheap CPU Modal producer range-reads the source, applies the frozen project validation/exclusion/mixture/packing logic, creates approximately-one-GiB schema-v2 shards, uploads each finalized shard to the private HF dataset Storage Bucket, performs complete read-back SHA-256 verification, commits the corresponding producer cursor to the Modal cache Volume, publishes the shard READY in a monotonic frontier, and then may evict the producer-local bytes.

The complete derived 10B corpus does not need to exist before H100 training starts.

Before any H100 function is spawned, an independent CPU staging function resolves the checkpoint-aligned next block and waits for:

- its current READY train shard;
- one READY successor when a successor is expected;
- the frozen 16-block validation prefix.

It downloads and SHA-256 verifies that lead window and commits the cache Volume. Only then does the launcher dispatch the H100.

During H100 training, the local dynamic reader consumes the immutable prefix in exact block order, asynchronously prefetches the next READY shard, evicts a train shard only after its final block is acknowledged, and waits rather than skips/reorders if the producer frontier ever falls behind.

The trainer-facing bootstrap manifest is immutable even if the producer finishes between training sessions; mutable READY/completion state lives in `shard_frontier.json`. The terminal full schema-v2 manifest remains the canonical completed-corpus record and closes the frontier with its SHA-256.

## Durability ordering

The required producer ordering is:

```text
upload shard
-> independent remote read-back verification
-> atomically write producer progress
-> Modal Volume commit of that progress
-> publish shard READY
-> local eviction
```

This prevents trainer-visible READY data from getting ahead of the durable source cursor after a producer-container failure.

## Remaining scientific launch gate

ADR 0050 still requires the completed 100M / 2B behavioral/capability qualification to show material improvement before the fresh 100M / 10B H100 trajectory is actually launched. Dataset CPU production/staging architecture can be qualified independently of that scientific authorization.
