---
status: reference
last_reviewed: 2026-08-13
---

# 100M / 10B incremental dataset and Modal contract

This is the current technical contract created by ADR 0058. It describes dataset production/consumption readiness; it does **not** close ADR 0050's separate behavioral authorization gate for H100 training.

## Frozen run contract

```text
training run ID: 100m-10b-data-001
dataset profile: modal-10b-b64
dataset run ID: modal-10b-b64-dataset-001
context: 2,048
prepared optimizer block: 64 sequences
nominal budget: 10,000,000,000 target tokens
whole-block horizon: 76,294 updates
exact consumed prefix: 10,000,007,168 targets
train shard target: approximately 1 GiB
live validation prefix: first 16 deterministic validation blocks
WSD: 3,815 warmup / 57,220 stable / 15,259 decay updates
WSD targets: 500,039,680 / 7,499,939,840 / 2,000,027,648
minimum LR ratio: 0.1
training topology: one Modal H100
```

The 7,168-token excess above nominal 10B is whole-block rounding. The trainer stops at the frozen horizon even if final packing produces an unused tail.

## Incremental producer and durability

The pinned `nvidia/Nemotron-ClimbMix` source is range-read on demand. A cheap CPU Modal producer applies the frozen project validation, cluster-11 exclusion, deterministic split, exact mixture scheduling, and context+1 packing. Each finalized train shard is:

```text
created locally
→ uploaded to the private HF dataset Storage Bucket
→ independently read back and SHA-256 verified
→ producer cursor/progress atomically persisted
→ Modal cache Volume committed
→ published READY in a monotonic frontier
→ optionally evicted locally
```

READY must never get ahead of durable producer progress. Existing bytes that are not bound to committed producer state are not trainer-visible data.

The first validation prefix covering 16 validation blocks is frozen once published. Later validation production does not change live-run evaluation identity.

## CPU-before-H100 barrier

Before an H100 training function is spawned, CPU staging must resolve the checkpoint-aligned next block and require:

- the current READY train shard;
- one READY successor when a successor exists;
- the frozen validation prefix;
- complete SHA-256 verification of the staged lead window;
- a committed shared cache Volume.

Readiness may wait on the producer. Identity, monotonicity, or checksum failures fail closed. The purpose is explicit: do not allocate an H100 and then leave it idle downloading or waiting for its first usable shard.

## Online H100 consumption

The trainer consumes READY shards in exact block order. While shard N is being trained, the rolling cache prefetches N+1 and CPU production may advance farther ahead. A consumed shard is evicted only after its final block is successfully acknowledged. If the producer frontier falls behind, training waits; it never skips, substitutes, or reorders blocks.

Resume re-enters through the same checkpoint-aligned CPU staging barrier.

## Completion and reuse

At terminal production, the ordinary immutable schema-v2 manifest closes the READY frontier with its SHA-256. That terminal manifest is the canonical finished-corpus record for reproducibility and later reuse. A later run/epoch can stream the already-produced immutable HF shards again; no Google Drive mirror or second corpus copy is required.

## Ownership

- `dataset/`: source reading, frozen run contract, producer cursor/durability, READY frontier, validation freezing, rolling cache, integrity checks, and dynamic shard resolution.
- `modal/`: Modal CPU function/container orchestration, Volume commits, checkpoint-aligned staging, and CPU-to-H100 dispatch.
- `tests/`: regression coverage.

## Scientific launch gate

ADR 0050 requires material behavioral/capability improvement from the completed 100M/2B model before the fresh 100M/10B **training run** is launched. The intrinsic 100M/2B scaling result is strong, but the exact ADR-0025 global-32-token qualitative qualification is still outstanding as of 2026-08-13. CPU dataset pipeline qualification can continue independently.

See ADRs 0050, 0057, and 0058 and [`../runbooks/100m_10b_incremental_modal.md`](../runbooks/100m_10b_incremental_modal.md).
