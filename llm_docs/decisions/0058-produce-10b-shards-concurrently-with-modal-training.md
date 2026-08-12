---
status: accepted
date: 2026-08-12
supersedes: null
---

# 0058 — Produce 10B shards concurrently with Modal training

## Context

The rolling 10B dataset implementation already avoids materializing the complete approximately-20-GB derived corpus in Modal and keeps only a bounded current-plus-next training-shard cache. However, the consumer still assumed a completed schema-v2 manifest before H100 training could begin. That stopped one step short of the intended architecture: the user wants derived shards to be produced directly from the pinned ClimbMix source while earlier ready shards are already being consumed by Modal.

## Decision

Adopt an incremental producer-consumer dataset contract for the 100M / 10B trajectory.

- The pinned ClimbMix source remains canonical input and is range-read on demand; there is no preliminary full-source download.
- The dataset producer creates deterministic approximately-1-GiB derived shards in order, verifies each finalized shard, publishes it immediately to the Hugging Face dataset bucket, and may evict the local copy after verified remote durability.
- A small immutable run contract freezes the dataset identity, source revision/policy, tokenizer/context, block64 geometry, approximately-10B target horizon policy, and standard 5%/75%/20% WSD policy before H100 allocation.
- A mutable, monotonic shard frontier records the contiguous set of remotely durable READY shards and the producer cursor. READY entries are immutable once published.
- Modal training no longer requires the terminal complete manifest before update 1. It may consume only READY shards from the frontier and must preserve exact block order.
- Before H100 allocation, cheap CPU Modal staging requires a checkpoint-aligned current shard plus at least one verified successor shard when a successor is expected. This establishes a two-shard lead buffer rather than allocating the H100 after only one shard exists.
- While the H100 trains shard N, the producer continues creating later shards and the rolling cache downloads READY shard N+1 in parallel. If the producer frontier ever fails to stay ahead, training blocks rather than skipping or reordering data.
- Once the producer reaches the target and final packing boundary, it publishes the normal immutable completed schema-v2 manifest. That terminal manifest remains the canonical finished-corpus record for reproducibility and future epochs/runs.
- Hugging Face remains the durable per-shard source of truth and enables later re-use or additional epochs without rebuilding already completed shards.
- Dataset semantics belong under `dataset/`; Modal-specific CPU gating, Volume commit, and CPU-to-H100 dispatch remain under `modal/`; regression tests remain under `tests/`.

## Consequences

The complete derived 10B corpus does not have to exist before H100 training starts, and no machine has to hold the complete corpus locally. Dataset production and GPU consumption overlap while preserving durable shard identity, deterministic ordering, checkpoint-aligned resume, and a final immutable corpus manifest.

This decision does not waive ADR 0050's behavioral/capability gate for launching the fresh 100M / 10B experiment.
