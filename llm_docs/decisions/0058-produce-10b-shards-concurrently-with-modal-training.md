---
status: accepted
date: 2026-08-12
supersedes: null
---

# 0058 — Produce 10B shards concurrently with Modal training

## Context

The rolling 10B dataset implementation already avoids materializing the complete approximately-20-GB derived corpus in Modal and keeps only a bounded current-plus-next training-shard cache. However, the first implementation still assumed a completed schema-v2 manifest before H100 training could begin. That stopped one step short of the intended architecture: derived shards should be produced directly from the pinned ClimbMix source while earlier READY shards are already being consumed by Modal.

True producer/consumer overlap also means the training horizon and validation set cannot depend on a terminal manifest that does not exist at update 1. Those scientific contracts therefore have to be frozen before H100 allocation.

## Decision

Adopt an incremental producer-consumer dataset contract for the 100M / 10B trajectory.

### Immutable prelaunch run contract

Before H100 allocation, freeze:

- pinned ClimbMix source revision and deterministic work-plan identity;
- the approved ClimbMix cluster weights and exclusion policy;
- GPT-2 token IDs, context length 2,048, and context+1 storage;
- block64 optimizer geometry;
- one-GiB target shard size;
- nominal training horizon 10,000,000,000 target tokens;
- exact whole-block training prefix of **76,294 optimizer blocks = 10,000,007,168 target tokens**;
- standard WSD policy from ADR 0057 over that exact prefix: **3,815 warmup updates / 57,220 stable / 15,259 decay**, corresponding to **500,039,680 / 7,499,939,840 / 2,000,027,648 target tokens**;
- minimum learning-rate ratio 0.1;
- a frozen training-validation prefix of **16 validation blocks**.

The 7,168-target-token excess above nominal 10B is solely whole-block rounding. The trainer stops after block 76,293 even if final producer packing creates a small additional train tail. That tail is recorded in the terminal corpus manifest but is not consumed by this trajectory.

ADR 0057 remains the schedule-policy decision (5%/75%/20%). This ADR clarifies that, for the concurrent producer path, the exact schedule boundaries are derived from the immutable prelaunch whole-block horizon rather than waiting for the terminal packed manifest.

### Incremental production and READY frontier

- The pinned ClimbMix source remains canonical input and is range-read on demand; there is no preliminary full-source download.
- The dataset producer creates deterministic approximately-1-GiB derived shards in order, verifies each finalized shard, publishes it immediately to the Hugging Face dataset bucket, and may evict the local copy after verified remote durability.
- The producer continues until **both** the accepted-source target is reached and all 76,294 frozen training blocks are durable. Hitting the source hard maximum before the frozen train horizon is available is a fail-closed error.
- A mutable, monotonic shard frontier records the contiguous set of remotely durable READY shards. READY entries are immutable once published.
- Crash-consistency order is mandatory: **upload and independently verify remote bytes → commit the producer source cursor/progress → publish the shard as READY → evict the local copy**. A crash may leave the READY frontier behind committed producer progress, never ahead of it.
- Remote bytes that exist but are not referenced by the committed producer cursor are not trainer-visible READY data.
- The first deterministic validation prefix covering 16 validation blocks becomes immutable once published. Later validation production does not change the validation set used by the live 100M / 10B run.

### CPU-before-H100 allocation barrier

Modal training no longer requires the terminal complete manifest before update 1.

Before any H100 function is spawned, cheap CPU Modal functions must:

1. start or resume the deterministic incremental producer;
2. resolve the checkpoint-aligned next training block;
3. wait until the run contract and READY frontier exist;
4. require the current train shard plus a verified successor shard whenever a successor is expected;
5. require the frozen validation prefix;
6. download and SHA-256-verify that lead window into the shared cache;
7. commit the cache Volume;
8. authorize H100 dispatch only after all previous checks pass.

The CPU stage may wait for not-yet-published producer metadata or READY shards. Integrity, identity, monotonicity, and checksum failures do not retry as ordinary readiness waits; they fail closed. Thus an H100 never sits idle merely because shard 0/1 or producer bootstrap metadata are still being created.

### Online H100 consumption

- The trainer consumes only READY shards and preserves exact block order.
- While the H100 trains shard N, the rolling cache downloads READY shard N+1 asynchronously and the CPU producer may continue creating later shards.
- The local dynamic frontier is cached so HF control-plane polling occurs around frontier/shard boundaries rather than once per optimizer update.
- A consumed shard is deleted only after its final block is successfully acknowledged.
- If production ever falls behind the consumer, the trainer waits for the required block rather than skipping, substituting, or reordering data.
- Checkpoint resume uses the saved last-consumed block and re-enters through the same CPU staging barrier.

### Terminal corpus record and reuse

Once production reaches the accepted-source target and final packing boundary, it publishes the normal immutable completed schema-v2 manifest and marks the frontier complete with that manifest's SHA-256. The terminal manifest remains the canonical finished-corpus record for reproducibility and future epochs/runs.

Hugging Face remains the durable per-shard source of truth. A later second epoch or another training trajectory can stream the already-produced immutable shards again without rebuilding them or introducing Google Drive.

### Ownership boundaries

- Dataset source reading, run-contract semantics, producer durability, READY-frontier publication, validation freezing, rolling cache, integrity checks, and dynamic shard resolution belong under `dataset/`.
- `modal/` owns only Modal-specific CPU function/container orchestration, Volume commit, checkpoint-aligned staging, and CPU-to-H100 dispatch.
- Regression tests live under `tests/`.

## Consequences

The complete derived 10B corpus does not have to exist before H100 training starts, and no machine has to hold the complete corpus locally. The first useful GPU update can begin once the CPU producer has established a verified two-shard lead plus frozen validation, while later dataset production overlaps training.

This design adds a mutable control-plane frontier, so monotonicity and crash ordering become part of correctness. The final immutable manifest is still required before the dataset can be called a completed reusable corpus.

This decision does not waive ADR 0050's behavioral/capability gate for launching the fresh 100M / 10B experiment.
