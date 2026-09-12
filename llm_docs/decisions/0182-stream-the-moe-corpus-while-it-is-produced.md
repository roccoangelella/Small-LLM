---
status: accepted
date: 2026-09-12
supersedes: null
---

# 0182 — Stream the MoE corpus while it is produced

## Context and problem statement

The static MoE launcher waits for a complete corpus. Production is still running in
public bucket `roccoangelella/small-llm-moe-100b-superbpe-dataset`, run
`moe-100b-superbpe-b64-dataset-001`. Its contract already fixes 762,940 updates,
approximately 5B/75B/20B WSD tokens, floor 0.1, and 16 validation blocks.

With the reported initial lead of about 14.5B tokens (29 roughly 1 GB uint16 shards),
the RTX 4090 trainer at 192k tokens/s closes the gap to the approximately 155k producer
at 37k tokens/s: `14.5B / 37k ≈ 4.5 days`, having consumed approximately 75B tokens.
It then runs at the producer's pace. At 397k tokens/s an H100 catches up in approximately
0.69 days and then idles most of the time: `1 - 155/397 ≈ 61%`. These are constant-rate
estimates from the supplied snapshot; catch-up time depends on the starting lead.

## Considered options

- Wait for the complete static corpus.
- Reuse the existing incremental READY frontier and rolling cache.
- Build a separate MoE streaming implementation.

## Decision outcome

Chosen option: **reuse the existing incremental/rolling mechanism**. Both bucket/run
flags enable streaming; both empty preserve static behavior. CPU preparation uses
`stage_incremental_window_when_ready` for the checkpoint-aligned current-plus-next
window and frozen validation, without requiring completion. Consumers skip bucket
creation. `resolve_schedule` uses the contract's trainer plan, providing the equivalent
of the dense `_install_incremental_plan_adapter` without importing provider runtime code.

The stager freezes the bootstrap `manifest.json`; only `shard_frontier.json` grows.
This existing separation preserves reader and checkpoint identities across segments
and producer completion, without checkpoint format changes. Both launchers pass the
trainer flags and use existing `HF_TOKEN` secrets. Modal mounts the destination writable,
commits CPU staging, then reloads it on the GPU. CPU bootstrap waiting remains bounded
at 24 hours.

`--dataset-shard-wait-timeout-seconds` defaults to 0, preserving dense behavior. MoE
streaming sets 10,800 seconds (3 hours): `_wait_for_shard` raises `TimeoutError` after
that long without new READY train shards, including during prefetch. READY growth
resets the timer. A clean early exit reports `waiting_for_corpus` with frontier position;
only `producer_complete` makes it `incomplete`. Explicit drain remains `drained`.

## Consequences

### Positive

- Training overlaps production using the existing verified transfer and resume contracts.
- The WSD horizon stays fixed; static callers and dense default waits remain compatible.
- Stalled production no longer causes an unbounded MoE frontier wait.

### Negative or limiting

- Once caught up, GPU throughput is producer-limited and waiting may remain billable.
- Timeout fails the segment; retry resumes from the last durable checkpoint. The timer
  bounds frontier polling, not a hung network call or shard transfer.
- Modal staging must use a writable mounted data, cache, or run volume.

## Validation

CPU tests cover flags, static defaults, incomplete staging, resume beyond the bootstrap
inventory, stable identities after growth/completion, frontier status, provider wiring,
Modal commit/reload, and timeout/reset/default behavior. Run the four production suites
plus the streaming, incremental cache, and incremental frontier suites linked below.

Changes remain uncommitted as requested. Live provider training is untested; the next
operational check is an explicitly authorized bounded segment and resume after READY
growth, verifying contiguous consumption and the unchanged contract horizon.

## Links

- [Original incremental decision](0058-produce-10b-shards-concurrently-with-modal-training.md)
- [Production execution contract](../../moe_production.py)
- [Incremental staging](../../dataset/incremental_stage.py)
- [Streaming tests](../../tests/test_moe_production_streaming.py)
- [Incremental cache tests](../../tests/test_incremental_cache.py)
- [Incremental frontier tests](../../tests/test_incremental_frontier.py)
