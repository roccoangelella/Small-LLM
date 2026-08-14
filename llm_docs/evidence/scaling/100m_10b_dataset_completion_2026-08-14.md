---
status: evidence
observed_at: 2026-08-14
run_id: modal-10b-b64-dataset-001
---

# 100M / 10B dataset completion and remote inventory

The VPS-fed incremental producer resumed at 2026-08-13 18:43 UTC and completed
at 2026-08-14 04:44 UTC, about ten hours later. Local durable progress reports:

```text
complete: true
training_horizon_reached: true
accepted source tokens: 10,000,000,560
planned training blocks: 76,294
produced training blocks: 76,333
produced tail beyond horizon: 39 blocks
validation blocks: 77
```

The canonical trainer consumes only the frozen 76,294-block prefix, equal to
10,000,007,168 target tokens. The extra 39 packed blocks are the allowed final
packing tail and do not change the training horizon.

The final schema-v2 manifest contains 21 train shards and 21 validation shards:

```text
train bytes:      20,019,820,068
validation bytes:    19,998,240
total bytes:      20,039,818,308
```

Authenticated HF inspection of
`roccoangelella/small-llm-100m-qualification-datasets` found all 42 binary
objects plus `run_contract.json`, `shard_frontier.json`, `manifest.json`, and
`ready.json`. The terminal READY/frontier state reports:

```text
target_reached: true
producer_complete: true
ready train shards: 21
last ready train block: 76,332
manifest SHA-256: d23e7e4641e30c25b56189093bf1270cd11e85efc8b26bc4660af1873edb96f1
```

The Beam `small-llm-cache` Volume independently listed the same 21 train and 21
validation filenames with matching byte sizes. That listing proves inventory
and size visibility, not a second full remote read-back hash. The VPS mirror
hook verified local hashes before each copy, and the training preseed guard
performs the required mounted-volume SHA-256 verification before consumption.

The producer PID had exited. Its remaining lock pathname is expected: the lock
uses `flock`, and no process retained the OS lock after completion.

