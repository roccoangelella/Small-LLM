---
status: accepted
date: 2026-08-12
---

# Stream the 10B corpus through 1 GiB HF shards and CPU-stage before H100 allocation

## Context

ADR 0050 authorizes a fresh approximately-100M / 10B pretraining experiment only after the completed 100M / 2B behavioral qualification confirms that the larger model's lower loss translates into meaningful capability. Dataset preparation itself does not require H100 compute and can proceed before that behavioral gate resolves.

A compact uint16 10B-token derived corpus is roughly 20 GB. Repeating the 2B migration pattern — materialize the whole derived corpus, transfer the whole directory into a Modal workspace, then train — would make dataset preparation unnecessarily workspace-specific and would scale poorly across disposable compute workspaces.

The existing dataset system already has the needed scientific primitives: the pinned Hugging Face ClimbMix source is read deterministically with HTTP byte-range requests; project validation, accepted-cluster policy, exact mixture scheduling and context+1 packing run as records arrive; prepared optimizer blocks are immutable schema-v2 units; and finalized shards already carry SHA-256 identity.

The operational requirement is also explicit: an H100 must not be allocated just to sit idle while the first large training shard downloads. Dataset transport must be staged on cheap CPU compute first.

## Decision

The 10B dataset and its Modal consumer use the following contract.

### Source and construction

- Keep the existing pinned ClimbMix Hugging Face revision as source truth.
- Do not mirror or download the whole ClimbMix release before construction.
- Read deterministic HTTP ranges from the pinned source and apply the existing project validation, cluster exclusion, train/validation split, exact mixture scheduler and context+1 packer on the fly.
- Keep context length 2,048 and the prepared optimizer block at 64 sequences. Storage sharding must not alter optimizer/update/checkpoint geometry or token ordering.

### Derived shard geometry

The canonical 10B profile is `modal-10b-b64`, dataset run ID `modal-10b-b64-dataset-001`.

Use a target shard size of exactly one GiB:

```text
target_shard_bytes = 1,073,741,824 = 1024^3
stored bytes / full optimizer block = 64 * 2,049 * 2 = 262,272
full optimizer blocks / 1 GiB target = 4,094
bytes in 4,094 full blocks = 1,073,741,568
unused gap below target = 256 bytes
target training tokens / full shard = 4,094 * 64 * 2,048 = 536,608,768
```

Thus a full shard contains about 0.537B target training tokens. A roughly-10B train stream is therefore on the order of nineteen full-size train shards plus a final partial shard, with validation stored separately.

### Canonical remote storage and bounded dataset production

- A private Hugging Face Storage Bucket is the canonical store for the complete derived dataset.
- Finalized immutable shards are uploaded independently under the dataset run namespace.
- A shard becomes remotely durable only after the uploaded object is independently downloaded/read back and its byte size and SHA-256 match the local finalized bytes.
- Dataset production writes its durable progress checkpoint before deleting any verified local finalized shard.
- After that progress commit, remotely verified finalized shards may be evicted locally. The builder therefore needs bounded local disk rather than enough disk for the entire 10B corpus.
- Producer resume may reconstruct metadata for missing historical local shards only through the explicit remote-eviction path, and only when the saved durability manifest exactly matches the producer state and the remote immutable-object inventory still contains those objects. Ordinary local-cache resume remains unchanged and fail-closed.
- Final dataset readiness is published only after the complete manifest and remote shard inventory are consistent.

### Modal CPU-before-H100 allocation barrier

For the `100M / 10B` Modal profile, dataset transport is `hf_rolling_shards` rather than a fully materialized dataset Volume.

Before any H100 function is spawned, a CPU-only Modal function must:

1. inspect the newest durable local/HF training checkpoint;
2. determine the exact next unconsumed optimizer block;
3. download the train shard containing that block into the shared cache Volume;
4. download the reusable validation shard set when training is not already complete;
5. verify the staged bytes against the immutable manifest SHA-256 values;
6. commit the cache Volume;
7. authorize H100 dispatch only after all previous steps succeed.

For a fresh run the required block is zero, so this stages `train-000000.bin`. On resume it stages the shard containing the checkpoint-aligned next block rather than assuming shard zero. If the CPU gate determines that all planned training blocks are already consumed, it must return completion without allocating an H100.

The rolling H100 function has no automatic Modal function retry. A fresh attempt after failure must return through the CPU staging gate so checkpoint and shard state cannot silently drift between attempts.

### H100 rolling cache

During real online training:

- keep the current train shard local;
- asynchronously prefetch one successor shard while the H100 trains on the current shard;
- expose a downloaded shard only after complete local size/SHA-256 verification;
- delete a consumed train shard only after its final prepared block has been successfully acknowledged;
- retain validation shards locally because they are reused at every evaluation boundary.

The default look-ahead is one shard, giving approximately current + next, or about 2 GiB of steady-state train-data storage rather than approximately 20 GB.

The short microbatch-qualification subprocess must use only the already CPU-staged current shard. It must not start a one-GiB successor download merely to run a four-step probe. Rolling successor prefetch begins in the actual online training subprocess.

### Ownership boundaries

- Provider-neutral source reading, remote shard storage, production durability, rolling-cache selection/prefetch/eviction and integrity checks belong under `dataset/`.
- `modal/` contains only Modal-specific CPU staging, Volume commit and CPU-to-H100 dispatch orchestration.
- Regression tests live under `tests/`.

## Relationship to the scientific experiment

This ADR changes dataset storage/transport and provider orchestration only. It does **not** change:

- model geometry;
- tokenizer or semantic vocabulary;
- accepted/excluded data policy;
- deterministic mixture/packing order;
- optimizer batch geometry;
- optimizer math;
- FP16/FP32-master precision policy;
- the ADR-0050 requirement that 100M / 2B behavioral qualification gates actual 10B H100 launch.

This storage decision also does not independently settle or supersede the exact 10B WSD horizon policy. Any later schedule change requires its own scientific decision.

## Alternatives considered

### Materialize the complete approximately-20-GB corpus in every Modal workspace

Rejected for 10B. It couples a durable scientific dataset to disposable compute workspaces and forces large repeated transfers before useful GPU work can start.

### Retain the 32 MiB shard target used for the 2B reblock migration

Rejected for 10B. It is technically workable but creates hundreds of objects and far more transport/checksum/transition events than necessary. One-GiB shards align almost exactly with 4,094 complete block-64 optimizer units and still give ample time for one-shard-ahead prefetch.

### Read training blocks directly from remote object storage without a local shard cache

Rejected as the default production path. It would put H100 utilization directly on remote object-store latency and range-read behavior. A small verified local rolling cache keeps GPU reads predictable while preserving bounded storage.

## Consequences

- No machine must hold the whole ClimbMix source or the complete derived 10B corpus locally.
- The derived dataset is portable across Modal workspaces because Hugging Face, not a Modal Volume, is canonical storage.
- Fresh and resumed H100 allocation is gated on a checkpoint-aligned shard already being present and verified.
- Training storage stays bounded to approximately two one-GiB train shards plus validation and small metadata.
- Dataset production can start while 100M / 2B finishes, without authorizing the 10B H100 run before ADR 0050's behavioral gate passes.
