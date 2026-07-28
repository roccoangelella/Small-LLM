# Small LLM Project Memory

_Last updated: 2026-07-28_

## Project Goal

Build a sub-1B decoder-only language model from random initialization that:

- Speaks and writes good English.
- Understands instructions and holds coherent conversations after post-training.
- Has useful basic and intermediate reasoning capability.
- Uses modern small-LLM training techniques.
- Serves primarily as a learning and research project for an AI MSc student.

The initial scope does **not** include deliberate coding capability. Coding can be
added later as a separate extension. The initial corpus may still contain
incidental code because semantic clusters are not perfectly pure.

---

## Current Resource Assumptions

- Model size: below 1B parameters; the final parameter count is still open.
- Initial development hardware: a single NVIDIA T4, likely with microbatch size 1
  and gradient accumulation.
- Local VPS storage budget: approximately 400 GB for active cache, checkpoints,
  temporary files, and working space.
- Durable bulk storage available: a 5 TB Google Drive account.
- Unique pretraining corpus target: 90B accepted source tokens, with an 80B
  minimum and a 100B hard maximum.
- Working training target: up to 2T token presentations through repeated passes,
  but this remains an experimental assumption that must be justified by
  validation loss and downstream evaluation.
- The first pass should overlap remote reading, data preparation, local caching,
  remote shard mirroring, and training rather than waiting for the complete
  corpus to be downloaded.

The expected bottleneck on a correctly buffered single-T4 run is GPU compute.
Network latency and source decoding can still stall the GPU if the pipeline is
implemented synchronously or without sufficient prefetching.

---

## Frozen Tokenizer Decision

Use the GPT-2 byte-level BPE vocabulary already used by Nemotron-ClimbMix.

- Nemotron-ClimbMix records already contain GPT-2 token IDs.
- Production code reuses those IDs directly; it does not detokenize and
  retokenize accepted documents.
- Base vocabulary size: 50,257 IDs.
- GPT-2 `<|endoftext|>` / EOD token: 50256.
- Raw token storage uses explicit little-endian `uint16`.
- The tokenizer files, configuration, and hashes must be frozen before model
  creation.
- Any additional project special tokens must be decided before the embedding
  matrix is finalized. Their compatibility with the source token IDs must be
  tested.

Tokenizer training is outside the current project scope unless a concrete
limitation of GPT-2 tokenization is discovered later.

---

## Frozen Dataset Source and Content Policy

The initial pretraining source is:

- Dataset: `nvidia/Nemotron-ClimbMix`
- Immutable revision:
  `5eaa64b9c0c85b7f56af01d7dffdb0795816b12b`
- Source files: root `part_*.tokenized.jsonl` files only
- Semantic signal: NVIDIA's numeric `cluster_id`
- Accepted clusters: 1-10 and 12-20
- Excluded cluster: 11, NVIDIA's explicit software/programming cluster
- Validation selection: deterministic document-level hash, approximately 0.1%

There is no production detokenization, language filter, code-density filter,
quality classifier, document-level semantic classifier, or LLM approval pass.
The result must be described as **programming-cluster-excluded**, not guaranteed
code-free.

The verified topic map is:

| ID | Published broad topic |
|---:|---|
| 1 | Mathematics, Algorithms, Data Analysis |
| 2 | Books, Education, Writing, Literature, Philosophy |
| 3 | Environmental Education, History, Architecture, Engineering |
| 4 | Education, Teaching, Science, Psychology |
| 5 | International Trade, Business, Economics |
| 6 | Genetics, Biotechnology, AI, Robotics, Healthcare |
| 7 | Chemistry, Taxonomy, Agriculture, Veterinary Science |
| 8 | Gaming, Strategy, Fantasy, Virtual Reality |
| 9 | Astronomy, Cosmology, Space Exploration, Urban Planning |
| 10 | Health, Sleep, Clinical Technology, Fitness |
| 11 | Software Development, Programming, Web Development, Databases |
| 12 | Technology, Mathematics, Legal, Energy, Industrial Equipment |
| 13 | Sports, Cultural Heritage, Competition |
| 14 | Music, Instrumental Practice, Theory, Composition |
| 15 | Film, Cinema, Horror, Sci-Fi, Comics, Criticism |
| 16 | Sustainability, Climate Change, Renewable Energy |
| 17 | Cardiovascular Health, Medical Research, Immunology, Cancer |
| 18 | Technology, Cybersecurity, Social Media, Cloud Computing |
| 19 | Digital Communication, Internet Culture, Psychology |
| 20 | Public Safety, Law Enforcement, Political History, Government |

A bounded live check sampled five documents per cluster and confirmed that the
published IDs are useful broad heuristics but not perfectly pure. The evidence is
stored in `cluster_map_validation.json`.

---

## Why Source Order Cannot Be Used Directly

Nemotron-ClimbMix is locally chunky by cluster. The current smoke test also found
individual byte regions dominated by one cluster. A naïve sequential stream can
therefore expose the optimizer to very long runs of one topic before switching to
another.

The final corpus histogram alone is not enough. These two streams can have the
same total proportions but produce different optimization:

```text
mixed:      A, B, A, C, A, B, D, A, ...
sequential: all A, then all B, then all C, then all D
```

Long sequential runs create an arbitrary curriculum:

- early clusters are seen with a less-developed model and usually a higher
  learning rate;
- later clusters are seen after optimizer momentum and model representations have
  adapted to earlier material;
- recent gradients can temporarily dominate and interfere with capabilities
  learned from previously seen clusters.

The remote source may remain cluster-ordered, but the sequence presented to the
optimizer must be locally cluster-interleaved.

---

## Decided Production Architecture

Do **not** wait for a complete 90B-token download before training, and do not
build one enormous final `train.bin`.

The accepted architecture is:

```text
pinned Nemotron byte ranges
        ↓
parallel memory-bounded readers and bounded prefetch
        ↓
structural validation and cluster-11 exclusion
        ↓
deterministic train/validation split
        ↓
per-cluster training queues
        ↓
continuous token-deficit cluster scheduler
        ↓
whole documents plus EOD
        ↓
provenance-aware sequence packer
        ↓
fixed context+1 next-token sequences
        ↓
prepared sequence blocks
       ↙                         ↘
local immutable uint16 shards    bounded trainer queue
       ↓
verified Google Drive mirror
```

The same prepared in-memory sequence block is sent to both local sinks:

1. It is appended durably to the current local cache shard.
2. It is made available to the trainer without rereading it from disk.

Finalized immutable shards are uploaded asynchronously to Google Drive after
local verification. Later passes read local shards; missing future shards may be
prefetched from Drive in the background.

The implementation must not serialize the pipeline as:

```text
remote request → one document → one GPU step → one tiny filesystem write
```

Remote reads, validation, scheduling, packing, shard writing, Drive mirroring,
and training must run through bounded producer-consumer queues with backpressure.
A configurable prefetch head start is required so the GPU does not immediately
catch the downloader.

---

## Cluster Mixture Policy

The target distribution is defined in **original source tokens**, not documents,
stored cache tokens, EOD tokens, padding, or repeated presentations.

The policy remains to preserve the accepted source mixture approximately; it
does not equalize all accepted clusters and does not add hand-designed topical
weights. The exact fixed per-cluster token-weight table required by the scheduler
has not yet been frozen. It must be derived reproducibly from an approved source,
such as published corpus statistics or a deterministic calibration scan, and
saved in configuration and manifest metadata before the production run.

Mixture counters are continuous across the entire first-pass stream. They do not
reset at:

- a microbatch;
- a gradient-accumulation/global-batch boundary;
- a prepared-block boundary;
- a cache-shard boundary;
- a checkpoint;
- an interruption or resume.

Mixture behavior should be inspected over:

- optimizer/global batches formed through gradient accumulation;
- rolling windows such as 1M, 10M, and 100M emitted source tokens;
- the complete cached corpus.

The scheduler can only choose among clusters whose queues currently contain
source documents. Therefore, correct counters alone are insufficient. The
pipeline must prefetch from multiple source regions, establish useful
multi-cluster queue coverage, and schedule incrementally rather than emptying a
single available cluster queue. It may wait briefly for missing clusters when
rolling mixture error is excessive, but the policy must have bounded waits and
must not deadlock when a cluster is temporarily unavailable or exhausted.

Validation documents do not enter the training scheduler. They are routed to
separate validation cache shards using the existing deterministic split.

---

## Token-Deficit Scheduler Decision

For each accepted training cluster `c`, maintain:

```text
weight[c]          fixed target token proportion
emitted[c]         cumulative emitted original source tokens from c
total_emitted      cumulative emitted original source tokens from all clusters
deficit[c]         weight[c] * total_emitted - emitted[c]
```

Choose the available cluster with the largest deficit. Use exact integer or
rational arithmetic and a deterministic seeded tie-breaker so the same
configuration and source state produce the same stream.

The scheduler consumes the **whole next document** from the selected cluster,
even when that document overshoots the instantaneous target. It then updates the
counters by the document's original source-token count. The overshoot becomes
negative deficit and is automatically repaid by choosing other clusters later.

Example:

```text
cluster target in a local window: 20,000 tokens
already emitted:                  18,500 tokens
next whole document:               4,000 tokens
new total:                         22,500 tokens
carried debt:                       2,500 tokens
```

Do not skip or indefinitely defer a document merely because it does not fit a
local quota. That would systematically disadvantage long documents and can make
a document larger than a local quota impossible to schedule.

---

## Frozen Sequence-Packing Contract

A dataset document is not the same thing as a training batch.

For a model context length `L`, every stored training sequence contains `L + 1`
tokens:

```text
stored:  [t0, t1, ..., tL]
input:   [t0, t1, ..., t(L-1)]
target:  [t1, t2, ..., tL]
```

The sequence stride is exactly `L`, not `L + 1`. Consecutive stored sequences
therefore overlap by one token:

```text
sequence 1: [t0, ..., tL]
sequence 2: [tL, ..., t(2L)]
```

This overlap is required so that the transition from `tL` to the following token
is trained. The overlap token is physically stored twice, but it is:

- a target in the preceding sequence;
- an input in the following sequence;
- counted only once in original source-token and cluster-mixture statistics.

The packer must also:

- receive whole scheduled documents;
- append EOD 50256 only when the document does not already end with it;
- concatenate short documents;
- split long documents across fixed-length sequences;
- preserve every intended next-token transition;
- track original source tokens, inserted EODs, overlap copies, and padding
  separately;
- carry incomplete state across documents, blocks, checkpoints, and resumes;
- attribute original source tokens accurately to the blocks and shards where they
  physically occur, without assigning an entire long document to its first
  sequence and without double-counting the overlap.

The exact model context length remains open pending T4 benchmarking. A likely
development configuration is `L = 2048`, which means 2,049 stored token IDs per
sequence and a stride of 2,048.

A likely T4 batch configuration is:

```text
microbatch: one packed sequence
gradient accumulation: multiple microbatches
optimizer/global batch: accumulated token window
```

Mixture control is evaluated over accumulated and rolling token windows, not
forced inside every individual microbatch.

---

## Cache Shard Decision

The cache remains permanently sharded. There is no required final merge.

Illustrative local layout:

```text
dataset/output/
├── train/
│   ├── train-000000.bin
│   ├── train-000001.bin
│   └── ...
├── validation/
│   ├── validation-000000.bin
│   └── ...
├── progress.json
├── work_plan.json
├── mixture.json
└── manifest.json
```

Requirements:

- explicit little-endian `uint16`;
- fixed context+1 sequence geometry recorded in the manifest;
- bounded in-memory writes rather than one write per document or sequence;
- bounded parsed-source batches rather than materializing an entire 256 MiB work
  item as Python objects;
- active files use a temporary name;
- completed shards are flushed, validated, and atomically renamed;
- finalized shards are immutable;
- every finalized shard records token count, sequence count, checksum, block
  range, exact source-token provenance, mixture counters, and completion state;
- shard size is configurable; the production value is still open;
- later epochs shuffle shard order and sequence order deterministically;
- incomplete `.tmp` or `.part` files are never exposed to the trainer;
- finalized shards are independently verifiable and reusable after interruption.

Because context+1 records overlap by one token, the physical cache contains a
small amount of intentional boundary duplication. Storage estimates of roughly
160 GB for 80B, 180 GB for 90B, and 200 GB for 100B source tokens remain useful
approximations, but exact capacity planning must include overlap, EOD, padding,
metadata, temporary space, model checkpoints, and safety margin.

---

## Durable Storage Architecture

### Local VPS SSD

The local VPS filesystem is the live training cache. The trainer reads only
complete local shards or the in-memory first-pass block stream. It must not use
Google Drive as a random-access training filesystem.

The local disk contains:

- active and finalized cache shards;
- a bounded prefetch window of Drive-restored shards;
- active training checkpoints;
- temporary resumable upload/download files;
- local manifests and progress state.

### Google Drive

Google Drive is the durable remote mirror for heavy immutable training and
validation shards.

Rules:

- upload only locally finalized and checksummed shards;
- use resumable transfers;
- verify remote file identity, size, and checksum before marking a shard
  `remote_durable`;
- record stable Google Drive file IDs in the manifest rather than relying only on
  folder paths or names;
- never overwrite an immutable shard with different bytes;
- download into `.part`, verify, and atomically rename before trainer visibility;
- permit background shard downloads while the trainer consumes already-local
  shards;
- never evict a shard that is active, currently consumed, required by the latest
  checkpoint, or not yet verified remotely.

A new VPS does not need the complete 160-200 GB cache before resuming. It restores
the latest checkpoint, downloads and verifies only the next required shard plus a
small prefetch window, resumes training, and continues fetching later shards in
the background.

### Private Hugging Face Model Repository

Use one private Hugging Face model repository with three logical areas:

```text
last/    complete resumable checkpoint
best/    best evaluated model snapshot
run/     stable configs, hashes, manifests, and provenance
```

`last/` contains everything required to resume training. `best/` contains the
best evaluated model and its metrics; it does not require optimizer state unless
we explicitly decide to make it resumable. `run/` contains stable model,
tokenizer, training, dataset, environment, code-commit, schema, and approved
cluster-weight metadata.

The large dataset shards remain on Google Drive, not in the Hugging Face model
repository. Every Hugging Face `last` checkpoint includes an exact snapshot of
the Drive manifest that identifies all shards required at that training point.

---

## Fixed-Window Joint Checkpoint Protocol

Training must be entirely pausable and safely resumable, including migration from
one VPS provider to another. We do not save after every step. We save at a fixed,
configurable window measured in completed optimizer steps, trained tokens, or
wall-clock time, plus an optional graceful shutdown checkpoint.

A checkpoint is valid only at a completed optimizer-step boundary with no unknown
partial optimizer update. The checkpoint must capture one coherent logical state.

### Trainer state

- model parameters;
- optimizer state, including Adam moments;
- learning-rate scheduler state;
- FP16 gradient-scaler state;
- global optimizer step;
- trained-token and presentation-token counters;
- gradient-accumulation position, which should normally be zero at publication;
- Python RNG state;
- framework CPU RNG state;
- CUDA RNG state for every device;
- data-loader and shuffle RNG state;
- current and best validation metrics.

### Data-pipeline state

- last fully consumed block ID;
- last locally durable block ID;
- last remotely durable Drive shard/block boundary;
- work-plan and record positions;
- reconstructable per-cluster source queues;
- continuous token-deficit counters and tie-break state;
- packer carry, overlap, and provenance state;
- pending sequences in incomplete prepared blocks;
- active and finalized shard state;
- exact Drive manifest snapshot;
- configuration, source, code, tokenizer, and schema hashes.

### Publication ordering

At each fixed checkpoint window:

```text
1. Finish the current optimizer step and pause new block consumption.
2. Force a clean cache boundary or create an equally safe immutable tail snapshot.
3. Flush, fsync, checksum, and finalize every shard referenced by the checkpoint.
4. Upload newly finalized shards to Google Drive with resumable transfers.
5. Verify Drive file IDs, sizes, checksums, and manifest entries.
6. Build the complete local trainer/data checkpoint in a temporary directory.
7. Flush, fsync, checksum, and atomically finalize the local checkpoint.
8. Upload the new checkpoint to a temporary or versioned Hugging Face path.
9. Verify every remote checkpoint component and its manifest.
10. Atomically publish the new `last` pointer/commit boundary.
11. Update `best` only when the configured evaluation metric improves.
12. Resume training.
```

A Hugging Face checkpoint must never reference a Drive shard that has not already
been verified as remotely durable.

The previous valid remote checkpoint remains available until the replacement is
complete and independently verified. Only then may old `last` files or repository
history be removed. Destructive history cleanup must be explicit, disabled by
default, and refused unless the new `last` and retained `best` are verified.

After an interruption, all unsaved work after the most recent complete checkpoint
may be discarded and replayed. Resume must restore exactly the serialized logical
state and continue from the block after `last_consumed_block_id`. Already mirrored
cache blocks are replayed locally or restored from Drive rather than downloaded
again from Nemotron.

Required guarantee:

- no silent data skipping;
- no unknown duplicate training range;
- no optimizer/model state that disagrees with the data cursor;
- exact restoration of every serialized logical state component;
- safe rollback to the latest complete checkpoint;
- portable continuation on another machine.

Best-effort stronger guarantee:

- bitwise-identical future arithmetic when hardware, CUDA, dependencies, kernels,
  and deterministic settings are also identical.

Moving to different hardware or software may introduce tiny future floating-point
differences, but it must not change the restored logical training state.

---

## Migration to a New VPS

A new empty VPS restores in this order:

```text
private Hugging Face last checkpoint
→ verify checkpoint manifest and identities
→ read embedded Google Drive manifest snapshot
→ identify next required block and shard
→ download and verify minimum local prefetch window
→ restore model, optimizer, scheduler, scaler, RNG, and data state
→ continue from the next block
→ prefetch later Drive shards in the background
```

Migration must not require downloading the complete cache before the first resumed
optimizer step. The restore command should fail closed on incompatible code,
schema, tokenizer, model configuration, cluster weights, source revision, or file
hashes.

Credentials for Google Drive and Hugging Face must come from environment variables
or mounted secret files. They must never be committed to Git.

---

## Current Repository Status Versus the Production Goal

### Validated legacy foundation

The original standard-library corpus builder still provides valuable tested
components:

- pinned source-file discovery;
- deterministic byte-region work plan;
- direct HTTP byte-range access;
- exact-once JSONL ownership by absolute record-start offset;
- structural validation;
- cluster-11 exclusion;
- deterministic train/validation split;
- direct GPT-2 token reuse;
- little-endian `uint16` writing;
- EOD handling;
- per-cluster counters;
- crash-safe monolithic checkpoint/resume;
- manifest verification and a bounded live interruption/resume smoke test.

The legacy single-file `build` path remains available but must not be used for the
final 90B production run.

### Streaming-cache prototype implemented

Commit `ea4107c4d438cf4460ab15a4732d2ff935ced78e` added a schema-v1 prototype with:

- bounded parallel reader concurrency and deterministic work-plan ordering;
- exact-integer token-deficit scheduling;
- per-cluster queues;
- context+1 sequence packing;
- prepared sequence blocks;
- durable active-shard block writes;
- immutable shard finalization;
- a bounded trainer-facing queue;
- streaming configuration and weight-file validation;
- substantial offline tests.

This prototype is a useful foundation but is **not production-safe yet**.

### Known defects and missing guarantees

Before production, the streaming path must fix or complete:

1. The context+1 packer currently advances by `L + 1`; it must advance by `L` so
   one next-token transition is not lost at every sequence boundary.
2. The adapter can still drain long runs from one locally available cluster. It
   needs real multi-region prefetch, incremental scheduling, bounded waiting, and
   rolling mixture-error checks.
3. A reader currently materializes a complete work item as parsed Python objects.
   It must emit memory-bounded record/token batches with backpressure.
4. Per-block and per-shard source-token attribution is currently too coarse for
   long documents and sequence/shard boundaries. Provenance must be exact and the
   overlap counted only once.
5. Streaming resume does not yet reconstruct queued documents, pending prepared
   sequences, active-shard state, and source cursors completely.
6. Interrupted and uninterrupted streaming runs do not yet have a complete
   byte-equivalence and logical-state-equivalence proof.
7. Streaming schema v1 is not yet fully integrated with `verify` and `status`.
8. Google Drive mirroring, Hugging Face `last`/`best` publication, joint
   checkpointing, and empty-VPS migration are not implemented.
9. There is no actual model or pretraining loop yet.

No final cluster-weight table is approved. The 90B production run remains blocked
until the correctness, checkpoint, remote-storage, and migration tests pass.

---

## Evaluation Plan

Evaluation must be defined before final training and must cover:

- English fluency, grammar, and coherence;
- reading comprehension;
- general knowledge;
- instruction following after post-training;
- logical and multi-step reasoning;
- consistency and reliability;
- held-out next-token loss;
- inference speed and memory use.

Keep private evaluation data separate from all training data. Check the final
corpus for overlap with private evaluation sets where feasible.

Evaluate and compare:

1. very small pipeline/model smoke runs;
2. base pretrained model;
3. reasoning-continued model;
4. instruction-tuned model;
5. distilled or preference-optimized model;
6. final optimized model.

The `best` Hugging Face snapshot is updated only through an explicitly selected
validation metric and evaluation configuration. A better training loss alone does
not automatically redefine `best` unless that is the frozen metric.

---

## Remaining Macro Steps

1. Fix and harden the schema-v1 streaming cache implementation.
2. Implement Google Drive shard mirroring and verified restoration.
3. Implement private Hugging Face `last`, `best`, and `run` publication.
4. Prove fixed-window joint checkpoint rollback and empty-VPS migration with a
   stateful mock trainer.
5. Integrate streaming schema v1 with verification and status reporting.
6. Freeze resource budget, final evaluation suite, and cluster weights.
7. Implement and validate a very small decoder-only Transformer and trainer.
8. Benchmark T4 throughput and freeze final sub-1B architecture and context length.
9. Run base pretraining while the first pass builds and mirrors the reusable cache.
10. Evaluate the base model.
11. Continue with reasoning-focused data, distillation, and instruction tuning.
12. Evaluate every major version.
13. Optimize inference, document results, and decide whether to release.

---

## Current Open Decisions

- Final parameter count and architecture.
- Exact context length; 2,048 is the current likely development value.
- Fixed per-cluster target token weights and the reproducible method used to
  derive them while preserving the accepted source distribution.
- Parallel reader count, bounded batch size, queue sizes, prefetch head start,
  wait limits, and mixture-error thresholds.
- Cache shard size and prepared-block size.
- Fixed checkpoint interval in optimizer steps, trained tokens, or wall-clock
  time.
- Google Drive folder/run layout, authentication method, and local prefetch/LRU
  policy.
- Hugging Face repository identity, best-model metric, and destructive history
  cleanup cadence.
- Whether a separate detailed document-offset/source-provenance index is retained
  beyond the exact per-block/per-shard accounting required for correctness.
- Any special tokens beyond GPT-2 EOD 50256.
- Whether the 2T presentation target remains justified after evaluation.
- Reasoning datasets and generation process.
- Teacher model used for distillation.
- Final benchmark suite.
- Final compute availability.
- Whether the model will be released publicly.

The core storage roles, sequence stride, continuous deficit accounting, periodic
joint-checkpoint guarantee, Google Drive shard mirror, and Hugging Face
`last`/`best` division are no longer open decisions.

---

## Current High-Level Training Flow

```text
define goals, evaluation, and compute budget
→ adopt and freeze GPT-2 tokenizer compatibility
→ freeze Nemotron source revision and cluster policy
→ freeze source-proportion cluster weights
→ stream pinned byte ranges with parallel memory-bounded prefetch
→ validate records, exclude cluster 11, and split validation deterministically
→ route training documents into per-cluster queues
→ interleave whole documents with continuous token-deficit scheduling
→ append EOD and pack context+1 sequences with stride equal to context length
→ send each sequence block to local immutable cache shards and the trainer queue
→ upload and verify finalized cache shards in Google Drive
→ periodically checkpoint complete trainer and pipeline state
→ publish verified Hugging Face last and conditionally update best
→ restore only the next Drive shard window when migrating to a new VPS
→ finish the first pass with a complete reusable mirrored cache
→ use deterministically shuffled local/Drive-restored shards for later passes
→ evaluate the base model
→ continue reasoning training and instruction tuning
→ evaluate, optimize, document, and release
```
