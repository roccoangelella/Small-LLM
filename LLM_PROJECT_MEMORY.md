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
- Dataset storage budget: approximately 400 GB.
- Unique pretraining corpus target: 90B accepted source tokens, with an 80B
  minimum and a 100B hard maximum.
- Working training target: up to 2T token presentations through repeated passes,
  but this remains an experimental assumption that must be justified by
  validation loss and downstream evaluation.
- The first pass should overlap remote reading, data preparation, local caching,
  and training rather than waiting for the complete corpus to be downloaded.

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
pinned remote byte ranges
        ↓
parallel readers and bounded prefetch
        ↓
structural validation and cluster-11 exclusion
        ↓
deterministic train/validation split
        ↓
per-cluster training queues
        ↓
token-deficit cluster scheduler
        ↓
whole documents plus EOD
        ↓
document/packing buffer
        ↓
fixed-length next-token sequences
        ↓
prepared sequence blocks
       ↙                         ↘
immutable uint16 cache shards    bounded trainer queue
```

The same prepared in-memory sequence block is sent to both sinks:

1. It is appended to the current local cache shard.
2. It is made available to the trainer without rereading it from disk.

This overlaps the first training pass with dataset acquisition. Later passes read
only the local cache.

The implementation must not serialize the pipeline as:

```text
remote request → one document → one GPU step → one tiny filesystem write
```

Remote reads, validation, scheduling, packing, shard writing, and training must
run through bounded producer-consumer queues with backpressure. A configurable
prefetch head start should be available so the GPU does not immediately catch
the downloader.

---

## Cluster Mixture Policy

The target distribution is defined in **tokens**, not documents.

The policy remains to preserve the accepted source mixture approximately; it
does not equalize all accepted clusters and does not add hand-designed topical
weights. The exact fixed per-cluster token-weight table required by the scheduler
has not yet been frozen. It must be derived reproducibly from an approved source,
such as published corpus statistics or a deterministic calibration scan, and
saved in configuration/manifest metadata before the production run.

Rigid quotas should not reset for every microbatch. With a T4, a microbatch may
contain only one fixed-length sequence, so it cannot meaningfully represent all
clusters. Mixture accounting is continuous and should be inspected over:

- an optimizer/global batch formed through gradient accumulation;
- rolling windows such as 1M, 10M, and 100M emitted source tokens;
- the complete cached corpus.

EOD tokens, padding, and any later repeated presentations do not change the
source-mixture counters. Cluster accounting uses original accepted source tokens.

---

## Token-Deficit Scheduler Decision

For each accepted training cluster `c`, maintain:

```text
weight[c]          fixed target token proportion
emitted[c]         cumulative emitted source tokens from c
total_emitted      cumulative emitted source tokens from all clusters
deficit[c]         weight[c] * total_emitted - emitted[c]
```

Choose the cluster with the largest deficit. Use a deterministic randomized
tie-breaker so the same configuration and source produce the same stream.

The scheduler consumes the **whole next document** from the selected cluster,
even when that document overshoots the instantaneous target. It then updates the
counters by the document's source-token count. The overshoot becomes negative
deficit and is automatically repaid by choosing other clusters later.

Example:

```text
cluster target in a local window: 20,000 tokens
already emitted:                  18,500 tokens
next whole document:               4,000 tokens
new total:                         22,500 tokens
carried debt:                       2,500 tokens
```

Do not skip or indefinitely defer the document merely because it does not fit a
batch quota. That would systematically disadvantage long documents and can make
a document larger than a cluster's local quota impossible to schedule.

Scheduler counters are continuous across microbatches, optimizer steps, cache
shard boundaries, checkpoints, interruptions, and resumes. They reset only when
starting a deliberately new corpus-generation run.

Validation documents do not enter the training scheduler. They are routed to
separate validation cache shards using the existing deterministic split.

---

## Sequence Packing and Batching

A dataset document is not the same thing as a training batch.

The packer:

- receives whole scheduled documents;
- appends EOD 50256 only when it is not already present;
- concatenates short documents when appropriate;
- splits long documents across fixed-length training sequences;
- preserves every accepted token exactly once in the first-pass cache;
- emits sequence-aligned blocks suitable for next-token prediction.

The exact context length is still open. A likely T4 development configuration is:

```text
microbatch: one packed sequence
gradient accumulation: multiple microbatches
optimizer/global batch: accumulated token window
```

Mixture control is evaluated over the accumulated and rolling token windows, not
forced inside every single microbatch.

---

## Cache Shard Decision

The cache remains permanently sharded. There is no required final merge.

Illustrative layout:

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
- fixed sequence geometry recorded in the manifest;
- bounded in-memory writes rather than one write per document or sequence;
- active files use a temporary name;
- completed shards are flushed, validated, and atomically renamed;
- each finalized shard records token count, sequence count, checksum, source
  provenance, mixture counters, and completion state;
- shard size is configurable; the production value is still open;
- later epochs shuffle shard order and sequence order deterministically;
- finalized shards are immutable and reusable after interruption.

The current estimate remains approximately 160 GB for 80B raw tokens, 180 GB for
90B, and 200 GB for 100B, before metadata, EOD overhead, temporary space, model
checkpoints, and safety margin.

---

## Crash, Resume, and Exactness Requirements

The existing byte-range builder already has strong exact-once source ownership,
durable binary checkpoints, and interruption/resume tests. Preserve those
properties.

The combined cache-and-train design introduces a new consistency boundary:
pipeline progress and model-training progress must agree on which prepared
sequence blocks have been consumed.

The production implementation must use stable block IDs and checkpoint:

- remote work-item/record positions;
- per-cluster source queues or their reproducible reconstruction state;
- deficit scheduler counters;
- packer carry buffer;
- active/finalized shard positions;
- sequence blocks made durable;
- sequence blocks acknowledged by the trainer;
- the matching model/optimizer/scheduler checkpoint step.

After a crash, the system may replay only the explicitly permitted uncommitted
tail from the last joint checkpoint. It must not silently skip data or train
twice on an unknown range. The exact joint-commit protocol remains an
implementation task and needs dedicated tests before the production run.

---

## Current Repository Status Versus the New Goal

### Already implemented and tested

The repository currently contains a working, standard-library-only corpus
builder in `dataset/` that:

- resolves the pinned root source files;
- divides them into deterministic byte regions;
- hash-shuffles a complete work plan;
- reads direct HTTP ranges;
- owns JSONL records exactly once by absolute record-start offset;
- performs structural validation;
- excludes cluster 11;
- creates a deterministic train/validation split;
- writes existing GPT-2 IDs as little-endian `uint16`;
- appends EOD when needed;
- maintains per-cluster counters;
- checkpoints and resumes safely;
- verifies sizes, hashes, schemas, and token ranges;
- passed a bounded live interruption/resume smoke test.

These modules are valuable foundations and should be reused rather than discarded.

### Current implementation that no longer matches the production decision

The current production path:

- processes one work item at a time;
- appends directly to one growing `train.bin` and one `validation.bin`;
- has no parallel source-reader pool or bounded producer-consumer queues;
- has no per-cluster document queues;
- has no fixed target mixture-weight table;
- has no token-deficit scheduler;
- relies on shuffled byte regions to approximate the final mixture but does not
  guarantee local optimizer batches are representative;
- has no fixed-length sequence packer;
- has no rotating immutable cache shards;
- has no trainer queue or first-pass training integration;
- has no joint data/model checkpoint protocol;
- contains no actual model or pretraining loop yet.

Therefore, the current 90B single-file build should not be started as the final
production run. Treat it as a validated source-access, validation, and
crash-safety foundation to refactor into the decided streaming cache-and-train
pipeline.

The smoke test observed only about 0.4-0.5 MiB/s for cold small random ranges.
That result is not a production benchmark, but the new implementation must expose
source throughput, queue depth, packer throughput, cache throughput, and trainer
input-wait time so bottlenecks can be measured rather than guessed.

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

---

## Remaining Macro Steps

1. Freeze resource budget and final evaluation suite.
2. Refactor the current corpus builder into the sharded, cluster-interleaved,
   streaming cache-and-train producer.
3. Validate source throughput, queue behavior, mixture accuracy, packing, and
   joint crash/resume on bounded live runs.
4. Implement and validate a very small decoder-only Transformer and trainer.
5. Freeze final sub-1B architecture and context length.
6. Run base pretraining while the first pass builds the reusable local cache.
7. Evaluate the base model.
8. Continue with reasoning-focused data and distillation.
9. Instruction-tune the model as an assistant.
10. Evaluate every major version.
11. Optimize inference, document results, and decide whether to release.

---

## Current Open Decisions

- Final parameter count and architecture.
- Exact context length.
- Fixed per-cluster target token weights and the reproducible method used to
  derive them while preserving the accepted source distribution.
- Parallel reader count, queue sizes, and prefetch head start.
- Cache shard size and block size.
- Exact deterministic concurrency/reordering strategy.
- Joint data-pipeline and model-checkpoint commit protocol.
- Whether a document-offset/source-provenance index is kept alongside packed
  sequence shards.
- Any special tokens beyond GPT-2 EOD 50256.
- Whether the 2T presentation target remains justified after evaluation.
- Reasoning datasets and generation process.
- Teacher model used for distillation.
- Final benchmark suite.
- Final compute and storage availability.
- Whether the model will be released publicly.

---

## Current High-Level Training Flow

```text
define goals, evaluation, and compute budget
→ adopt and freeze GPT-2 tokenizer compatibility
→ freeze Nemotron source revision and cluster policy
→ freeze source-proportion cluster weights
→ stream pinned byte ranges with parallel bounded prefetch
→ validate records, exclude cluster 11, and split validation deterministically
→ route training documents into per-cluster queues
→ interleave whole documents with continuous token-deficit scheduling
→ append EOD and pack fixed-length next-token sequences
→ send each sequence block to immutable local cache shards and the trainer queue
→ checkpoint pipeline state together with model-training state
→ finish the first pass with a complete reusable local cache
→ use local shuffled shards for later passes
→ evaluate the base model
→ continue reasoning training and instruction tuning
→ evaluate, optimize, document, and release
```
