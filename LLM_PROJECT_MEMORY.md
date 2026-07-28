# Small LLM Project Memory

_Last updated: 2026-07-28_

## Project Goal

Build a decoder-only language model with fewer than 1B parameters from random initialization that:

- speaks and writes good English;
- follows instructions and holds coherent conversations after post-training;
- develops useful basic and intermediate reasoning;
- uses modern small-model architecture and training techniques;
- serves primarily as a learning and research project for an AI MSc student.

The initial scope does **not** target deliberate coding capability. Coding can be added later as a separate extension. Because semantic clusters are imperfect, incidental code may remain in the initial corpus.

---

## Current Resource Assumptions

- Initial development hardware: one NVIDIA T4, probably with microbatch size 1 and gradient accumulation.
- Local VPS storage budget: approximately 400 GB for the live cache, checkpoints, temporary files, and working space.
- Durable bulk storage: a 5 TB Google Drive account.
- Unique pretraining corpus: target 90B accepted source tokens, minimum 80B, hard maximum 100B.
- Working presentation target: up to 2T token presentations through repeated passes, subject to validation and downstream evaluation.
- The first pass should overlap source streaming, preparation, local caching, Google Drive mirroring, and training.

With correct buffering, the expected steady-state bottleneck on a single T4 is GPU compute. Source reading, parsing, cloud transfer, or cache writes must not be allowed to starve the GPU.

---

## Frozen Tokenizer Decision

Use the GPT-2 byte-level BPE vocabulary already embedded in Nemotron-ClimbMix.

- Nemotron-ClimbMix records already contain GPT-2 token IDs.
- Production code reuses those IDs directly; it does not detokenize and retokenize accepted documents.
- Base vocabulary size: 50,257 IDs.
- EOD token: GPT-2 `<|endoftext|>`, ID 50256.
- Cache token storage: explicit little-endian `uint16`.
- Tokenizer files, configuration, and hashes must be frozen before model creation.
- Additional special tokens, if any, must be decided before the embedding matrix is finalized.

Tokenizer training is outside the present project scope unless a concrete limitation is demonstrated.

---

## Frozen Dataset Source and Content Policy

Initial pretraining source:

- Repository: `nvidia/Nemotron-ClimbMix`
- Immutable revision: `5eaa64b9c0c85b7f56af01d7dffdb0795816b12b`
- Included files: root `part_*.tokenized.jsonl` only
- Semantic signal: NVIDIA numeric `cluster_id`
- Accepted clusters: 1-10 and 12-20
- Excluded cluster: 11, NVIDIA's explicit software/programming cluster
- Validation split: deterministic document-level hash, approximately 0.1%

There is no production detokenization, language filter, code-density filter, quality classifier, document-level semantic classifier, or LLM approval pass. The resulting corpus must be described as **programming-cluster-excluded**, not guaranteed code-free.

Verified broad topic map:

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

A bounded live sample confirmed that these IDs are useful broad heuristics but are not perfectly pure. Evidence is stored in `cluster_map_validation.json`.

---

## Why Source Order Cannot Be Used Directly

Nemotron-ClimbMix is locally chunky by cluster. Sequentially consuming its byte regions could expose the optimizer to long topical runs and create an arbitrary curriculum tied to file order and learning-rate phase.

The remote source may remain cluster-ordered, but the optimizer stream must be locally interleaved through deterministic multi-region prefetch and token-based scheduling.

---

## Decided Dataset Architecture

Do not wait for the complete 90B-token corpus before training, and do not build one enormous final `train.bin`.

```text
pinned Nemotron byte ranges
        ↓
deterministic multi-region, memory-bounded readers
        ↓
structural validation and cluster-11 exclusion
        ↓
deterministic train/validation split
        ↓
per-cluster training queues
        ↓
continuous token-deficit scheduler with rolling-mixture backpressure
        ↓
whole documents plus EOD
        ↓
provenance-aware context+1 packer
        ↓
prepared sequence blocks
       ↙                         ↘
local immutable uint16 shards    bounded training-block consumer
       ↓
verified Google Drive mirror
```

The same prepared training block is durably written locally before being exposed to the trainer. Validation blocks use a separate consumer and a separate block-ID namespace.

Later epochs read deterministically shuffled local shards. Missing shards may be restored from Google Drive into a bounded local prefetch window.

---

## Cluster Mixture Policy

The target distribution is defined in **original source tokens**, not documents, physical cache tokens, EOD tokens, padding, overlap copies, or repeated presentations.

The current intent is to preserve the accepted source distribution approximately rather than equalizing clusters or hand-designing a topical curriculum. The final fixed per-cluster token-weight table remains unapproved. It must be derived reproducibly and frozen in configuration and manifests before production.

Mixture counters are continuous across the first-pass stream and do not reset at microbatch, accumulation window, prepared block, shard, checkpoint, interruption, or resume boundaries.

Mixture is monitored cumulatively and over rolling source-token windows such as 1M, 10M, and 100M tokens. Rolling-mixture backpressure may briefly delay a candidate that would worsen excessive local error, but waits must be bounded and must not deadlock when clusters are temporarily unavailable or exhausted.

Validation documents never enter the training scheduler.

---

## Token-Deficit Scheduler Decision

For each accepted training cluster `c`, maintain cumulative original source-token counts:

```text
weight[c]
emitted[c]
total_emitted
deficit[c] = weight[c] * total_emitted - emitted[c]
```

Choose the available cluster with the largest deficit using exact integer/rational arithmetic and a deterministic seeded tie-breaker.

The scheduler emits the entire next document. Overshoot is allowed and carried forward as negative deficit. Never skip or indefinitely defer a long document merely because it exceeds a local quota.

The current schema-v2 implementation also applies rolling-mixture candidate checks and interleaves bounded batches across active work items deterministically.

---

## Frozen Sequence-Packing Contract

For model context length `L`, each stored record contains `L + 1` tokens:

```text
stored: [t0, t1, ..., tL]
input:  [t0, t1, ..., t(L-1)]
target: [t1, t2, ..., tL]
```

The stride is exactly `L`, so consecutive records overlap by one token:

```text
sequence 1: [t0, ..., tL]
sequence 2: [tL, ..., t(2L)]
```

The overlap is physically duplicated but source-counted once. It is the previous sequence's final target and the next sequence's first input, preserving every intended next-token transition.

The packer:

- appends EOD 50256 only when absent;
- concatenates short documents;
- splits long documents;
- distinguishes source, inserted EOD, overlap, and padding provenance;
- carries incomplete state through checkpoints;
- attributes original source tokens exactly across sequence, block, and shard boundaries.

The final context length is still open. The likely development value is 2,048, giving 2,049 stored token IDs and stride 2,048.

---

## Cache Shards

The cache remains permanently sharded. No final merge is required.

```text
dataset/output/
├── train/
│   ├── train-000000.bin
│   └── ...
├── validation/
│   ├── validation-000000.bin
│   └── ...
├── manifest.json
└── progress.json
```

Requirements:

- little-endian `uint16`;
- fixed context+1 geometry recorded in metadata;
- bounded source batches and bounded writes;
- active shards use temporary names;
- complete blocks are flush+fsync durable before trainer visibility;
- finalized shards are atomically renamed and immutable;
- each shard records checksum, byte/token/sequence counts, split-local block range, and exact per-cluster source-token attribution;
- `.tmp` and `.part` files are never exposed to the trainer;
- finalized shards are independently verifiable and reusable.

The physical cache includes small intentional overlap duplication plus EOD, possible padding, metadata, temporary space, checkpoints, and safety margin. Rough 160/180/200 GB estimates for 80B/90B/100B source tokens remain planning approximations.

---

## Durable Storage Roles

### Local VPS SSD

The local filesystem is the live training cache. The trainer reads only complete local shards or first-pass in-memory prepared blocks. Google Drive is not used as a random-access training filesystem.

### Google Drive

Google Drive is the durable mirror for finalized immutable train and validation shards.

- Upload only locally finalized, checksummed shards.
- Use resumable transfers and bounded range downloads.
- Record stable Drive file IDs.
- Verify local SHA-256 and provider metadata before marking a shard remotely durable.
- Never overwrite an immutable logical shard with different bytes.
- Download into `.part`, verify, and atomically install.
- Keep a bounded local prefetch window.

### Private Hugging Face Model Repository

Use one private model repository with logical `last`, `best`, and `run` areas.

- `last`: complete resumable trainer plus pipeline checkpoint.
- `best`: best evaluated model snapshot and metrics; optimizer state is optional and not required by default.
- `run`: stable configurations, hashes, manifests, code revision, tokenizer identity, and approved weights.

Large dataset shards remain on Google Drive. Every `last` checkpoint embeds the exact Drive-manifest snapshot it references.

---

## Fixed-Window Joint Checkpoint Guarantee

Training must be pausable and safely resumable, including migration between VPS providers. Checkpoints occur at a configurable interval and only at a completed optimizer-step boundary with no ambiguous partial update.

Trainer state will eventually include model weights, optimizer, LR scheduler, FP16 scaler, optimizer step, token counters, accumulation position, Python/framework/CUDA RNG states, dataloader/shuffle RNG, and evaluation state.

Dataset/pipeline state includes:

- last consumed and last durable train block IDs;
- separate validation block state;
- durable source-reader/work-plan cursor;
- reconstructable per-cluster queues;
- deficit and rolling-mixture state;
- packer carry and provenance;
- pending prepared sequences;
- finalized writer/shard state;
- Drive durability state and exact manifest snapshot;
- configuration, source, code, tokenizer, and schema hashes.

Publication order:

```text
finish optimizer step and pause consumption
→ finalize all referenced shard tails
→ upload and verify new Drive shards
→ build, fsync, checksum, and atomically finalize local joint checkpoint
→ upload and read-back verify versioned Hugging Face checkpoint
→ publish latest pointer
→ update best only under the frozen metric and direction
→ resume training
```

The previous valid remote checkpoint remains recoverable until the replacement is complete and verified. Unsaved work after the latest checkpoint may be discarded and deterministically replayed.

Required guarantee: no silent skip, no unknown duplicate training range, no model/optimizer state inconsistent with the data cursor, exact restoration of serialized logical state, and portable rollback/resume.

Bitwise-identical future arithmetic is only best effort when hardware, CUDA, dependencies, kernels, and deterministic settings also match.

---

## Dataset Readiness Versus Full-System Readiness

These are separate milestones.

### Dataset-component readiness

The dataset component is ready when it can independently:

- build and resume schema-v2 shards from the pinned source;
- preserve deterministic source ownership and interleaving;
- checkpoint its complete source cursor and producer state;
- pass schema-v2 verification;
- mirror and restore the correct required shard window;
- survive an authenticated bounded Drive/Hugging Face migration test;
- run with approved production cluster weights and frozen operational thresholds.

This milestone does **not** require the final model architecture or real optimizer implementation.

### Full training-system readiness

The complete system additionally requires:

- the model architecture and context length;
- a real framework trainer and block consumer;
- gradient accumulation and optimizer logic;
- framework/CUDA RNG checkpoint integration;
- evaluation and the frozen `best` metric/direction;
- an end-to-end trainer-plus-dataset migration test;
- a bounded live T4 pilot.

Undefined trainer/model behavior must not be mislabeled as a defect in the dataset algorithms. The dataset exposes framework-independent consumer and checkpoint interfaces specifically so model/trainer work can be completed later.

---

## Current Repository Status

Latest reviewed implementation: commit `4ae2f2ebf4592d5772d3c9f9a33db594263f4357`, schema-v2 streaming cache.

### Core dataset algorithms implemented and substantially tested

- exact source filtering, token reuse, and deterministic validation split;
- deterministic multi-region bounded-batch interleaving;
- exact-integer whole-document deficit scheduling;
- rolling-mixture backpressure and compact constant-memory accounting;
- correct context+1 packing with stride `L`;
- exact token provenance across blocks and shards;
- separate train and validation consumers and block namespaces;
- immutable local shards with durability-before-consumer semantics;
- serialization/restoration of queues, scheduler, rolling state, packers, pending prepared sequences, block counters, and previous shard metadata;
- interrupted-versus-continuous producer equivalence tests;
- Google Drive and Hugging Face storage backends plus extensive offline fakes;
- manifest/path/checksum validation and staged empty-VPS restore primitives.

### Remaining dataset-only runtime work

The following remain before declaring the dataset component production-ready:

1. Update `verify` and `status` fully for schema v2 and separate train/validation block namespaces. The current verifier still expects schema v1.
2. Persist and restore a durable remote source-reader/work-plan cursor, including active work-item and record positions.
3. Add executable production orchestration such as `stream-cache build` and `stream-cache resume`; the current CLI is only a weights/configuration preflight.
4. On empty-VPS restore, select the train shard containing `last_consumed_block_id + 1` and the following prefetch window, not simply the first manifest shards.
5. Run a bounded authenticated smoke test against the real Google Drive folder and private Hugging Face repository using dummy or synthetic checkpoint data.
6. Freeze final per-cluster weights, queue/prefetch/mixture thresholds, shard size, and prepared-block size.
7. Implement explicit, verification-gated Hugging Face history cleanup if the project retains the policy of removing superseded checkpoint history.

### Trainer/model-dependent work, not dataset blockers

- Final decoder architecture and parameter count.
- Final context length.
- Real PyTorch trainer adapter and CUDA/framework RNG serialization.
- Optimizer, LR schedule, gradient accumulation, scaler, and checkpoint cadence.
- `best` metric, `min`/`max` direction, and slim inference-only best snapshot.
- Full trainer-integrated interrupted/migrated equivalence proof.

No 90B production run is authorized until both the dataset component and the later trainer/model integration pass their respective acceptance tests.

---

## Immediate Next Steps

1. Complete the remaining dataset-only runtime pass listed above.
2. Freeze the dataset subsystem after schema-v2 verification, source-cursor resume, operational CLI, correct restore window, and authenticated storage smoke tests pass.
3. In parallel, specify and implement a very small decoder-only smoke model and trainer.
4. Use the smoke trainer to validate the joint checkpoint adapter and block acknowledgement contract.
5. Benchmark candidate model sizes and context lengths on the T4.
6. Freeze the final architecture, training hyperparameters, cluster weights, checkpoint cadence, and evaluation suite.
7. Run a bounded live end-to-end pilot before authorizing base pretraining.

---

## Current Open Decisions

### Dataset/runtime

- Final fixed cluster weights and derivation method.
- Reader/batch/queue/prefetch/wait/rolling-error parameters.
- Shard and prepared-block sizes.
- Google Drive run/folder layout and local prefetch/LRU policy.
- Hugging Face repository identity and verified cleanup cadence.
- Whether to retain a detailed document-offset provenance index beyond required block/shard accounting.

### Model/training

- Final architecture and parameter count.
- Exact context length; 2,048 remains the likely development value.
- Any special tokens beyond EOD 50256.
- Optimizer, LR schedule, initialization, global token batch, and checkpoint cadence.
- Evaluation suite and `best` metric/direction.
- Whether a 2T presentation target remains justified.
- Reasoning datasets, teacher model, and post-training process.
- Final compute availability and release policy.

The core tokenizer/source policy, sequence stride, continuous deficit accounting, Google Drive shard role, Hugging Face `last`/`best` split, and fixed-window logical-resume guarantee are no longer open decisions.

---

## Current High-Level Training Flow

```text
freeze goals, evaluation, compute, tokenizer, source, and cluster policy
→ approve source-proportion cluster weights
→ stream pinned byte ranges with deterministic memory-bounded prefetch
→ validate, exclude cluster 11, and split validation
→ interleave whole documents with continuous token-deficit scheduling
→ append EOD and pack context+1 sequences with stride equal to context length
→ durably write local shards and expose train blocks to the trainer
→ mirror and verify finalized shards on Google Drive
→ periodically checkpoint coherent trainer and pipeline state
→ publish verified Hugging Face last and conditionally update best
→ migrate by restoring the checkpoint and the shard containing the next block
→ finish the reusable first-pass cache
→ train later passes from deterministically shuffled local/Drive-restored shards
→ evaluate the base model
→ continue reasoning and instruction training
→ evaluate, optimize, document, and decide on release
```
