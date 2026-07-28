# Small LLM Project Memory

_Last updated: 2026-07-28_

## Project Goal

Build a decoder-only language model with fewer than 1B parameters from random initialization that:

- speaks and writes good English;
- follows instructions and holds coherent conversations after post-training;
- develops useful basic and intermediate reasoning;
- uses modern small-model architecture and training techniques;
- serves primarily as a learning and research project for an AI MSc student.

The initial scope does **not** deliberately target coding capability. Coding may be added later as a separate extension. Because semantic clusters are imperfect, incidental code can remain even after excluding the explicit programming cluster.

---

## Current Resource Assumptions

- Initial accelerator: one NVIDIA T4.
- Likely initial microbatch size: 1, with gradient accumulation.
- Local VPS storage budget: approximately 400 GB for live cache, checkpoints, temporary files, and working space.
- Durable dataset storage: 5 TB Google Drive.
- Unique first-pass corpus target: 90B accepted source tokens.
- Minimum acceptable completed corpus: 80B accepted source tokens.
- Hard maximum: 100B accepted source tokens.
- Potential repeated-presentation target: up to 2T tokens, subject to later validation.

The first pass is intended to overlap source streaming, preparation, local caching, Drive mirroring, and model training. With sufficient buffering, the T4 should be the steady-state bottleneck; network and preprocessing must not starve it.

---

## Frozen Tokenizer Decision

Use the GPT-2 byte-level BPE IDs already embedded in Nemotron-ClimbMix.

- Tokenizer ID: `gpt2`.
- Vocabulary size: 50,257.
- EOD token: `<|endoftext|>`, ID 50256.
- Cache encoding: explicit little-endian `uint16`.
- Accepted records are not detokenized and retokenized.
- Additional special tokens, if any, must be decided before finalizing the embedding matrix.

Tokenizer training is outside the current project scope unless a concrete limitation is demonstrated.

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

There is no production detokenization, language filter, code-density filter, quality classifier, document-level semantic classifier, or LLM approval pass. Describe the result as **programming-cluster-excluded**, not guaranteed code-free.

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

The clusters are broad heuristics rather than perfectly pure categories. Bounded sample evidence is stored in `cluster_map_validation.json`.

---

## Exact Cluster Mixture Decision

The desired training mixture is the empirical source-token distribution of the released Nemotron-ClimbMix corpus, conditioned on cluster 11 being excluded.

For every cluster `c`, calculate:

```text
source_tokens[c] = sum(record.token_count for records where cluster_id == c)
```

The production scheduler weights for retained clusters are the exact integer `source_tokens[c]` totals for clusters 1-10 and 12-20. Integer totals are used as relative weights; no rounded percentages or hand-designed curriculum are used.

Cluster 11 is removed by conditioning:

```text
weight[c] / sum(weight[j] for j != 11)
```

The ratio does not need to be written as floating point. The existing scheduler normalizes integer weights with exact rational arithmetic.

### Calibration implementation

PR #3, merged at `a851242ff121a706ac5041319c27bba6c7e1dbf1`, added:

```bash
uv run python -m dataset.mixture \
  --output-dir /data/climbmix-mixture-calibration \
  --workers 8 \
  --max-in-flight-work-items 16
```

The calibration pass:

- scans the complete pinned release once;
- reads `cluster_id` and `token_count` without constructing Python token arrays;
- uses bounded deterministic byte-range concurrency;
- checkpoints a deterministic completed-work-item prefix;
- resumes without double counting;
- fails closed on malformed metadata, source changes, work-plan drift, missing clusters, or output hash mismatches;
- produces all-cluster totals plus the accepted-cluster weight file.

Outputs:

```text
work_plan.json
mixture_progress.json
mixture_report.json
climbmix_code_free_weights.json
```

The exact full calibration has **not yet been run**. It requires transferring the complete approximately 2.04 TB pinned release on the fast-network host. The generated weight file is not approved until the report and file hashes are reviewed.

Mixture accounting is continuous across documents, microbatches, gradient-accumulation windows, prepared blocks, shards, checkpoints, interruptions, and resumes. It is not reset per GPU batch. Small microbatches therefore do not need to contain every cluster.

---

## Decided Dataset Architecture

Do not wait for the complete 90B-token corpus before training, and do not create one enormous final binary.

```text
pinned Nemotron byte ranges
        ↓
deterministic bounded multi-region readers
        ↓
structural validation and cluster-11 exclusion
        ↓
deterministic train/validation split
        ↓
per-cluster training queues
        ↓
continuous exact token-deficit scheduler
        ↓
whole documents plus EOD
        ↓
provenance-aware context+1 packer
        ↓
prepared sequence blocks
       ↙                         ↘
local immutable uint16 shards    bounded trainer consumer
       ↓
verified Google Drive mirror
```

The same prepared block is made locally durable before trainer visibility. Validation has a separate consumer and block-ID namespace.

Later presentations read deterministically shuffled local shards. Missing shards may be restored from Google Drive into a bounded local prefetch window.

---

## Token-Deficit Scheduler Contract

For each accepted training cluster `c`, track cumulative original source-token counts:

```text
weight[c]
emitted[c]
total_emitted
deficit[c] = weight[c] * total_emitted - emitted[c] * sum(weight)
```

Choose the available cluster with the largest deficit using exact integer/rational arithmetic and a deterministic seeded tie-breaker.

The scheduler emits whole documents. Overshoot is carried forward as negative deficit. A long document is not split or indefinitely deferred merely to satisfy a local quota.

Mixture is monitored cumulatively and over rolling source-token windows such as 1M, 10M, and 100M tokens. Bounded rolling-mixture backpressure may delay a candidate briefly but must not deadlock when clusters are temporarily unavailable.

Validation documents do not enter the training scheduler.

---

## Frozen Sequence-Packing Contract

For context length `L`, each stored sequence contains `L + 1` tokens:

```text
stored: [t0, t1, ..., tL]
input:  [t0, t1, ..., t(L-1)]
target: [t1, t2, ..., tL]
```

Stride is `L`, so consecutive sequences overlap by one token. The overlap is physically duplicated but source-counted once, preserving every intended next-token transition.

The packer:

- appends EOD 50256 only when absent;
- concatenates short documents;
- splits long documents;
- distinguishes source, inserted EOD, overlap, and padding provenance;
- checkpoints incomplete carry state;
- attributes original source tokens across sequence, block, and shard boundaries.

Final context length remains open. The likely development value is 2,048, producing 2,049 stored IDs per sequence.

---

## Cache and Durability Contract

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
├── progress.json
├── drive_manifest.json
└── work_plan.json
```

Requirements:

- little-endian `uint16`;
- fixed context+1 geometry in metadata;
- bounded source batches, queues, and writes;
- active shards use temporary names;
- blocks are flush+fsync durable before trainer visibility;
- finalized shards are atomically renamed and immutable;
- checksums, counts, split-local block ranges, and per-cluster source-token attribution are recorded;
- `.tmp` and `.part` files are never exposed to the trainer;
- finalized shards are independently verifiable and reusable.

Local SSD is the live training cache. Google Drive is the durable mirror, not a random-access training filesystem.

A production cursor advances only after every referenced immutable shard is verified remotely. Failed publication leaves the prior cursor recoverable and permits deterministic replay.

---

## Dataset Production Implementation Status

PR #2, merged at `4f7822d128b6b4e563efffd4a197642403a743c3`, added the production dataset orchestrator.

Implemented and covered by repository tests:

- deterministic pinned-source work plans and bounded range readers;
- structural record validation and cluster-11 exclusion;
- exact source-token scheduling and rolling mixture accounting;
- context+1 packing and exact provenance;
- immutable local shards with durability-before-consumer semantics;
- schema-v2 verification;
- durable source-reader and producer resume state;
- deterministic interruption/resume equivalence;
- 80B minimum, 90B target, and 100B hard maximum enforcement;
- whole-document stopping;
- 1B accepted-source-token production checkpoint cadence;
- verified Google Drive mirroring before cursor advancement;
- configuration, schema, policy, and work-plan drift rejection;
- single-writer locking, disk preflight, retry policy, orphan cleanup, and interrupted-finalization recovery;
- correct shard-window restore primitives;
- CI on Python 3.13.

The dataset software implementation is considered **code-complete**. It should now be frozen except for defects revealed by operational acceptance testing.

---

## Remaining Dataset Operational Gates

The dataset component is not fully operationally qualified until all of the following pass:

1. Run the complete exact mixture calibration on the pinned release.
2. Review `mixture_report.json` and approve the SHA-256 of `climbmix_code_free_weights.json`.
3. Configure the real Google Drive service account and production folder.
4. Run the authenticated bounded 10M-token dataset pilot.
5. Interrupt the pilot after a durable checkpoint and resume with identical arguments.
6. Run full verification on the bounded pilot.
7. Confirm a second completed `--resume` does not upload duplicate Drive objects.
8. Confirm no `.tmp`, `.part`, or finalization-backup artifacts remain.
9. Record throughput, retry counts, Drive upload behavior, disk use, and recovery behavior.

The pilot commands and acceptance criteria are documented in `dataset/PRODUCTION_RUNBOOK.md`.

A separate private Hugging Face checkpoint test belongs to trainer/joint-checkpoint integration. The dataset-only production command mirrors dataset shards to Google Drive and does not start a trainer.

---

## 90B Build Timing Decision

Do **not** start the complete 90B build now.

Correct production launch sequence:

1. Approve exact cluster weights.
2. Pass the authenticated bounded dataset pilot.
3. Implement the model and trainer consumer.
4. Pass a small end-to-end trainer-plus-dataset pilot.
5. Start the production dataset process.
6. Establish a bounded cache head start.
7. Start model training while dataset preparation and Drive mirroring continue concurrently.

A full prebuild is justified only if measurements show that source reading/preparation cannot keep the GPU fed. Even then, build only the headroom needed to avoid starvation unless evidence supports a complete prebuild.

---

## Fixed-Window Joint Checkpoint Goal

Training must be pausable and safely resumable, including migration between VPS providers. Checkpoints occur only at completed optimizer-step boundaries.

Trainer state must eventually include:

- model weights;
- optimizer and LR scheduler;
- FP16 scaler;
- optimizer step and token counters;
- accumulation position;
- Python, framework, CUDA, and data-order RNG states;
- evaluation state.

Pipeline state must include:

- last consumed and last durable block IDs;
- validation state;
- durable source/work-plan cursor;
- queue, scheduler, rolling-mixture, and packer state;
- pending prepared sequences;
- finalized shard state;
- exact Drive manifest snapshot;
- configuration, source, code, tokenizer, schema, and approved-weight hashes.

Publication order:

```text
finish optimizer step and pause consumption
→ finalize referenced shard tails
→ upload and verify Drive shards
→ atomically finalize local joint checkpoint
→ upload and read-back verify versioned private-Hub checkpoint
→ publish latest pointer
→ conditionally update best
→ resume training
```

No silent skip, unknown duplicate range, or model/data-cursor mismatch is acceptable. Bitwise-identical arithmetic after migration is best effort unless hardware and software environments also match exactly.

---

## Immediate Next Steps

1. Merge status: production dataset and exact-mixture scanner are already merged.
2. Run the exact full mixture calibration on the fast-network host.
3. Approve the generated weight and report hashes.
4. Have a coding agent prepare and document the authenticated bounded acceptance-test harness.
5. Execute the real Drive pilot, interruption, resume, and verification procedure.
6. Freeze the dataset subsystem after the pilot report passes.
7. Specify and implement a very small decoder-only smoke model and trainer.
8. Connect the trainer to the schema-v2 block consumer and joint checkpoint interfaces.
9. Benchmark candidate architecture/context/global-batch settings on the T4.
10. Pass a bounded end-to-end training and migration pilot before authorizing base pretraining.

---

## Current Open Decisions

### Dataset operations

- Final approved exact weight-file hash, pending full calibration.
- Real Google Drive folder identity and service-account deployment.
- Operational reader/queue/prefetch settings after the live pilot.
- Final shard and prepared-block sizes after throughput measurements.
- Local cache prefetch/LRU policy during later presentations.
- Retention/cleanup policy for remote checkpoint history.

### Model and training

- Final architecture and parameter count.
- Exact context length; 2,048 remains the likely development value.
- Any special tokens beyond EOD 50256.
- Optimizer, LR schedule, initialization, global token batch, and checkpoint cadence.
- Evaluation suite and `best` metric/direction.
- Whether a 2T presentation target remains justified.
- Reasoning datasets, teacher model, and post-training procedure.
- Final compute availability and release policy.

The source revision, accepted/excluded cluster policy, tokenizer, sequence stride, exact empirical-mixture derivation, continuous deficit accounting, Google Drive shard role, and overlapping first-pass training strategy are no longer open decisions.
