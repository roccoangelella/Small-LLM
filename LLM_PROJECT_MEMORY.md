# Small LLM Project Memory

_Last updated: 2026-07-31_

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
- Durable dataset storage: personal Google Drive/Google One with approximately 5 TB available.
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

For every cluster `c`:

```text
source_tokens[c] = sum(record.token_count for records where cluster_id == c)
```

The production scheduler weights for retained clusters are the exact integer `source_tokens[c]` totals for clusters 1-10 and 12-20. Integer totals are used as relative weights; no rounded percentages or hand-designed curriculum are used.

Cluster 11 is removed by conditioning:

```text
weight[c] / sum(weight[j] for j != 11)
```

The existing scheduler normalizes integer weights with exact rational arithmetic.

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

Mixture accounting is continuous across documents, microbatches, gradient-accumulation windows, prepared blocks, shards, checkpoints, interruptions, and resumes. It is not reset per GPU batch.

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

## Personal Google Drive OAuth Status

The durable store is a personal Google Drive account. Service-account storage and API-key authentication are not used.

PR #4, merged at `cc8d551b76a0478664d78ccee77414694abdd29b`, added installed-app OAuth support using the narrow `https://www.googleapis.com/auth/drive.file` scope.

Local secret files:

```text
.secrets/google-drive-oauth-client.json
.secrets/google-drive-authorized-user.json
.env
```

`.secrets/` and `.env` are ignored by Git. Real credentials, tokens, folder IDs, API keys, and account identifiers must never be committed.

The setup command is:

```bash
uv run python -m dataset.drive_auth setup \
  --client-secrets .secrets/google-drive-oauth-client.json \
  --token-file .secrets/google-drive-authorized-user.json
```

The command:

- validates the OAuth client type;
- performs one-time browser authorization and obtains a refresh token;
- atomically writes authorized-user credentials;
- refreshes expired access tokens automatically;
- creates or reuses `Small LLM Storage/dataset-shards`;
- writes `SMALL_LLM_GOOGLE_OAUTH_TOKEN` and `SMALL_LLM_DRIVE_FOLDER_ID` to `.env`;
- runs a real upload, metadata-read, download-hash, and cleanup smoke test.

Commit `1ab7b3b8b5abce006512b96c4a153642489ef78e` corrected the Google API keyword from `file_id` to `fileId` in metadata, download, and cleanup calls and updated the regression test.

Operational verification completed on 2026-07-28:

- the existing authorized-user token was reused without a new browser flow;
- all 16 focused Drive OAuth unit tests were reported passing;
- the real Google Drive smoke test passed upload, metadata validation, download SHA-256/MD5 verification, and cleanup;
- the target folder tree exists and is accessible by the authorized personal account.

The OAuth feature PR passed GitHub Actions before merge. The direct `fileId` hotfix has no separate GitHub Actions status attached; its focused unit-test and real smoke-test results are recorded from the live execution.

### Environment-loading note

`dataset.drive_auth setup` writes `.env`, while `dataset.production` reads process environment variables or explicit CLI options. Until automatic dotenv loading is added, invoke production commands with:

```bash
uv run --env-file .env python -m dataset.production ...
```

or pass `--google-oauth-token` and `--drive-folder-id` explicitly.

---

## Remaining Dataset Operational Gates

The Google Drive authentication and tiny real-object smoke-test gate is complete.

The dataset component is not fully operationally qualified until all of the following pass:

1. Run the complete exact mixture calibration on the pinned release.
2. Review `mixture_report.json` and approve the SHA-256 of `climbmix_code_free_weights.json`.
3. Implement or finalize the reproducible authenticated acceptance-test harness.
4. Run the authenticated bounded 10M-token dataset pilot using `uv run --env-file .env`.
5. Interrupt the pilot after a durable checkpoint and resume with identical semantic arguments.
6. Run full schema-v2 verification on the bounded pilot.
7. Confirm a second completed `--resume` does not upload duplicate Drive objects.
8. Confirm no `.tmp`, `.part`, smoke-test, or finalization-backup artifacts remain.
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

## Initial Model Architecture Decision

The initial model direction is a hybrid decoder with **Gated DeltaNet-2 as the dominant sequence mixer** and periodic full causal softmax-attention layers.

- Use Gated DeltaNet-2 for most layers.
- Use ordinary multi-head attention (MHA), not grouped-query attention (GQA), in the periodic full-attention layers for the first implementation.
- The reason for keeping independent key and value heads is that the model is below 1B parameters and the likely development context is only 2,048 tokens. At this scale, GQA's main savings would be KV-cache size and inference bandwidth rather than a dramatic reduction in training compute, while sharing key/value heads would remove some attention capacity.
- The leading macroarchitecture is a 3:1 Gated DeltaNet-2-to-MHA pattern, but the exact ratio remains subject to controlled T4 benchmarks.
- Use **pre-RMSNorm** throughout the decoder: normalize before every sequence-mixer branch and before every FFN branch. Start with `eps = 1e-6`.
- Gated DeltaNet-2 layers use their causal recurrence without explicit positional encoding.
- Every periodic MHA layer uses fixed RoPE on its query and key vectors only; values are not rotated. Start with full-head RoPE and a conventional base near 10,000 for the likely 2,048-token development context.
- RoPE is not applied to FFNs and is not applied inside Gated DeltaNet-2 in the first implementation. Applying RoPE to the recurrent mixer, partial RoPE, learned frequencies, and NoPE in MHA remain possible later ablations.
- Use a dense **SwiGLU FFN with SiLU gating in every decoder block**, after both Gated DeltaNet-2 and MHA mixers. Each layer owns independent `W_gate`, `W_up`, and `W_down` parameters; FFN weights are shared across token positions within a layer but not across layers. The exact intermediate width remains open until model geometry is selected.
- GQA remains a later optimization option if longer contexts, serving throughput, or KV-cache pressure make it worthwhile.

The first architecture comparison should keep tokenizer, data, parameter budget, optimizer, and training tokens matched as closely as possible. A modern all-MHA decoder remains the conventional baseline against which the hybrid is measured.

---

## Immediate Next Steps

1. Run the exact full mixture calibration on the fast-network host.
2. Approve the generated weight and report hashes.
3. Implement or finalize the authenticated bounded acceptance-test harness.
4. Execute the real 10M-token Drive pilot, interruption, resume, idempotence, cleanup, and verification procedure.
5. Freeze the dataset subsystem after the pilot report passes.
6. Specify and implement a very small Gated DeltaNet-2/MHA hybrid smoke model and trainer.
7. Connect the trainer to the schema-v2 block consumer and joint checkpoint interfaces.
8. Benchmark model geometry, the Gated DeltaNet-2-to-MHA ratio, context, and global-batch settings on the T4 against an all-MHA baseline.
9. Pass a bounded end-to-end training and migration pilot before authorizing base pretraining.

---

## Current Open Decisions

### Dataset operations

- Final approved exact weight-file hash, pending full calibration.
- Operational reader, queue, prefetch, and retry settings after the live pilot.
- Final shard and prepared-block sizes after throughput measurements.
- Local cache prefetch/LRU policy during later presentations.
- Retention and cleanup policy for remote dataset and checkpoint history.
- Whether to add automatic `.env` loading inside production and acceptance CLIs rather than relying on `uv run --env-file .env`.

### Model and training

- Final parameter count, depth, width, head geometry, FFN intermediate width, and Gated DeltaNet-2-to-MHA layer ratio.
- Exact context length; 2,048 remains the likely development value.
- Any special tokens beyond EOD 50256.
- Optimizer, LR schedule, initialization, global token batch, and checkpoint cadence.
- Evaluation suite and `best` metric/direction.
- Whether a 2T presentation target remains justified.
- Reasoning datasets, teacher model, and post-training procedure.
- Final compute availability and release policy.

The source revision, accepted/excluded cluster policy, tokenizer, sequence stride, exact empirical-mixture derivation, continuous deficit accounting, personal Google Drive OAuth identity, Google Drive shard role, overlapping first-pass training strategy, Gated DeltaNet-2 as the dominant mixer, ordinary MHA in periodic full-attention layers, pre-RMSNorm placement, RoPE placement in the MHA layers, and dense SwiGLU FFNs in every decoder block are no longer open decisions.