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

## Documentation Policy

`LLM_PROJECT_MEMORY.md` is the compact source of truth for frozen decisions, current status, and open questions.

Detailed topic specifications live in `llm_docs/`:

- `llm_docs/model_architecture.md`
- `llm_docs/model_geometry.md`
- `llm_docs/dataset_and_tokenization.md`
- `llm_docs/training_and_evaluation.md`
- `llm_docs/decisions_and_ablations.md`

When a project decision changes, update both this memory file and the relevant topic document. Do not silently erase replaced decisions; record what changed and which evidence justified the change.

---

## Current Resource Assumptions

- Initial accelerator: one NVIDIA T4.
- Likely initial microbatch size for larger trials: 1, with gradient accumulation.
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
- Semantic vocabulary size: 50,257.
- EOD token: `<|endoftext|>`, ID 50256.
- Cache encoding: explicit little-endian `uint16`.
- Accepted records are not detokenized and retokenized.
- Tokenizer training is outside the current project scope unless a concrete limitation is demonstrated.
- Additional semantic special tokens, if any, must be decided before finalizing a production embedding matrix.

The initial model implementation may pad the physical embedding/output matrix to 50,304 rows for hardware alignment. IDs 50,257–50,303 are implementation padding only: they must never occur in the dataset, count as valid targets, or be sampled as model outputs.

---

## Frozen Dataset Source and Content Policy

Initial pretraining source:

- Repository: `nvidia/Nemotron-ClimbMix`
- Immutable revision: `5eaa64b9c0c85b7f56af01d7dffdb0795816b12b`
- Included files: root `part_*.tokenized.jsonl` only
- Semantic signal: NVIDIA numeric `cluster_id`
- Accepted clusters: 1–10 and 12–20
- Excluded cluster: 11, NVIDIA's explicit software/programming cluster
- Validation split: deterministic document-level hash, approximately 0.1%

There is no production detokenization, language filter, code-density filter, quality classifier, document-level semantic classifier, or LLM approval pass. Describe the result as **programming-cluster-excluded**, not guaranteed code-free.

The clusters are broad heuristics rather than perfectly pure categories. The broad topic map and bounded sample evidence are retained in the repository, including `cluster_map_validation.json`.

---

## Exact Cluster Mixture Decision

The desired training mixture is the empirical source-token distribution of the released Nemotron-ClimbMix corpus, conditioned on cluster 11 being excluded.

For every cluster `c`:

```text
source_tokens[c] = sum(record.token_count for records where cluster_id == c)
```

The production scheduler weights for retained clusters are the exact integer `source_tokens[c]` totals for clusters 1–10 and 12–20. Integer totals are used as relative weights; no rounded percentages or hand-designed curriculum are used.

Cluster 11 is removed by conditioning:

```text
weight[c] / sum(weight[j] for j != 11)
```

The existing scheduler normalizes integer weights with exact rational arithmetic.

PR #3, merged at `a851242ff121a706ac5041319c27bba6c7e1dbf1`, added the resumable full calibration command:

```bash
uv run python -m dataset.mixture \
  --output-dir /data/climbmix-mixture-calibration \
  --workers 8 \
  --max-in-flight-work-items 16
```

It scans the approximately 2.04 TB pinned release, reads `cluster_id` and `token_count` without materializing token arrays, checkpoints deterministic work, and emits:

```text
work_plan.json
mixture_progress.json
mixture_report.json
climbmix_code_free_weights.json
```

The full exact calibration is currently an operational gate. The generated weight file is not approved until the report and hashes are reviewed.

Mixture accounting is continuous across documents, microbatches, accumulation windows, prepared blocks, shards, checkpoints, interruptions, and resumes. It is not reset per GPU batch.

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

The same prepared block is made locally durable before trainer visibility. Validation has a separate consumer and block-ID namespace. Later presentations read deterministically shuffled local shards, restoring missing shards from Google Drive through a bounded local prefetch window.

### Token-deficit scheduler contract

For each accepted training cluster `c`:

```text
weight[c]
emitted[c]
total_emitted
deficit[c] = weight[c] * total_emitted - emitted[c] * sum(weight)
```

Choose the available cluster with the largest deficit using exact arithmetic and a deterministic seeded tie-breaker. Emit whole documents. Carry overshoot forward as negative deficit. Validation documents never enter the training scheduler.

### Sequence-packing contract

For context length `L`, each stored sequence contains `L + 1` tokens:

```text
stored: [t0, t1, ..., tL]
input:  [t0, t1, ..., t(L-1)]
target: [t1, t2, ..., tL]
```

Stride is `L`, so consecutive sequences overlap by one physically duplicated token while preserving every intended next-token transition. The packer tracks source, inserted EOD, overlap, and padding provenance and checkpoints incomplete carry state.

The initial development and architecture-trial context is frozen at 2,048 input tokens, producing 2,049 stored IDs per sequence. Longer contexts are deferred until the base architecture and training pipeline are validated.

### Cache and durability contract

The cache remains permanently sharded. No final merge is required. Shards are explicit little-endian `uint16`, active files use temporary names, finalized shards are immutable and checksummed, and trainer visibility occurs only after flush+fsync durability. Google Drive is the durable mirror, not a random-access training filesystem.

A production cursor advances only after every referenced immutable shard is verified remotely. Failed publication must leave the prior cursor recoverable and permit deterministic replay.

---

## Dataset Implementation and Operational Status

PR #2, merged at `4f7822d128b6b4e563efffd4a197642403a743c3`, added the production dataset orchestrator. The dataset software is considered **code-complete** and should remain frozen except for defects revealed by operational acceptance testing.

Implemented and covered by repository tests include deterministic pinned-source work plans, bounded range readers, structural validation, cluster exclusion, exact mixture scheduling, context+1 packing, provenance, immutable shards, schema-v2 verification, interruption/resume equivalence, 80B/90B/100B enforcement, Google Drive mirroring, drift rejection, locking, disk preflight, retry policy, orphan cleanup, and restore primitives.

PR #4, merged at `cc8d551b76a0478664d78ccee77414694abdd29b`, added personal Google Drive installed-app OAuth using the narrow `drive.file` scope. Commit `1ab7b3b8b5abce006512b96c4a153642489ef78e` fixed the Google API `fileId` keyword. Real upload, metadata, download-hash, and cleanup smoke tests passed on 2026-07-28.

Secrets remain local under `.secrets/` and `.env` and must never be committed. Until automatic dotenv loading is added, production commands use:

```bash
uv run --env-file .env python -m dataset.production ...
```

Remaining operational gates:

1. Complete the exact mixture calibration.
2. Review `mixture_report.json` and approve the weight-file SHA-256.
3. Finalize the reproducible authenticated acceptance-test harness.
4. Run the bounded 10M-token dataset pilot.
5. Interrupt and resume it with identical semantic arguments.
6. Run full schema-v2 verification.
7. Verify a second completed resume uploads no duplicate Drive objects.
8. Confirm no temporary or finalization-backup artifacts remain.
9. Record throughput, retries, Drive behavior, disk use, and recovery behavior.

Do not start the complete 90B build until the exact weights, bounded dataset pilot, model/trainer consumer, and small end-to-end training pilot pass.

---

## Fixed-Window Joint Checkpoint Goal

Training must be pausable and safely resumable, including migration between VPS providers. Checkpoints occur only at completed optimizer-step boundaries.

Trainer state must eventually include model weights, optimizer and scheduler, FP16 scaler, optimizer step and token counters, accumulation position, RNG states, and evaluation state.

Pipeline state must include consumed/durable block IDs, validation state, source/work-plan cursor, scheduler and packer state, pending prepared sequences, finalized shards, Drive manifest snapshot, and hashes for configuration, source, code, tokenizer, schema, and approved weights.

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

No silent skip, unknown duplicate range, or model/data-cursor mismatch is acceptable. Bitwise-identical arithmetic after migration is best effort unless hardware and software environments match exactly.

---

## Frozen Base Model Architecture

The model is a dense decoder-only hybrid with Gated DeltaNet-2 as the dominant sequence mixer and periodic ordinary multi-head full causal attention.

### Decoder pattern

The frozen initial pattern is:

```text
[GDN-2, GDN-2, GDN-2, MHA] × N
```

This is the default 3:1 GDN-2-to-MHA ratio. Other ratios remain later controlled ablations, but the smoke and first substantive configurations use 3:1.

Every block uses sequential pre-norm residual branches:

```text
x = x + Mixer(RMSNorm(x))
x = x + SwiGLU(RMSNorm(x))
```

Use RMSNorm with initial `eps = 1e-6`. Apply one final RMSNorm after the last decoder block and immediately before the tied LM head.

### Full-attention layers

Use ordinary MHA, not GQA, in the initial implementation. Each MHA layer has independent Q, K, and V heads and full causal softmax attention.

Apply fixed full-head RoPE to Q and K only in every MHA layer. Do not rotate V. Start with a conventional RoPE base near 10,000 for the initial 2,048-token context. Do not use learned absolute positional embeddings.

GQA remains a later optimization only if longer contexts, serving throughput, or KV-cache pressure justify it.

### Gated DeltaNet-2 layers

GDN-2 layers use their causal recurrence without explicit positional encoding. Do not apply RoPE inside GDN-2 in the initial implementation.

Follow the reference structure with independent Q/K/V projections, channel-wise erase and write gates, causal recurrent matrix state, short depthwise Q/K/V convolutions with initial kernel size 4, gated output normalization, output projection, chunkwise training, and recurrent inference.

Initial GDN key and value widths, head counts, and per-head dimensions match each other. Do not use grouped value geometry initially. Negative-eigenvalue mode, RoPE in GDN-2, grouped values, and short-convolution variants remain controlled ablations.

### Feed-forward network

Every decoder block uses a dense SwiGLU FFN with SiLU gating:

```text
g = W_gate x
u = W_up x
h = SiLU(g) ⊙ u
y = W_down h
```

`d_ff` is the width of both expanded branches and of the elementwise-combined result; the branches are not concatenated. Every layer owns independent `W_gate`, `W_up`, and `W_down` matrices. There is no activation after `W_down`.

### Embedding and output

Tie the input token-embedding matrix and output language-model projection at every model scale. Apply the final RMSNorm before the tied projection.

The implementation must be fully geometry-configurable and must report exact total and per-component parameter counts.

---

## Frozen Initial Model Geometries

### Approximately 20M smoke model

Purpose: implementation and integration correctness only, not meaningful quality comparison.

| Quantity | Value |
|---|---:|
| Context | 2,048 |
| Residual width `d_model` | 256 |
| Decoder layers | 8 |
| GDN-2 layers | 6 |
| MHA layers | 2 |
| Pattern | `[GDN-2, GDN-2, GDN-2, MHA] × 2` |
| SwiGLU width `d_ff` | 704 |
| MHA heads | 4 |
| MHA head dimension | 64 |
| GDN key heads | 4 |
| GDN value heads | 4 |
| GDN key/value head dimension | 64 |
| Tied embeddings | yes |
| Final RMSNorm | yes |

The exact parameter count must be obtained from the implementation.

### Approximately 100M first substantive model

Purpose: first real architecture comparison against a parameter-matched all-MHA baseline.

| Quantity | Value |
|---|---:|
| Context | 2,048 |
| Residual width `d_model` | 512 |
| Decoder layers | 20 |
| GDN-2 layers | 15 |
| MHA layers | 5 |
| Pattern | `[GDN-2, GDN-2, GDN-2, MHA] × 5` |
| SwiGLU width `d_ff` | 1,408 |
| MHA heads | 8 |
| MHA head dimension | 64 |
| GDN key heads | 8 |
| GDN value heads | 8 |
| GDN key dimension per head | 64 |
| GDN value dimension per head | 64 |
| GDN short-convolution kernel | 4 |
| Tied embeddings | yes |
| Final RMSNorm | yes |
| Semantic vocabulary | 50,257 |
| Initial physical vocabulary padding | 50,304 |

Working parameter estimate:

- tied embedding/output matrix: approximately 25.76M;
- 20 SwiGLU FFNs: approximately 43.25M;
- 5 MHA mixers: approximately 5.24M;
- 15 GDN-2 mixers: approximately 25.67M;
- total: approximately 99.9M plus/minus exact small/reference-specific parameters.

The implemented parameter counter is authoritative.

### Scale templates

The following are planning templates, not yet authorized production models:

| Role | Approx. parameters | `d_model` | Layers | `d_ff` | Heads × head dimension |
|---|---:|---:|---:|---:|---:|
| Kernel smoke | 20M | 256 | 8 | 704 | 4 × 64 |
| Intermediate debug | 44M | 384 | 12 | 1,024 | 6 × 64 |
| First substantive | 100M | 512 | 20 | 1,408 | 8 × 64 |
| Medium trial | approximately 200M | 768 | 20 | 2,048 | 12 × 64 |
| Serious trial | approximately 344M | 1,024 | 20 | 2,816 | 16 × 64 |

Dimensions are chosen to be divisible by hardware-friendly tile sizes. For ordinary MHA, `d_model = n_heads × d_head`. The first substantive model uses `512 = 8 × 64`; its `d_ff = 1408 = 22 × 64`.

Before accepting a larger model, benchmark exact parameters, peak T4 memory, stable microbatch, tokens per second, mixer kernel time, checkpoint behavior, and matched loss curves against the all-MHA baseline.

---

## Immediate Next Steps

1. Complete and approve the exact full mixture calibration.
2. Pass the authenticated bounded dataset pilot and freeze the dataset subsystem.
3. Implement the approximately 20M smoke model using the frozen block specification.
4. Implement exact parameter accounting by component.
5. Connect the model/trainer to the schema-v2 consumer and joint-checkpoint interfaces.
6. Validate forward, backward, generation, interruption, resume, and migration.
7. Benchmark smoke geometry and kernels on the T4.
8. Implement and train the approximately 100M hybrid and a parameter-matched all-MHA baseline.
9. Scale only after measured quality, memory, and throughput evidence.

---

## Current Open Decisions

### Dataset operations

- Final approved exact weight-file hash, pending full calibration.
- Operational reader, queue, prefetch, and retry settings after the live pilot.
- Final shard and prepared-block sizes after throughput measurements.
- Local cache prefetch/LRU policy during later presentations.
- Retention and cleanup policy for remote dataset and checkpoint history.
- Whether to add automatic `.env` loading inside production and acceptance CLIs.

### Remaining architecture details

- Exact bias policy outside reference-required GDN-2 parameters.
- Dropout policy, with zero dropout the leading default.
- Exact initialization, gate initialization, and depth-dependent residual scaling.
- Whether to use QK-Norm.
- Whether MHA uses an attention output gate or ordinary output projection.
- Exact invalid-logit masking implementation for padded vocabulary rows.
- Larger-scale geometry beyond the frozen smoke and approximately 100M models.

### Training and post-training

- Optimizer, LR schedule, initialization implementation, global token batch, clipping, precision policy, and checkpoint cadence.
- Evaluation suite and `best` metric/direction.
- Training-token budget for each experiment.
- Whether a 2T presentation target remains justified.
- Reasoning datasets, teacher model, and post-training procedure.
- Final compute availability and release policy.

The source revision, cluster policy, tokenizer, sequence stride, exact empirical-mixture derivation, continuous deficit accounting, Drive identity and role, overlapping first-pass strategy, geometry-scalable model system, GDN-2-dominant 3:1 pattern for initial models, ordinary MHA, pre-RMSNorm, final RMSNorm, MHA-only RoPE placement, dense SwiGLU FFNs, tied embeddings, initial 2,048-token context, approximately 20M smoke geometry, and approximately 100M first substantive geometry are no longer open decisions.
