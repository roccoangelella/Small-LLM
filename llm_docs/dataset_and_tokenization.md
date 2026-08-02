# Dataset and Tokenization

_Last updated: 2026-08-02_

## Tokenizer contract

The model consumes the GPT-2 byte-level BPE IDs already embedded in the pinned Nemotron-ClimbMix records.

- tokenizer ID: `gpt2`;
- semantic vocabulary size: 50,257;
- EOD token: `<|endoftext|>`, ID 50256;
- cache type: explicit little-endian `uint16`;
- accepted records are not detokenized and retokenized;
- tokenizer training is outside the current project scope unless a concrete limitation is demonstrated;
- additional semantic special tokens, if any, must be decided before finalizing a production embedding matrix.

The model may allocate an internally padded embedding/output matrix of 50,304 rows for hardware alignment. IDs 50,257–50,303 are implementation padding only. They must never occur in the dataset, count as valid targets, or be sampled as outputs.

## Pinned source and content policy

Initial pretraining source:

- repository: `nvidia/Nemotron-ClimbMix`;
- immutable revision: `5eaa64b9c0c85b7f56af01d7dffdb0795816b12b`;
- included files: root `part_*.tokenized.jsonl` only;
- semantic signal: NVIDIA numeric `cluster_id`;
- accepted clusters: 1–10 and 12–20;
- excluded cluster: 11, NVIDIA's explicit software/programming cluster;
- validation split: deterministic document-level hash, approximately 0.1%.

There is no production detokenization, language filter, code-density filter, quality classifier, document-level semantic classifier, or LLM approval pass. Describe the result as **programming-cluster-excluded**, not guaranteed code-free.

The clusters are broad heuristics rather than perfectly pure categories. The broad topic map and bounded sample evidence are retained in the repository, including `cluster_map_validation.json`.

## Exact cluster-mixture contract

The desired training mixture is the empirical source-token distribution of the pinned release, conditioned on cluster 11 being excluded.

For every cluster `c`:

```text
source_tokens[c] = sum(record.token_count for records where cluster_id == c)
```

The production scheduler weights for retained clusters are the exact integer `source_tokens[c]` totals for clusters 1–10 and 12–20. Integer totals are used as relative weights; no rounded percentages or hand-designed curriculum are used.

Cluster 11 is removed by conditioning:

```text
weight[c] / sum(weight[j] for j != 11)
```

The scheduler normalizes integer weights with exact rational arithmetic.

Mixture accounting is continuous across documents, microbatches, gradient-accumulation windows, prepared blocks, shards, checkpoints, interruptions, and resumes. It is not reset per GPU batch.

### Full calibration

PR #3, merged at `a851242ff121a706ac5041319c27bba6c7e1dbf1`, added the resumable full-corpus calibration:

```bash
uv run python -m dataset.mixture \
  --output-dir /data/climbmix-mixture-calibration \
  --workers 8 \
  --max-in-flight-work-items 16
```

The pass scans the approximately 2.04 TB pinned release, reads `cluster_id` and `token_count` without materializing token arrays, checkpoints deterministic work, resumes without double counting, and fails closed on malformed metadata or source/work-plan drift.

Outputs:

```text
work_plan.json
mixture_progress.json
mixture_report.json
climbmix_code_free_weights.json
```

The full scan completed successfully on 2026-08-01 after covering all 100 pinned source files and all 7,457 deterministic work items.

Measured corpus totals:

- source bytes scanned: `1,987,970,304,099`;
- records: `553,315,056`;
- all-cluster source tokens: `356,864,528,972`;
- accepted source tokens after excluding cluster 11: `351,792,454,745`;
- excluded cluster-11 source tokens: `5,072,074,227` (`1.421288%` of all source tokens);
- accepted documents: `544,684,421`;
- excluded cluster-11 documents: `8,630,635`.

The approval review passed all required checks: exact source-byte coverage, positive totals for clusters 1–20, cluster 11 present in the all-cluster report but absent from the accepted weight file, exact agreement between accepted report totals and the weight file, consistent embedded hashes, successful production stream-configuration loading, and byte-identical hashes after a completed `--resume` validation on a copied output directory. The run logged 84 transient first-attempt network warnings, all recovered without exhausted retries, errors, or tracebacks.

Approved calibration identity:

```text
source revision:
5eaa64b9c0c85b7f56af01d7dffdb0795816b12b

work-plan self-hash:
a09e74aea4308528a0035d517d6987a47f7fb0021aa867252f1831a7df82a601

climbmix_code_free_weights.json SHA-256:
76e82e22760adcac59c7294fe9bac11358f5a8b7a26035aae64c3f2e6fa1acb7

mixture_report.json raw-file SHA-256:
52d06f27dd5ed034504a9656cb664d3ded57cd073cc647eca332021cf5bbd07f

mixture_report.json canonical self-hash:
a8b52650e4001dee957cfd9a13cab2a4daacdb58bf1229a0f8ff38f51b035d47
```

The exact production weight file is approved under SHA-256 `76e82e22760adcac59c7294fe9bac11358f5a8b7a26035aae64c3f2e6fa1acb7`. Despite its historical filename, it guarantees only that the explicit programming cluster is excluded; it is not guaranteed code-free.

## Production architecture

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

## Token-deficit scheduler

For each accepted training cluster `c`, track:

```text
weight[c]
emitted[c]
total_emitted
deficit[c] = weight[c] * total_emitted - emitted[c] * sum(weight)
```

Choose the available cluster with the largest deficit using exact arithmetic and a deterministic seeded tie-breaker.

The scheduler:

- emits whole documents;
- carries overshoot forward as negative deficit;
- does not split or indefinitely defer a long document merely to satisfy a local quota;
- monitors cumulative and rolling source-token mixture;
- may briefly delay a candidate through bounded rolling-mixture backpressure but must not deadlock when clusters are temporarily unavailable;
- never sends validation documents into the training scheduler.

## Sequence-packing contract

For context length `L`, each stored sequence contains `L + 1` tokens:

```text
stored: [t0, t1, ..., tL]
input:  [t0, t1, ..., t(L-1)]
target: [t1, t2, ..., tL]
```

Stride is `L`, so consecutive sequences overlap by one physically duplicated token while preserving every intended next-token transition.

The packer:

- appends EOD 50256 only when absent;
- concatenates short documents;
- splits long documents;
- distinguishes source, inserted EOD, overlap, and padding provenance;
- checkpoints incomplete carry state;
- attributes original source tokens across sequence, block, and shard boundaries.

The initial development and architecture-trial context is frozen at 2,048 input tokens, producing 2,049 stored IDs per sequence. Longer contexts are deferred until the base architecture and training pipeline are validated.

## Cache and durability

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

- explicit little-endian `uint16`;
- fixed context+1 geometry in metadata;
- bounded source batches, queues, and writes;
- active shards use temporary names;
- blocks are flush+fsync durable before trainer visibility;
- finalized shards are atomically renamed, immutable, and independently verifiable;
- checksums, counts, split-local block ranges, and per-cluster source-token attribution are recorded;
- `.tmp` and `.part` files are never exposed to the trainer.

Local SSD is the live training cache. Google Drive is the durable mirror, not a random-access training filesystem.

A production cursor advances only after every referenced immutable shard is verified remotely. Failed publication must leave the prior cursor recoverable and permit deterministic replay.

## Model-facing batch contract

The trainer consumer must provide at minimum:

- `input_ids` with shape `[batch, 2048]`;
- `target_ids` with shape `[batch, 2048]`;
- any required loss mask for padding or invalid positions;
- deterministic block and sequence identifiers for resume and audit;
- source/provenance counters needed by joint checkpointing.

The causal mask belongs to full-attention layers. GDN-2 consumes the sequence causally through its recurrent or chunkwise implementation.

## Implementation status

PR #2, merged at `4f7822d128b6b4e563efffd4a197642403a743c3`, added the production dataset orchestrator. The dataset software is considered **code-complete** and should remain frozen except for defects revealed by operational acceptance testing.

Implemented and covered by repository tests include deterministic work plans, bounded range readers, structural validation, cluster exclusion, exact mixture scheduling, context+1 packing, provenance, immutable shards, schema-v2 verification, interruption/resume equivalence, 80B/90B/100B enforcement, Drive mirroring, drift rejection, locking, disk preflight, retry policy, orphan cleanup, and restore primitives.

## Personal Google Drive OAuth

The durable store is a personal Google Drive account. Service-account storage and API-key authentication are not used.

PR #4, merged at `cc8d551b76a0478664d78ccee77414694abdd29b`, added installed-app OAuth using the narrow `https://www.googleapis.com/auth/drive.file` scope. Commit `1ab7b3b8b5abce006512b96c4a153642489ef78e` corrected the Google API `fileId` keyword.

Real upload, metadata-read, download-hash, and cleanup smoke tests passed on 2026-07-28.

Secrets remain local under:

```text
.secrets/google-drive-oauth-client.json
.secrets/google-drive-authorized-user.json
.env
```

They must never be committed. Until automatic dotenv loading is added, production commands use:

```bash
uv run --env-file .env python -m dataset.production ...
```

## Remaining operational gates

1. Run the reproducible authenticated bounded 10M-token dataset pilot.
2. Interrupt and resume it with identical semantic arguments.
3. Run full schema-v2 verification.
4. Verify that a second completed resume uploads no duplicate Drive objects.
5. Confirm that no temporary or finalization-backup artifacts remain.
6. Record throughput, retries, Drive behavior, disk use, and recovery behavior.

Do not start the complete 90B build until the bounded dataset pilot, model/trainer consumer, and small end-to-end training pilot pass.

## Open dataset decisions

- Operational reader, queue, prefetch, and retry settings after the live pilot.
- Final shard and prepared-block sizes after throughput measurements.
- Local cache prefetch/LRU policy during later presentations.
- Retention and cleanup policy for remote dataset and checkpoint history.
- Whether to add automatic `.env` loading inside production and acceptance CLIs.
