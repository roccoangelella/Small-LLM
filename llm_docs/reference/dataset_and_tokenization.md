# Dataset and Tokenization

_Last updated: 2026-08-03_

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

PR #3, merged at `a851242ff121a706ac5041319c27bba6c7e1dbf1`, added the resumable full-corpus calibration. That one-time scan completed and was independently published; ADR 0037 therefore retired its executable from the active Small-LLM dataset package. The original implementation and command remain available in Git history, while the standalone reproducible calibration package lives in `roccoangelella/climbmix-token-mixture`.

The completed pass scanned the approximately 2.04 TB pinned release, read `cluster_id` and `token_count` without materializing token arrays, checkpointed deterministic work, resumed without double counting, and failed closed on malformed metadata or source/work-plan drift.

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

### Public calibration release

On 2026-08-02 the calibration process, focused tests, exact aggregate artifacts, and verification report were published in the public repository `roccoangelella/climbmix-token-mixture` at commit `5ef5839800f712f773f1f9bde7fe5216829f58da`.

The public release contains no source documents, token arrays, credentials, model code, or private project state. It includes the exact work plan, final progress, report, accepted weights, an offline verifier, and the standalone standard-library calibration package. Publication followed additional checks beyond the original approval:

- the pinned Hugging Face tree was re-resolved and exactly matched all 100 published paths, sizes, and the `1,987,970,304,099`-byte total;
- 31 focused ownership, work-plan, metadata, retry, and resume tests passed in the extracted public package;
- 1,000 records, 10 from every source file, were fully JSON-parsed with zero `token_count != len(tokens)` or GPT-2 token-bound mismatches;
- all 20 per-cluster document counts exactly matched the independently published `gvlassis/ClimbMix` counts;
- a completed `--resume` through the public package left all calibration artifact hashes byte-identical.

The related `gvlassis/ClimbMix` release already makes the corpus easier to consume and publishes document-count ratios. The new repository is complementary: it publishes source-token totals and conditioned integer weights for token-budgeted scheduling, so other users do not need to repeat the approximately-2-TB metadata scan.

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

## Authenticated 10M pilot result

The accepted authenticated pilot ran on 2026-08-02 at commit `e4776501d68e39746f8a75dcbb9c49515f215abd` on Linux `aarch64` with Python 3.13.13. It used the approved weight SHA-256 `76e82e22760adcac59c7294fe9bac11358f5a8b7a26035aae64c3f2e6fa1acb7`, the real personal-Google-Drive backend, and the committed 10M target / 9M minimum / 11M maximum / 2M checkpoint policy.

Accepted evidence:

- the first durable checkpoint contained 2,000,112 accepted source tokens, 2,814 consumed documents, one local immutable shard, and one matching Drive entry;
- the actual producer process group was terminated with exit status 143, not merely its wrapper shell;
- `--resume` continued from the identical production identity and completed at 10,000,662 accepted source tokens and 14,136 consumed documents;
- the final cache contained 10,021,659 stored token IDs: 10,011,414 train tokens and 10,245 validation tokens;
- seven immutable local shards occupied 20,043,318 bytes and had seven unique, matching, remotely durable Drive file IDs;
- full schema-v2 verification, a second completed `--resume`, and semantic idempotence all passed;
- the accepted pilot logs contained no warnings, retries, errors, or tracebacks;
- the fail-closed acceptance verifier passed environment, calibration, current offline tests, calibration-run evidence, Drive smoke, pilot, interruption/resume, and completed-resume idempotence.

The canonical report is `/data/climbmix-ops/dataset_acceptance_report.json`, SHA-256 `b18decde4aa0e6e7376c3fecd3dda4406dee983f11224537cf73dd22a66bc00b`. The interrupted snapshot hash is `5df9bfe8df148a93848ff30aa4ca52c120abeac3f507c12dca6e99d71e94f610`; the completed-resume baseline hash is `0633255f2ea3e3fab749eb5afea23b84f9e13d49458fa44a91004878c6fe6f5c`.

## Pilot-derived operational findings

### OAuth and environment setup

The Google client must be an OAuth **Desktop app** client. While the consent screen is in Testing, the authorizing account must be a test user in the same Google Cloud project that owns the downloaded client. `dataset.drive_auth setup` should create `.secrets/google-drive-authorized-user.json`, populate `.env`, and finish with a real upload / metadata-read / download-hash / cleanup smoke test; the authorized-user file must not be handcrafted or committed.

The repository's base dependency set does not include the model stack. A fresh VPS that must run the complete offline suite needs `uv sync --locked --extra model` as well as `uv pip install -r dataset/requirements-remote.txt`. Orchestration must use `uv run python` or the project interpreter; this VPS had no bare `python` executable.

### Interruption semantics

Killing only a background wrapper shell is not an interruption test. In the rejected first attempt, the child producer continued, completed the cache, and caused the attempted resume to fail on the production lock. That attempt was archived locally and on Drive and excluded from acceptance.

Future interruption evidence must launch the producer in a dedicated process group, snapshot only after the referenced shard is remotely durable, terminate the whole group, wait for it to exit, confirm no producer descendant or lock holder remains, and only then issue `--resume`. Exit status 143 is meaningful only together with those process and durable-state checks.

### Resume scalability

The current source-reader contract deliberately replays the immutable source plan from the beginning up to `documents_consumed` to verify the durable cursor before emitting new blocks. At the pilot cursor of 2,814 documents, the resumed run took about 44 seconds before reaching the next 4M-token checkpoint, while later 2M-token checkpoint intervals were about 5–7 seconds. This is consistent with replay dominating early resume time.

That auditable replay is acceptable for the bounded pilot but is a production-scale risk because restart cost grows with the consumed document cursor. Before a 90B launch, late-cursor resume must be benchmarked and either bounded by a more direct seekable cursor/checkpoint design or explicitly accepted with measured recovery-time limits.

### Disk and cache capacity

The accepted cache stored exactly two bytes per token ID: 20,043,318 bytes for 10,021,659 stored IDs. The production disk preflight, including configured EOD overhead and safety multiplier, requires about 222.3 GiB for a 90B target and 247.0 GiB for the 100B hard maximum. The pilot VPS had about 95 GiB free.

Therefore the current VPS cannot retain the entire production cache at once. Full production requires a larger local volume or a proven bounded-cache lifecycle in which the trainer consumes durable shards and an explicit retention/LRU policy safely releases local space. `--allow-unsafe-low-disk` remains restricted to bounded tests and must not bypass this production requirement.

### Mixture and throughput interpretation

The 10M pilot was an operational acceptance run, not a representative training-mixture benchmark. Its training scheduler emitted tokens from seven accepted clusters (`4, 6, 7, 12, 16, 17, 18`); the other twelve accepted clusters emitted zero tokens at this small budget. Cumulative and 10M-window normalized mixture error were `0.08533077992520376`, while the accepted command inherited the production CLI default `maximum_rolling_mixture_error=1.0`. The run therefore did not enforce a tight mixture bound and is not evidence that every cluster is represented in a small experiment.

The production phase took about 116 seconds from initial start through resumed completion, averaging roughly 86k accepted source tokens/s including source resolution, replay, Drive durability, checkpointing, and finalization. The complete orchestrated acceptance sequence took 119 seconds. These short-run values are useful for regression detection only; startup/replay costs, tiny checkpoint shards, cache warmth, and the bounded token budget make them unsuitable as a 90B throughput forecast.

## Remaining dataset gates before full production

1. Complete the schema-v2 trainer consumer and a small end-to-end training pilot with joint checkpointing.
2. Resolve the 90B/100B local-capacity shortfall through a larger disk or a verified bounded eviction/retention design.
3. Measure and bound late-cursor source replay, or replace it with an equally auditable direct-resume mechanism.
4. Use a production-grade process-group orchestrator for all interruption evidence and automated operations.
5. Freeze an explicit production rolling-mixture-error bound; do not inherit the current permissive CLI default of `1.0` without measured justification.
6. Decide whether bounded model-comparison datasets must explicitly bootstrap all accepted clusters or simply use a larger token budget.
7. Decide retention and cleanup for the archived rejected attempt and future superseded Drive run folders.

Do not start the complete 90B build until these gates and the model/trainer integration gates pass.

## Open dataset decisions

- Operational reader, queue, prefetch, and retry settings after live trainer measurement.
- Final shard and prepared-block sizes after representative sustained-throughput measurements.
- Direct/seekable source cursor design versus full replay on resume.
- Local cache capacity, trainer-consumption watermark, and safe eviction/LRU policy.
- Final `maximum_rolling_mixture_error` for production and bounded comparison runs; the current production CLI default is `1.0`.
- Whether bounded comparison datasets require all-cluster bootstrap coverage.
- Retention and cleanup policy for remote dataset and checkpoint history.
- Whether to add automatic `.env` loading inside production and acceptance CLIs.
