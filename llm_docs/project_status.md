# Project Status

_Last updated: 2026-08-02_

## Current phase

The dataset software, model reference package, corrected T4 qualification harness, and first single-device training system are code-complete and CPU-tested.

The corrected schema-v2 T4 run has now passed every recurrent-versus-chunkwise mathematical parity case. The project is no longer blocked by a demonstrated GDN-2 logic mismatch. The current model-side work is **integrated and operational qualification**:

- use the parity-qualified FP16 chunk-32 candidate for bounded T4 training experiments;
- keep FP16 chunk 64 unqualified because it still produces non-finite values under full-model autocast;
- use normal initialization for the next bounded FP16 smoke run, while keeping the final initialization policy formally open;
- qualify schema-v2 trainer, checkpoint/resume, and longer-run stability;
- address the large throughput gap between ordinary-PyTorch GDN-2 and Plan B.

The complete 90B dataset build and approximately-100M architecture comparison remain unauthorized.

## Completed foundations

### Dataset implementation

PR #2, merged at `4f7822d128b6b4e563efffd4a197642403a743c3`, added the production dataset orchestrator. Implemented and repository-tested behavior includes:

- deterministic pinned-source work plans and bounded range readers;
- structural validation and cluster-11 exclusion;
- exact source-token scheduling and rolling mixture accounting;
- context+1 packing and provenance;
- immutable local shards with durability-before-consumer semantics;
- schema-v2 verification;
- durable interruption and deterministic resume;
- 80B minimum, 90B target, and 100B hard maximum;
- verified Google Drive mirroring before durable cursor advancement;
- configuration, schema, policy, weight, and source drift rejection;
- locking, disk preflight, retry policy, orphan cleanup, and empty-VPS restore primitives.

The dataset subsystem remains frozen except for defects revealed by operational acceptance testing and narrow trainer compatibility surfaces.

### Exact-mixture calibration

PR #3, merged at `a851242ff121a706ac5041319c27bba6c7e1dbf1`, added resumable full-corpus calibration. The complete scan finished successfully on 2026-08-01:

- 100 pinned source files and 7,457/7,457 work items;
- 1,987,970,304,099 source bytes;
- 553,315,056 records;
- 356,864,528,972 all-cluster source tokens;
- 351,792,454,745 accepted source tokens after excluding cluster 11;
- 5,072,074,227 excluded cluster-11 tokens, or 1.421288% of the released corpus.

All documented integrity checks passed, including exact byte coverage, positive counts for every cluster, report/weight agreement, embedded-hash agreement, production configuration loading, and completed-resume idempotence. The 84 transient first-attempt network warnings all recovered; there were no errors, tracebacks, or exhausted retries.

The exact production weight file is approved at SHA-256 `76e82e22760adcac59c7294fe9bac11358f5a8b7a26035aae64c3f2e6fa1acb7`. The work-plan self-hash is `a09e74aea4308528a0035d517d6987a47f7fb0021aa867252f1831a7df82a601`, and the canonical report self-hash is `a8b52650e4001dee957cfd9a13cab2a4daacdb58bf1229a0f8ff38f51b035d47`.

### Remote durability

PR #4, merged at `cc8d551b76a0478664d78ccee77414694abdd29b`, added installed-app Google Drive OAuth with the narrow `drive.file` scope. Real upload, metadata-read, download-hash, and cleanup smoke tests passed on 2026-07-28. Joint checkpoints support verified two-phase publication and empty-VPS restoration.

### Model architecture and reference package

The initial family remains frozen as a dense decoder-only hybrid with dominant GDN-2, periodic full MHA, a `[GDN-2, GDN-2, GDN-2, MHA]` pattern, sequential pre-RMSNorm blocks, final RMSNorm, MHA-only RoPE, MHA QK-RMSNorm and output gating, dense SwiGLU, zero dropout, tied padded embeddings, semantic-logit cropping, and 2,048-token context.

The PyTorch package contains:

- scalable approximately-20M and approximately-100M geometry;
- a readable recurrent GDN-2 oracle;
- a differentiable FP32-internal WY-style chunkwise GDN-2 backend;
- configurable chunk size with a frozen architecture default of 64;
- gated full MHA and `SWA-512`;
- primary, Plan-B, and Plan-C assembly;
- tie-aware parameter accounting and optimizer decay exclusions;
- initialization candidates and CPU numerical tests.

The substantive hybrid has 101,252,280 parameters at `d_ff=1408`. Plan B and Plan C each have 101,237,760 parameters at matched `d_ff=1603`.

### Corrected T4 GDN-2 qualification

The schema-v2 Kaggle run used a Tesla T4, PyTorch 2.10.0 with CUDA 12.8, context 2,048, and microbatch 1.

Mathematical parity passed in all 12 cases:

- chunk sizes 16, 32, and 64;
- FP32 and FP16-quantized recurrence inputs;
- zero-state training and bounded carried-state profiles;
- token outputs, final recurrent state, and gradients for Q, K, V, log-decay, erase, write, and initial state.

The earlier schema-v1 parity failure is conclusively classified as a harness-input defect. It used unnormalized Gaussian Q/K and an order-one random state, unlike the real layer contract.

Full-model operational results were:

- FP32 chunks 16, 32, and 64 passed;
- FP16 chunks 16 and 32 passed;
- FP16 chunk 64 failed with non-finite chunkwise values;
- FP16 chunk 32 is the current qualified GDN-2 candidate at approximately 1,291 tokens/s and 2,347 MiB peak allocated memory;
- Plan B passed at approximately 17,260 tokens/s, around 13.4 times faster than GDN-2 chunk 32 in this short ordinary-PyTorch benchmark.

The current evidence therefore establishes GDN-2 correctness and T4 execution feasibility. Remaining concerns are mixed-precision chunk-64 stability and throughput, not recurrent/chunkwise algebra.

### Initialization screening

The corrected initializer probe used FP16 chunk 32 at context 256.

- normal initialization passed with decreasing loss, finite gradients, and no overflow;
- Xavier initialization failed with NaN gradients and three scaler reductions in three measured steps.

Normal initialization is the current candidate for bounded T4 FP16 smoke work. Final initialization remains open pending repeated and integrated evidence.

### Training system

The `trainer/` package includes:

- live schema-v2 consumer and deterministic immutable-shard reader;
- restored Drive-manifest/cache-window reader for migration;
- prepared-block atomic optimizer updates with internal microbatching;
- semantic next-token cross-entropy;
- AdamW parameter grouping from the model decay contract;
- FP32, CUDA FP16 plus `GradScaler`, and BF16 modes;
- gradient clipping and bounded overflow retry;
- token-count constant and WSD-style schedules;
- token-weighted validation and greedy generation checks;
- complete model, optimizer, scheduler, scaler, and RNG state;
- joint checkpoint save/load at an agreed consumed block;
- deterministic CPU interruption/resume tests;
- a bounded `python -m trainer` CLI.

The trainer CLI still contains a safety gate and message referring to the old T4 parity defect. That code is now stale relative to the corrected result and must be revised before trusted GDN-2 integration runs. This is a consistency task, not evidence that parity remains blocked.

## Remaining operational gates

### Dataset

1. Run the reproducible authenticated 10M-token acceptance pilot.
2. Interrupt and resume it with identical semantic arguments.
3. Pass schema-v2 verification and completed-resume idempotence.
4. Record throughput, retries, Drive behavior, disk use, and cleanup.

### Model and trainer

1. Revise the trainer CLI's obsolete parity-defect gate and wording.
2. Run integrated approximately-20M schema-v2 training with GDN-2 chunk 32, FP16, and normal initialization.
3. Measure longer-run loss, gradients, scaler behavior, memory, throughput, and data starvation.
4. Intentionally interrupt and resume at a joint checkpoint.
5. Restore into an empty environment and continue from the prefetched Drive cache window.
6. Verify the next consumed block, counters, scaler, scheduler, RNG state, and model trajectory.
7. Investigate FP16 chunk 64 or leave it unqualified and explicitly configure chunk 32.
8. Attempt or implement a faster T4-compatible GDN-2 backend and require the same parity contract.
9. Compare the qualified GDN-2 path with Plan B and Plan C under matched training conditions.
10. Repeat initialization screening across seeds and a longer bounded run before freezing it.

## Immediate next steps

1. Pass the authenticated bounded dataset pilot using the approved exact weight file.
2. Align the trainer safety gate with the corrected T4 result.
3. Run integrated smoke training with GDN-2 chunk 32 and normal initialization.
4. Validate interruption, local resume, empty-VPS migration, validation, and generation from trainer-produced checkpoints.
5. Profile or replace the slow ordinary-PyTorch GDN-2 backend.
6. Screen learning rate, global token batch, clipping, decay, and schedule on bounded runs.
7. Train the approximately-100M hybrid and matched Plan-B/Plan-C references only after these gates pass.
8. Scale only from measured quality, stability, memory, and throughput evidence.

## Current open decisions

### Dataset operations

- reader, queue, prefetch, and retry settings after live measurement;
- final shard and prepared-block sizes;
- ongoing local cache prefetch/LRU policy;
- remote checkpoint and dataset retention policy;
- automatic `.env` loading.

### Model operations

- whether to replace the frozen default chunk size 64 with the T4-qualified FP16 candidate 32;
- whether to repair chunk-64 mixed-precision execution or simply leave it unqualified;
- which optimized GDN-2 backend, if any, can close the throughput gap;
- final global and gate initialization;
- depth-dependent residual scaling;
- larger geometries beyond the frozen smoke and approximately-100M models.

### Training and post-training

- optimizer and optimizer-state strategy beyond the AdamW baseline;
- LR schedule, warmup, peak, and floor;
- global token batch, accumulation, clipping, and final precision policy;
- weight decay and any additional exclusions;
- checkpoint/evaluation cadence and best-metric rule;
- token budget for every comparison;
- whether repeated presentation up to 2T tokens remains justified;
- reasoning data, teacher model, post-training procedure, compute, and release policy.

## Decisions no longer open

Frozen choices include the source revision and cluster policy, GPT-2 token IDs and EOD token, context+1 packing and stride, exact empirical-mixture derivation, the approved exact weight-file SHA-256 `76e82e22760adcac59c7294fe9bac11358f5a8b7a26035aae64c3f2e6fa1acb7`, Google Drive's durable-mirror role, overlapping first-pass preparation/training strategy after gates pass, PyTorch, the geometry-scalable model family, the GDN-2-dominant 3:1 pattern, the differentiable chunkwise backend, full MHA in attention layers, QK-RMSNorm and output gating, pre-RMSNorm/final RMSNorm, MHA-only RoPE, dense SwiGLU, zero dropout, tied padded embeddings, initial 2,048 context, smoke and substantive reference geometries, fallback ordering, and matched transformer FFN widths.

The frozen training-system contract remains: schema-v2 prepared blocks are acknowledged only after complete optimizer updates, and joint checkpoints bind the exact consumed block to complete trainer and RNG state. It does not freeze the training-recipe values.
