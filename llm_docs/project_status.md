# Project Status

_Last updated: 2026-08-01_

## Current phase

The dataset software, model reference package, corrected T4 qualification harness, and first single-device training system are code-complete and CPU-tested. The first T4 run proved GDN-2 execution feasibility and exposed a real FP16 chunk-64 operational failure, but its recurrent/chunkwise parity conclusion was invalidated by unrealistic test inputs. The project is in **correctness and operational qualification**: rerun the corrected model-like parity test, approve the exact mixture, pass the authenticated bounded dataset pilot, and validate trainer interruption/resume.

The complete 90B dataset build and approximately-100M architecture comparison are not authorized yet.

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

The dataset subsystem remains frozen except for defects revealed by operational acceptance testing and narrow compatibility surfaces needed by the trainer.

### Exact-mixture calibration

PR #3, merged at `a851242ff121a706ac5041319c27bba6c7e1dbf1`, added resumable full-corpus calibration. The generated report and weight file are not approved yet.

### Remote durability

PR #4, merged at `cc8d551b76a0478664d78ccee77414694abdd29b`, added installed-app Google Drive OAuth with the narrow `drive.file` scope. Real upload, metadata-read, download-hash, and cleanup smoke tests passed on 2026-07-28. Joint checkpoints support verified two-phase publication and empty-VPS restoration.

### Model architecture and reference package

The initial family is frozen as a dense decoder-only hybrid with dominant GDN-2, periodic full MHA, a `[GDN-2, GDN-2, GDN-2, MHA]` pattern, sequential pre-RMSNorm blocks, final RMSNorm, MHA-only RoPE, MHA QK-RMSNorm and output gating, dense SwiGLU, zero dropout, tied padded embeddings, semantic-logit cropping, and 2,048-token context.

The PyTorch package contains:

- scalable approximately-20M and approximately-100M geometry;
- a readable recurrent GDN-2 oracle;
- a differentiable FP32-internal WY-style chunkwise GDN-2 backend;
- default chunk size 64 with shorter final chunks;
- gated full MHA and `SWA-512`;
- primary, Plan-B, and Plan-C assembly;
- tie-aware parameter accounting and explicit optimizer decay exclusions;
- initialization candidates and CPU numerical tests.

The substantive hybrid has 101,252,280 parameters at `d_ff=1408`. Plan B and Plan C each have 101,237,760 parameters at matched `d_ff=1603`.

### T4 qualification harness

The corrected Kaggle/T4 harness separates mathematical parity from operational mixed-precision behavior.

Parity now uses L2-normalized Q/K and two model-relevant FP32-state profiles: zero state for independent training records and a small bounded state for carried-state/cache behavior. It compares token outputs, final state, and named gradients outside CUDA autocast for FP32 and FP16-quantized inputs. The schema-v1 harness used unnormalized Gaussian Q/K and an order-one random state; its first parity failures are therefore reclassified as a test-harness defect rather than evidence of a GDN-2 algebra defect.

The full-model benchmark remains unchanged in purpose: it runs real context-2,048 optimizer steps under CUDA FP16 autocast, measures throughput and memory, and catches operational failures. The first run validly established that FP32 chunks 16/32/64 and FP16 chunks 16/32 can execute short smoke steps, while FP16 chunk 64 fails with non-finite values and Plan B succeeds. A corrected schema-v2 Kaggle report is now required before GDN-2 is trusted for pretraining.

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
- complete model/optimizer/scheduler/scaler/RNG state;
- joint checkpoint save/load at an agreed consumed block;
- deterministic CPU interruption/resume tests;
- a bounded `python -m trainer` CLI.

This freezes the implementation boundary only. Optimizer, schedule, LR, batch, initialization, cadence, and token budget remain experiment decisions. Trusted GDN-2 CLI training remains blocked until the corrected T4 qualification passes.

## Remaining operational gates

### Dataset

1. Complete and inspect the exact full mixture calibration.
2. Approve `mixture_report.json` and the weight-file SHA-256.
3. Finalize and run the reproducible authenticated 10M-token acceptance pilot.
4. Interrupt/resume it with identical semantic arguments.
5. Pass full schema-v2 verification and completed-resume idempotence.
6. Record throughput, retries, Drive behavior, disk use, and cleanup.

### Model and trainer

1. Rerun schema-v2 T4 parity with normalized Q/K, zero-state training, and bounded carried-state profiles.
2. Determine whether any remaining failure is mathematical, FP16/autocast-specific, or chunk-size-specific.
3. Select a trustworthy GDN-2 chunk candidate, or use the documented fallback if correction is not practical.
4. Qualify trainer/checkpoint plumbing on a verified schema-v2 pilot with Plan B while GDN-2 remains blocked.
5. Intentionally interrupt and resume at a joint checkpoint.
6. Restore into an empty environment and continue from the prefetched Drive cache window.
7. Verify the next consumed block, counters, scaler, scheduler, and model trajectory.
8. Record throughput, peak memory, starvation, checkpoint latency, and numerical stability.
9. Repeat integrated qualification with corrected GDN-2 before freezing initialization or training policy.

## Immediate next steps

1. Finish and approve mixture calibration.
2. Pass the authenticated bounded dataset pilot.
3. Run the corrected schema-v2 T4 harness and inspect parity plus FP16 chunk behavior.
4. If chunk-64 FP16 still fails after parity passes, isolate autocast-sensitive operations or choose a smaller qualified chunk.
5. Run the integrated approximately-20M schema-v2 trainer qualification with Plan B for plumbing and corrected GDN-2 for architecture validation.
6. Validate interruption, local resume, empty-VPS migration, validation, and generation from trainer-produced checkpoints.
7. Screen learning rate, global token batch, initialization, decay, clipping, and schedule on bounded runs.
8. Train the approximately-100M hybrid and matched Plan-B/Plan-C references only after the preceding gates pass.
9. Scale only from measured quality, memory, and throughput evidence.

## Current open decisions

### Dataset operations

- approved exact weight-file hash;
- reader, queue, prefetch, and retry settings after live measurement;
- final shard and prepared-block sizes;
- ongoing local cache prefetch/LRU policy;
- remote checkpoint and dataset retention policy;
- automatic `.env` loading.

### Remaining architecture details

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

Frozen choices include the source revision and cluster policy, GPT-2 token IDs and EOD token, context+1 packing and stride, exact empirical-mixture derivation, Google Drive's durable-mirror role, overlapping first-pass preparation/training strategy after gates pass, PyTorch, the geometry-scalable model family, the GDN-2-dominant 3:1 pattern, the differentiable chunkwise backend with default chunk size 64, full MHA in attention layers, QK-RMSNorm and output gating, pre-RMSNorm/final RMSNorm, MHA-only RoPE, dense SwiGLU, zero dropout, tied padded embeddings, initial 2,048 context, smoke and substantive reference geometries, fallback ordering, and matched transformer FFN widths.

The frozen training-system contract is: schema-v2 prepared blocks are acknowledged only after complete optimizer updates, and joint checkpoints bind the exact consumed block to complete trainer and RNG state. It does not freeze the values of the training recipe.
