# Project Status

_Last updated: 2026-07-31_

## Current phase

The dataset software is code-complete and undergoing operational qualification. The model and trainer are moving from architecture specification into implementation, beginning with the approximately 20M smoke configuration.

The complete 90B dataset build is not authorized yet. The exact mixture calibration, bounded dataset pilot, model/trainer consumer, and small end-to-end training pilot must pass first.

## Completed foundations

### Dataset implementation

PR #2, merged at `4f7822d128b6b4e563efffd4a197642403a743c3`, added the production dataset orchestrator.

Implemented and covered by repository tests include:

- deterministic pinned-source work plans and bounded range readers;
- structural validation and cluster-11 exclusion;
- exact source-token scheduling and rolling mixture accounting;
- context+1 packing and provenance;
- immutable local shards with durability-before-consumer semantics;
- schema-v2 verification;
- durable interruption and resume state;
- deterministic interruption/resume equivalence;
- 80B minimum, 90B target, and 100B hard-maximum enforcement;
- verified Google Drive mirroring before cursor advancement;
- configuration, schema, policy, and work-plan drift rejection;
- locking, disk preflight, retry policy, orphan cleanup, and restore primitives.

The dataset subsystem should remain frozen except for defects revealed by operational acceptance testing.

### Exact-mixture calibration implementation

PR #3, merged at `a851242ff121a706ac5041319c27bba6c7e1dbf1`, added the resumable full-corpus mixture calibration. The generated weight file is not approved until `mixture_report.json` and the file hashes are reviewed.

### Personal Google Drive authentication

PR #4, merged at `cc8d551b76a0478664d78ccee77414694abdd29b`, added installed-app OAuth using the narrow `drive.file` scope. Commit `1ab7b3b8b5abce006512b96c4a153642489ef78e` corrected the Google API `fileId` keyword.

Real upload, metadata-read, download-hash, and cleanup smoke tests passed on 2026-07-28. Secrets remain local under `.secrets/` and `.env` and must never be committed.

### Model architecture

The following are frozen for the initial model family:

- dense decoder-only hybrid;
- Gated DeltaNet-2 as the dominant mixer;
- periodic ordinary MHA layers;
- `[GDN-2, GDN-2, GDN-2, MHA]` repeating pattern;
- sequential pre-RMSNorm blocks;
- final RMSNorm before the tied LM head;
- fixed full-head RoPE on MHA Q and K only;
- dense SwiGLU FFN in every block;
- tied input embeddings and output projection;
- 2,048-token initial context;
- approximately 20M smoke geometry;
- approximately 100M first substantive geometry.

See `model_architecture.md`, `model_geometry.md`, and `decisions_and_ablations.md` for the implementation-level specification.

## Remaining dataset operational gates

1. Complete the exact full mixture calibration.
2. Review `mixture_report.json` and approve the weight-file SHA-256.
3. Finalize the reproducible authenticated acceptance-test harness.
4. Run the bounded 10M-token dataset pilot.
5. Interrupt and resume it with identical semantic arguments.
6. Run full schema-v2 verification.
7. Verify that a second completed resume uploads no duplicate Drive objects.
8. Confirm that no temporary or finalization-backup artifacts remain.
9. Record throughput, retries, Drive behavior, disk use, and recovery behavior.

Until automatic dotenv loading is added, production commands use:

```bash
uv run --env-file .env python -m dataset.production ...
```

## Immediate next steps

1. Complete and approve the full exact mixture calibration.
2. Pass the authenticated bounded dataset pilot and freeze the dataset subsystem.
3. Implement the approximately 20M smoke model using the frozen block specification.
4. Implement exact parameter accounting by component.
5. Connect the model and trainer to the schema-v2 consumer and joint-checkpoint interfaces.
6. Validate forward pass, backward pass, generation, interruption, resume, and migration.
7. Benchmark the smoke geometry and kernels on the T4.
8. Implement and train the approximately 100M hybrid and a parameter-matched all-MHA baseline.
9. Scale only after measured quality, memory, and throughput evidence.

## Current open decisions

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
- Whether MHA uses an attention output gate or an ordinary output projection.
- Exact invalid-logit masking for padded vocabulary rows.
- Larger-scale geometry beyond the frozen smoke and approximately 100M models.

### Training and post-training

- Optimizer and optimizer-state strategy.
- Learning-rate schedule, warmup, peak LR, and minimum LR.
- Global token batch, gradient accumulation, clipping, and precision policy.
- Weight decay and parameter exclusions.
- Checkpoint cadence.
- Evaluation suite and `best` metric/direction.
- Training-token budget for each experiment.
- Whether a 2T repeated-presentation target remains justified.
- Reasoning datasets, teacher model, and post-training procedure.
- Final compute availability and release policy.

## Decisions no longer open

The following are frozen unless a controlled experiment later replaces them:

- source revision and cluster policy;
- GPT-2 tokenizer IDs and EOD token;
- sequence stride and context+1 packing;
- exact empirical-mixture derivation and continuous deficit accounting;
- personal Google Drive identity and durable-mirror role;
- overlapping first-pass dataset/training strategy;
- geometry-scalable model family;
- GDN-2-dominant 3:1 initial pattern;
- ordinary MHA rather than GQA in initial attention layers;
- pre-RMSNorm and final RMSNorm;
- MHA-only RoPE placement;
- dense SwiGLU FFNs;
- tied embeddings;
- initial 2,048-token context;
- approximately 20M smoke geometry;
- approximately 100M first substantive geometry.
