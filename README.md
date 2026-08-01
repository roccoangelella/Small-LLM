# Small-LLM

This repository is building a dense decoder-only English language model below 1B parameters. It now contains the deterministic pretraining-corpus pipeline in `dataset/`, the hybrid model package in `model/`, and the first single-device pretraining system in `trainer/`.

> [!WARNING]
> **Production cluster weights are still open and unapproved.** No complete production corpus run may start until the exact full-corpus mixture report and generated weight-file SHA-256 have been reviewed.

## Current architecture

The primary model is a geometry-scalable dense hybrid:

```text
[GDN-2, GDN-2, GDN-2, gated full MHA] × N
```

It uses sequential pre-RMSNorm blocks, dense SwiGLU FFNs, tied padded embeddings with semantic-logit cropping, MHA QK-RMSNorm and output gating, and a 2,048-token initial context. The approximately-20M smoke geometry is the integration target; the approximately-100M geometry is the first substantive comparison. Plan B is a matched `SWA-512`/full-attention hybrid and Plan C is the matched all-MHA baseline.

## Dataset paths

The dataset pipeline provides:

1. **Schema-v2 deterministic streaming cache** in `dataset/src/streaming.py`: validated documents become fixed `context+1` blocks, each active-shard block is flushed and fsynced before trainer visibility, and shards are atomically finalized.
2. **Legacy monolithic binary build** through `dataset.main build`: retained for compatibility and prebuild experiments.

Both use the pinned Nemotron-ClimbMix revision `5eaa64b9c0c85b7f56af01d7dffdb0795816b12b`, accept clusters 1–10 and 12–20, and exclude cluster 11.

The production dataset command is:

```bash
uv run --env-file .env python -m dataset.production ...
```

It enforces the 80B/90B/100B envelope, exact source-token accounting, verified Google Drive durability, configuration-drift rejection, and crash-safe resume. Run the authenticated bounded pilot in `dataset/PRODUCTION_RUNBOOK.md` before authorizing the complete corpus.

## Trainer

`trainer/` consumes schema-v2 prepared blocks, treats one complete block as one atomic optimizer update, microbatches its sequences, and acknowledges it only after a successful step. It supports AdamW, token-count constant or warmup/stable/decay schedules, FP32/FP16/BF16, gradient scaling and clipping, validation, generation checks, and joint dataset/model checkpoints.

A bounded smoke run against a completed local cache is:

```bash
uv run --extra model python -m trainer \
  --dataset-dir /data/climbmix-pilot \
  --checkpoint-dir /data/small-llm-checkpoints \
  --steps 10 \
  --sequences-per-block <pilot-block-size> \
  --model-size smoke \
  --architecture swa_hybrid \
  --device cuda \
  --precision fp16 \
  --microbatch-size 1
```

Use `--resume step-XXXXXXXX` with the same semantic configuration. A restored checkpoint can use `--dataset-manifest <checkpoint>/drive_manifest.json` while `--dataset-dir` points at its prefetched `cache/` directory.

The CLI defaults are qualification defaults, not frozen substantive-run hyperparameters. See `llm_docs/training_system.md`.

## Core contracts

- Every schema-v2 record stores `context_length + 1` little-endian uint16 tokens.
- Stride equals context length, so each real next-token transition is trained once.
- Data ordering and source-token mixture accounting remain deterministic across concurrency, shards, interruptions, and resumes.
- Trainer acknowledgements occur only after complete optimizer steps.
- `CheckpointCoordinator` atomically stores trainer state and data-pipeline state at the same consumed block.
- Immutable shards are mirrored to Google Drive and verified by ID, size, and SHA-256. Versioned joint checkpoints can be published to a private Hugging Face repository only after all referenced shards are remotely durable.

## Qualification

The Kaggle/T4 hardware harness is:

```bash
python -m tests.t4_qualification --require-t4 --include-plan-b
```

It tests recurrent/chunkwise GDN-2 parity, FP32/FP16 smoke steps, chunk sizes 16/32/64, memory, throughput, overflow behavior, initialization candidates, and the Plan-B fallback. The first T4 run proved execution feasibility but found a blocking parity defect: FP32 exceeded strict recurrent-reference tolerances, FP16 parity was non-finite for all tested chunk sizes, and FP16 chunk 64 also failed the full smoke step. The CLI therefore requires `--allow-unqualified-gdn2` for diagnostic GDN-2 runs; Plan B is the safe trainer-plumbing qualification path until the defect is fixed.

Run the complete local test suite with:

```bash
uv run --extra model --with-requirements dataset/requirements-remote.txt python -m unittest discover -v
```

Project decisions and status live in `llm_docs/`. Dataset operations are documented in `dataset/README.md` and `dataset/PRODUCTION_RUNBOOK.md`.
