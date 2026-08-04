# Small-LLM

This repository is building a dense decoder-only English language model below 1B parameters. It contains the deterministic pretraining-corpus pipeline in `dataset/`, the hybrid model package in `model/`, and the first single-device pretraining system in `trainer/`.

> [!WARNING]
> The exact mixture and authenticated 10M dataset pilot have passed, but the complete 90B corpus is still unauthorized. The approximately-20M integrated training qualification, production disk/cache plan, and late-cursor resume gates must pass first.

## Current architecture

The primary model is a geometry-scalable dense hybrid:

```text
[GDN-2, GDN-2, GDN-2, gated full MHA] × N
```

It uses sequential pre-RMSNorm blocks, dense SwiGLU FFNs, tied padded embeddings with semantic-logit cropping, MHA QK-RMSNorm and output gating, and a 2,048-token initial context.

The approximately-20M smoke geometry is the integration target. The approximately-100M geometry is the first substantive comparison. Plan B is a matched `SWA-512`/full-attention hybrid, and Plan C is the matched all-MHA baseline.

## Dataset paths

The dataset pipeline provides:

1. **Schema-v2 deterministic streaming cache** in `dataset/src/streaming.py`: validated documents become fixed `context+1` blocks, each active-shard block is flushed and fsynced before trainer visibility, and shards are atomically finalized.
2. **Legacy monolithic binary build** through `dataset.main build`: retained for compatibility and prebuild experiments.

Both use the pinned Nemotron-ClimbMix revision `5eaa64b9c0c85b7f56af01d7dffdb0795816b12b`, accept clusters 1–10 and 12–20, and exclude cluster 11.

The approved exact weight file has SHA-256:

```text
76e82e22760adcac59c7294fe9bac11358f5a8b7a26035aae64c3f2e6fa1acb7
```

The authenticated 10M pilot passed real Drive durability, interruption/resume, schema-v2 verification, and completed-resume idempotence. See `dataset/PRODUCTION_RUNBOOK.md` and `llm_docs/project_status.md`.

The production dataset command is:

```bash
uv run --env-file .env python -m dataset.production ...
```

It enforces the 80B/90B/100B envelope, exact source-token accounting, verified Google Drive durability, configuration-drift rejection, and crash-safe resume.

## Trainer

`trainer/` consumes schema-v2 prepared blocks, treats one complete block as one atomic optimizer update, microbatches its sequences, and acknowledges it only after a successful step.

It supports:

- hybrid whole-matrix Muon + AdamW;
- pure AdamW as a matched control;
- token-count constant or warmup/stable/decay schedules;
- FP32, FP16, and BF16;
- gradient scaling and clipping;
- validation and generation checks;
- joint dataset/model checkpoints.

The trusted T4 FP16 GDN-2 path uses chunk size 32. The CLI resolves that chunk automatically and rejects a different FP16 chunk unless the run is explicitly marked diagnostic.

A bounded trusted smoke command has this shape:

```bash
uv run --extra model python -m trainer \
  --dataset-dir /data/climbmix-training-pilot \
  --checkpoint-dir /data/small-llm-checkpoints \
  --steps <bounded-step-count> \
  --sequences-per-block <selected-training-block-size> \
  --model-size smoke \
  --architecture gdn2_hybrid \
  --gdn-chunk-size 32 \
  --initialization normal \
  --optimizer hybrid_muon_adamw \
  --device cuda \
  --precision fp16 \
  --microbatch-size 1
```

Use `--resume step-XXXXXXXX` with the same semantic configuration. A restored checkpoint can use `--dataset-manifest <checkpoint>/drive_manifest.json` while `--dataset-dir` points at its prefetched `cache/` directory.

The placeholders are deliberate. The accepted 10M operational cache used 512 sequences per block, which means about 1.05M target tokens per optimizer update at context 2,048. That cache is valid dataset evidence but is not the training-qualification cache. See `llm_docs/20m_training_readiness.md`.

## Optimizer

The bounded trainer CLI defaults to the selected hybrid optimizer:

```text
ordinary feature-transform matrices: whole-matrix Muon
embedding, norms, biases, dynamics, depthwise filters: AdamW
```

The first Muon implementation uses FP32 Nesterov momentum and a ten-step hybrid Newton-Schulz update. Routing fails closed, and checkpoint state binds the exact recipe and routed parameter names.

Use the pure-AdamW control explicitly with:

```text
--optimizer adamw
```

See `llm_docs/optimizer_strategy.md`.

## Core contracts

- Every schema-v2 record stores `context_length + 1` little-endian uint16 tokens.
- Stride equals context length, so each real next-token transition is trained once.
- Data ordering and source-token mixture accounting remain deterministic across concurrency, shards, interruptions, and resumes.
- One complete prepared block is one optimizer/update/checkpoint unit.
- Microbatching changes accelerator memory usage, not the effective target-token batch.
- Trainer acknowledgements occur only after complete optimizer steps.
- `CheckpointCoordinator` atomically stores trainer state and data-pipeline state at the same consumed block.
- Immutable shards are mirrored to Google Drive and verified by ID, size, and SHA-256.

## Qualification

The corrected Kaggle/T4 harness is:

```bash
python -m tests.t4_qualification --require-t4 --include-plan-b
```

It established:

- recurrent/chunkwise parity for chunks 16, 32, and 64;
- FP32 full-model execution for chunks 16, 32, and 64;
- FP16 full-model execution for chunks 16 and 32;
- FP16 chunk 64 remains unqualified because it produced non-finite values;
- normal initialization passed the bounded probe;
- chunk 32 is the current trusted FP16 GDN-2 candidate.

Run the complete local test suite with:

```bash
uv run --extra model --with-requirements dataset/requirements-remote.txt python -m unittest discover -v
```

Project decisions and status live in `llm_docs/`. The first-run choices still requiring discussion are centralized in `llm_docs/20m_training_readiness.md`.
