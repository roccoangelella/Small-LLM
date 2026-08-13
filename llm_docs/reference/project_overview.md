# Project overview

_Last reviewed: 2026-08-13_

## Goal

Build and study a modern dense decoder-only English language-model family below 1B parameters from random initialization, with reproducible data, training, checkpoint, and evaluation contracts. The long-term target is useful general language/knowledge after pretraining and conversational instruction following only after separate post-training.

Coding capability is not an initial target. The pinned source excludes Nemotron-ClimbMix cluster 11, the explicit software/programming cluster, but incidental code can remain because source clusters are not perfectly pure.

## Current model family

The production hybrid macroarchitecture is:

```text
[GDN-2, GDN-2, GDN-2, gated full MHA] × N
```

Completed primary geometries:

| label | learned parameters | d_model | layers | d_ff | completed scaling endpoints |
|---|---:|---:|---:|---:|---|
| 20M | 20,637,592 | 256 | 8 | 704 | 100M, 500M, 2B-token campaigns |
| 100M | 101,252,280 | 512 | 20 | 1,408 | completed 2B-token Modal campaign |

Context remains 2,048. The 100M/2B result is the current strongest completed base-model endpoint by frozen intrinsic evaluation. The next authorized *conditional* lane is fresh 100M/10B with an approximately-5B continuation gate under ADR 0050; the scientific H100 launch still depends on the frozen behavioral gate.

## Production execution

GDN-2 CUDA execution uses `fla-core==0.5.2` under FP32 master parameters plus CUDA FP16 autocast. Checkpoints save `gdn_chunk_size=32`; FLA executes its internal chunk size 64. The adaptive PyTorch recurrence is retained as correctness/reference fallback.

Kaggle production training uses exact-batch two-T4 DDP for qualified finite-data profiles. Modal training is single-H100. Those are execution-topology choices, not model-architecture changes.

## Dataset strategy

The canonical source is pinned GPT-2-tokenized `nvidia/Nemotron-ClimbMix` revision `5eaa64b9c0c85b7f56af01d7dffdb0795816b12b`.

The pipeline:

- accepts clusters 1-10 and 12-20 and excludes cluster 11;
- preserves the measured conditioned source-token mixture;
- assigns validation documents deterministically;
- packs context+1 schema-v2 sequences;
- writes immutable verified little-endian `uint16` shards;
- supports exact resume and dataset/model identity binding.

For new production, Hugging Face Storage Buckets are the only remote dataset durability backend. Google Drive belongs to historical completed artifacts only; legacy schema names containing `drive_` remain readable compatibility fields under ADR 0054.

The fresh 100M/10B path produces approximately-1-GiB HF shards incrementally and uses CPU production/staging before H100 allocation rather than requiring the complete corpus up front.

## Model durability

Hugging Face model-repository storage is unified under ADR 0055:

```text
run/<run_id>/...       live exact-resume checkpoints
models/<run_id>/...    stable completed model artifacts
```

Stable artifacts are native project checkpoints, not Transformers exports. Their integrity contract is `local_manifest.json`; live two-phase publication additionally uses its publication manifest/pointers.

## Evaluation and current evidence

`eval_core_v1` is frozen and provides full/fast intrinsic metrics, domain slices, calibration, context-position buckets, and bootstrap intervals. ADR 0025 separately freezes deterministic full qualitative generation at `temperature=0`, `top_p=1`, `top_k=0`, seed 17, one sample, and global 32-new-token cap.

The completed three-way intrinsic comparison shows continued but uneven 20M data scaling from 500M→2B and a much larger uniform gain from 20M→100M at fixed 2B tokens. See [`../evidence/scaling/20m_500m_20m_2b_100m_2b_full_eval_2026-08-13.md`](../evidence/scaling/20m_500m_20m_2b_100m_2b_full_eval_2026-08-13.md).

## Post-training

The first 20M/500M S0 SFT run learned its masked held-out objective but failed behavioral qualification (0/30 deterministic instruction cases, 100% runaway). Treat it as pipeline evidence, not a promoted SFT recipe.

## Memory precedence

Use `../current/status.md` for present facts, `../current/roadmap.md` for immediate gates, and `../decisions/README.md` for durable authorization. Historical evidence and archives do not override current accepted contracts.
