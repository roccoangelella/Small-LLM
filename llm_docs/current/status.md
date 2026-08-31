---
status: current
last_reviewed: 2026-08-31
---

# Current project status

This file is high-density working memory. Detailed measurements live in `../evidence/`, durable choices in `../decisions/`, contracts in `../reference/`, and commands in `../runbooks/`.

## Completed pretraining endpoints

| Endpoint | Learned Parameters | Geometry | Checkpoint | Consumed Targets | Status / Artifact |
|---|---:|---|---|---:|---|
| 20M / 100M | 20,637,592 | `d_model=256, d_ff=704`, 8L | `step-00003053` | 100,018,176 | Completed historical scaling point |
| 20M / 500M | 20,637,592 | `d_model=256, d_ff=704`, 8L | `step-00015264` | 500,156,416 | Completed baseline; stable HF artifact |
| 20M / 2B | 20,637,592 | `d_model=256, d_ff=704`, 8L | `step-00061066` | 2,001,000,448 | Completed; capacity-constrained at 2B |
| 100M / 2B | 101,252,280 | `d_model=512, d_ff=1408`, 20L | `step-00015267` | 2,001,000,448 | Completed Modal/H100 artifact (`models/100m-2b-data-001/step-00015267`) |

*All models use context length 2,048.*

## Frozen three-way intrinsic comparison (`eval_core_v1`)

Manifest SHA-256: `aa7b6157e5f420dd53a99552685eaed01962ee45c23cbe438e1321a886422792`

| Metric | 20M / 500M | 20M / 2B | 100M / 2B |
|---|---:|---:|---:|
| Loss | 4.007289 | 3.894576 | **3.338815** |
| Perplexity | 54.997550 | 49.135214 | **28.185701** |
| BPB | 1.250808 | 1.215627 | **1.042155** |
| Top-1 | 0.343950 | 0.355129 | **0.398875** |
| Top-5 | 0.547491 | 0.561084 | **0.618154** |
| Top-10 | 0.620226 | 0.634112 | **0.692041** |
| Cluster macro loss | 4.031874 | 3.899664 | **3.349121** |
| Mixture-weighted cluster loss | **3.529477** | 3.589138 | **3.042600** |

*Interpretation: 20M capacity is constrained by 2B tokens. 100M/2B improves all 19/19 clusters and all 8/8 context buckets.*  
Evidence: [`../evidence/scaling/20m_500m_20m_2b_100m_2b_full_eval_2026-08-13.md`](../evidence/scaling/20m_500m_20m_2b_100m_2b_full_eval_2026-08-13.md).

## Active pretraining: 100M / 10B Deep-Decay Continuation

- **Schedule & Authority**: ADR 0071 / ADR 0095 (deep decay from uncooled step 15,500) / ADR 0114 (Modal H100 execution).
- **Run ID**: `100m-10b-deep-decay-from-step15500` (horizon: 76,294 updates / 10,000,007,168 target tokens).
- **Active Execution Lane**: Modal 1x H100, `microbatch=16`, 4 accumulations, 64-sequence optimizer block; Kaggle dual-T4 is a recovery/continuation lane.
- **Remote Durability Backend (ADR 0132)**:
  - Rolling exact-resume `latest`: private HF Storage Bucket `<REPO_ID>-checkpoints` -> **Bucket `step-00070750`** (block 70,749; val loss 2.827240757760592; rolling publication succeeded before the Kaggle rank-1 abort).
  - Strict validation-loss `best`: dedicated per-run model repo (`<owner>/<base>-best-<run_id>`) -> **`step-00068250`** (val loss 2.824985).
- **Kaggle CPU Gate Repair**: Compares Bucket vs legacy model-repo pointers, verifies newest on CPU, and selects Bucket checkpoints before any GPU allocation (ADR 0132).
- **Dataset**: Pinned ClimbMix 10B shards complete on HF and Beam (21 train / 21 val shards; manifest SHA-256 `d23e7e4641e30c25b56189093bf1270cd11e85efc8b26bc4660af1873edb96f1`). Evidence: [`../evidence/scaling/100m_10b_dataset_completion_2026-08-14.md`](../evidence/scaling/100m_10b_dataset_completion_2026-08-14.md).
- **Execution & Recovery Evidence**:
  - Kaggle rank-1 best-publication abort at step 70,750 and primary-rank side-effect fix: [`../evidence/scaling/100m_10b_kaggle_rank1_best_publication_abort_2026-08-31.md`](../evidence/scaling/100m_10b_kaggle_rank1_best_publication_abort_2026-08-31.md).
  - Modal step 61,750 / 70,250 bucket resume: [`../evidence/scaling/100m_10b_modal_step61750_bucket_resume_2026-08-30.md`](../evidence/scaling/100m_10b_modal_step61750_bucket_resume_2026-08-30.md).
  - Kaggle stale pointer repair: [`../evidence/scaling/100m_10b_kaggle_stale_model_repo_resume_2026-08-31.md`](../evidence/scaling/100m_10b_kaggle_stale_model_repo_resume_2026-08-31.md).
  - Historical Beam logs: [`../evidence/scaling/100m_10b_beam_launch_2026-08-14.md`](../evidence/scaling/100m_10b_beam_launch_2026-08-14.md).

## Production GDN-2 CUDA backend

- **Engine**: Mixed FLA on `fla-core==0.5.2` with FP32 master parameters + FP16 autocast.
- **Chunk configuration**: Saved `gdn_chunk_size=32`; FLA internal runtime chunk 64. Adaptive PyTorch recurrence remains reference fallback.
- Technical contract: [`../reference/gdn2_fla_backend.md`](../reference/gdn2_fla_backend.md).

## Post-training & R-SFT status

- **Parent Baseline**: `100m-2b-sft-s0-001` (privately published at `roccoangelella/small-llm-100m-2b-sft-s0-001`).
- **Accepted R-SFT Chat Model**: `100m-2b-rsft-r0-12306-001` (`step-00000361`, 12,306 unique normalized prompts).
  - CLI: `.venv/bin/python chat.py --model_params 100M --num_tokens 2B --r-sft`
- **Expanded Reasoning Corpus**: `artifacts/rsft-superior-instruction-r0-expanded/reasoning.jsonl` (16,716 unique prompts, SHA-256 `d13052b6fc33108ec65511b790a75f6473144855059b16b55167b046f787c405`, ADR 0106/0108). Native bundle: 417 train blocks / 13,420,823 loss targets.
- **Experimental Landmark (ADR 0117)**: `100m-2b-rsft-r0-16716-e3-001` (`step-00001251`) demonstrated protocol acquisition (14/14 reasoning tags) but regressed eval-core and accuracy; **rejected as qualified default**.
- Evidence: [`../evidence/rsft_e3_full_qualification_2026-08-24.md`](../evidence/rsft_e3_full_qualification_2026-08-24.md), [`../evidence/rsft_expanded_corpus_completion_2026-08-21.md`](../evidence/rsft_expanded_corpus_completion_2026-08-21.md). Active procedure: [`../runbooks/rsft_r0_atomic_production.md`](../runbooks/rsft_r0_atomic_production.md).

## Sources of truth

- Priorities and milestone gates: [`roadmap.md`](roadmap.md)
- Durable choices: [`../decisions/README.md`](../decisions/README.md)
- Architecture & contracts: [`../reference/model_architecture.md`](../reference/model_architecture.md), [`../reference/dataset_and_tokenization.md`](../reference/dataset_and_tokenization.md), [`../reference/training_and_evaluation.md`](../reference/training_and_evaluation.md)
- Active Modal runbook: [`../runbooks/100m_10b_deep_decay_modal.md`](../runbooks/100m_10b_deep_decay_modal.md)
