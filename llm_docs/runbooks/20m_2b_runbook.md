# 20M model / 2B-token data-scaling reproduction runbook

_Last reviewed: 2026-08-13 Europe/Rome_

This trajectory is **completed**. This runbook is retained for reproduction/resume interpretation; it is not an active authorization to restart or alter the finished experiment.

## Frozen experiment identity

```text
profile: 20m-2b-data-scaling-v1
dataset run ID: 20m-2b-dataset-001
W&B run ID: 20m-2b-data-001
source revision: 5eaa64b9c0c85b7f56af01d7dffdb0795816b12b
tokenizer: GPT-2 token IDs
cluster 11: excluded
context: 2,048
model: 20,637,592-parameter gdn2_hybrid
seed: 17
global optimizer block: 16 sequences
microbatch: 4
final evaluated checkpoint: step-00061066
final consumed loss-bearing targets: 2,001,000,448
```

The run starts fresh relative to 500M; it is not continuation training from `step-00015264`.

## Dataset path

The scientific run used the completed deterministic finite 2B corpus published privately to Kaggle. GPU training reads the attached Kaggle-local schema-v2 shards; it does not stream Nemotron-ClimbMix during the GPU job. Historical publication metadata may contain `drive_*` legacy fields because this dataset predates ADR 0054; that does not re-enable Google Drive for new production.

## Production execution

Current Kaggle production topology for this profile is the qualified exact-batch two-T4 DDP adapter from ADR 0056:

```text
2 × Tesla T4
8 ordered sequences per rank
2 local microbatches per rank at microbatch 4
no_sync on the first local backward
synchronized final backward
one global 16-sequence optimizer update
rank-zero-only W&B/validation/checkpoint publication
```

The scientific global block, token order, optimizer, and checkpoint keys remain topology-neutral.

CUDA GDN-2 uses `fla-core==0.5.2`, FP32 masters + FP16 autocast, serialized chunk 32 / FLA internal chunk 64.

## Human-facing launcher

The registered profile remains available through the unified launcher for verified resume/reproduction operations:

```bash
python kaggle/launch.py train --model 20M --tokens 2B
```

Do not invent a new run ID or silently switch to continuation from the 500M endpoint. Resume must use a checkpoint whose dataset/config identity matches the frozen trajectory.

## Final evaluation

The completed endpoint at `step-00061066` was evaluated on full `eval_core_v1` with loss 3.894576 and perplexity 49.135214. The three-way comparison against 20M/500M and 100M/2B is recorded at [`../evidence/scaling/20m_500m_20m_2b_100m_2b_full_eval_2026-08-13.md`](../evidence/scaling/20m_500m_20m_2b_100m_2b_full_eval_2026-08-13.md).

For exact canonical qualitative comparison, follow ADR 0025 and [`post_pretraining_prompt_suite.md`](post_pretraining_prompt_suite.md), including the global 32-new-token cap.
