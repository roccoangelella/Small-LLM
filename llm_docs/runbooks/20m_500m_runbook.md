# 20M Model / 500M-Token Experiment Runbook

_Last updated: 2026-08-10 Europe/Rome_

The independent seed-17 20M/500M trajectory is completed historical evidence. This runbook records its fixed identity and the current reproduction/resume command surface. Separate 500M launcher, training-overlay, publisher-overlay, and shell-wrapper files have been removed; use `kaggle/launch.py`.

## Fixed dataset identity

```text
profile: 20m-500m-data-scaling-v1
dataset run ID: 20m-500m-dataset-001
target accepted source tokens: 500,000,000
minimum: 450,000,000
maximum: 550,000,000
producer durable checkpoint cadence: 20,000,000 source tokens
context length: 2,048
sequences per optimizer block: 16
target shard size: 8 MiB
source revision: 5eaa64b9c0c85b7f56af01d7dffdb0795816b12b
tokenizer: existing GPT-2 token IDs
programming cluster 11: excluded
remote durability: required
```

The finite dataset is built deterministically on the VPS, mirrored durably to Drive, privately published to Kaggle, round-trip verified, and then consumed from Kaggle-local immutable shards. Do not live-stream the source corpus during GPU training.

## VPS environment

Reference template:

```text
kaggle/500m-publish.env.example
```

Use `KAGGLE_USERNAME` or the dedicated handle namespace:

```env
SMALL_LLM_500M_KAGGLE_DATASET_HANDLE=owner/small-llm-20m-500m-dataset-001
```

Optional profile-specific path overrides remain:

```env
SMALL_LLM_500M_WEIGHTS_FILE=/data/climbmix-mixture-calibration/climbmix_code_free_weights.json
SMALL_LLM_500M_DATASET_DIR=/data/small-llm/20m-500m-dataset-001
SMALL_LLM_500M_OPS_DIR=/data/small-llm/20m-500m-ops
SMALL_LLM_KAGGLE_READY_TIMEOUT_SECONDS=900
```

The unified Python launcher replaces the removed shell wrapper and self-bootstraps publication with `uv`, Python 3.13, `.env`, and `kaggle/requirements-100m-publish.txt`.

## Build, verify, and privately publish

```bash
cd /path/to/Small-LLM
git switch main
git pull --ff-only
python kaggle/launch.py publish --model 20M --tokens 500M
```

Rerun the identical command after interruption. Build/publish resume is automatic and fail-closed.

Successful evidence remains under:

```text
/data/small-llm/20m-500m-ops/
```

with `build-and-push-summary.json` completed/already-published and `kaggle-publish-state.json` verified.

## Fixed training identity

```text
model parameters: 20,637,592
architecture: gdn2_hybrid
fresh initialization seed: 17
context: 2,048
precision: FP16 autocast with FP32 master parameters
optimizer: hybrid Muon + AdamW
learning rate: 3e-4
weight decay: 0.1
schedule: one-pass WSD from the verified 500M plan
training microbatch: 4
saved/configured GDN chunk: 32
CUDA FLA runtime chunk: 64
local checkpoint cadence: 250 updates
held-out validation cadence: 250 updates
verified remote publication cadence: 250 updates
W&B run ID: 20m-500m-data-001
W&B run name: 20M model on 500M tokens
launch commit: c0214d00047c61a290d9a138a6bd94ed5701337c
```

The 500M trajectory is independent of the completed 100M checkpoint. It started from seed 17. Its accepted checkpoint chain later migrated from the adaptive GDN-2 backend to the checkpoint-compatible qualified mixed FLA backend; saved model configuration remains `gdn_chunk_size=32` while FLA executes its internal fixed chunk 64.

## Kaggle launch / exact resume

Attach the exact verified private 500M dataset to an NVIDIA T4 notebook with Internet enabled. Required training secrets are:

```text
WANDB_API_KEY
HF_TOKEN
SMALL_LLM_HF_REPO_ID
optional: WANDB_ENTITY
```

Run:

```bash
cd /kaggle/working/Small-LLM
git switch main
git pull --ff-only
python kaggle/launch.py train --model 20M --tokens 500M
```

Fresh 500M training skips the old microbatch probe because microbatch 4 was already qualified for this model/T4 path. Resume checks only the `20m-500m-dataset-001` remote checkpoint namespace and restores exact model, optimizer, scheduler, FP16 scaler, RNG, and data-cursor state.

For a deliberate bounded diagnostic only:

```bash
python kaggle/launch.py train --model 20M --tokens 500M --max-steps-this-session 250
```

## Canonical completed checkpoint

```text
final checkpoint: step-00015264
consumed training target tokens: 500,156,416
W&B run ID: 20m-500m-data-001
```

Use the frozen evaluation/prompt suites for comparison rather than changing the training contract during historical reproduction.
