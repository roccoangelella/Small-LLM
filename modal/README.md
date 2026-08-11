# Modal training

`modal/launch.py` is the canonical Modal entry point for new single-GPU Small-LLM pretraining runs. All operator interaction for this lane happens from the VPS. Kaggle is only the remote source that already stores the verified 2B finite dataset.

## One-time VPS setup

Use the repository `.venv`:

```bash
cd ~/Projects/Small-LLM
source .venv/bin/activate
uv pip install kaggle 'modal>=1.1,<2'
```

Authenticate Modal on the VPS:

```bash
modal setup
modal volume create small-llm-data
modal volume create small-llm-runs
modal volume create small-llm-cache
modal secret create small-llm-training \
  WANDB_API_KEY="$WANDB_API_KEY" \
  HF_TOKEN="$HF_TOKEN" \
  SMALL_LLM_HF_REPO_ID="$SMALL_LLM_HF_REPO_ID"
```

Authenticate the Kaggle CLI on the same VPS. The preferred current token environment is:

```bash
export KAGGLE_API_TOKEN='YOUR_KAGGLE_API_TOKEN'
```

An official Kaggle token file also works. No Kaggle notebook credentials are needed.

## Prepare the block-64 2B corpus

Do not manually download paths or choose output directories. From the VPS:

```bash
git pull --ff-only
python modal/prepare_dataset.py
```

The helper performs the complete handoff:

```text
Kaggle private dataset
  small-llm-20m-2b-dataset-001
        |
        v
VPS verified source cache
  ~/small-llm-data/kaggle/small-llm-20m-2b-dataset-001
        |
        v
byte-preserving dataset.reblock
        |
        v
VPS derived corpus
  ~/small-llm-data/modal-2b-b64-dataset-001
        |
        v
Modal Volume small-llm-data
  /datasets/modal-2b-b64-dataset-001
```

The command discovers the exact authenticated `owner/small-llm-20m-2b-dataset-001` handle with the Kaggle CLI, downloads/unzips only when the verified source cache is absent, verifies production run ID `20m-2b-dataset-001`, reblocks 16 -> 64 without changing sequence bytes, verifies the derivative, and uploads it to Modal.

It is stage-idempotent. Normal reruns reuse completed stages. Repair controls are:

```bash
python modal/prepare_dataset.py --force-download
python modal/prepare_dataset.py --force-reblock
python modal/prepare_dataset.py --force-upload
python modal/prepare_dataset.py --no-upload
```

If automatic Kaggle owner discovery cannot resolve the private dataset:

```bash
export SMALL_LLM_2B_KAGGLE_DATASET_HANDLE='OWNER/small-llm-20m-2b-dataset-001'
python modal/prepare_dataset.py
```

The derived profile is:

```text
profile: modal-2b-b64
dataset run ID: modal-2b-b64-dataset-001
context: 2,048
sequences per optimizer block: 64
target shard size: 32 MiB
train target tokens: 1,999,994,880
optimizer updates: 15,259
final train block: 48 sequences
```

## Dry run

```bash
modal run modal/launch.py --model 100M --tokens 2B --dry-run
```

Dry-run resolution does not rent a GPU. It should report `modal-2b-b64` and automatic microbatch qualification over 16, 32, 48, and 64.

## Launch

```bash
modal run --detach modal/launch.py --model 100M --tokens 2B --gpu H100
```

`H100` intentionally permits Modal's compatible H200 automatic upgrade. Use `--gpu 'H100!'` only when a strict H100 is required for benchmarking.

The first GPU container runs real forward/backward probes at microbatch 16, 32, 48, and 64. Candidates that OOM, produce non-finite loss/gradients, or reserve more than 90% of GPU memory are rejected. The fastest safe measured candidate is frozen before optimizer step 1.

The optimizer block remains 64 sequences regardless of selected execution microbatch: 16 means four slices/update, 32 means two, 48 means 48+16, and 64 means one full-block pass.

## Training contract

```text
context length: 2,048
prepared optimizer block: 64 sequences
full optimizer update: ~131,072 target tokens
architecture: gdn2_hybrid
saved/configured gdn_chunk_size: 32
FLA internal runtime chunk: 64
precision: FP16 autocast with FP32 master parameters
optimizer: hybrid Muon + AdamW
base LR: 3e-4
weight decay: 0.1
Muon momentum: 0.95
Muon target update RMS: 0.18
gradient clip: 1.0
schedule: manifest-derived one-pass WSD
warmup tokens: 100,007,936
stable tokens: 1,499,987,968
decay tokens: 399,998,976
minimum LR ratio: 0.1
seed: 17
checkpoint cadence: 250 successful updates
validation cadence: 250 successful updates
W&B: stable run ID + exact resume
```

## Checkpointing and resume

Verified joint checkpoints are stored under `small-llm-runs/<run-id>/checkpoints/` every 250 successful optimizer updates and at the final update. Modal Volume durability is canonical for this trajectory; legacy dataset-keyed Hugging Face checkpoint publication remains disabled to avoid cross-model collisions.

W&B runs online in project `Small-LLM` with stable run ID `100m-2b-data-001`. A fresh Modal container verifies available `step-XXXXXXXX` checkpoints, restores the newest valid model/optimizer/scheduler/scaler/RNG/data-cursor state, and resumes the same W&B run.

Manual recovery is the identical VPS command:

```bash
modal run --detach modal/launch.py --model 100M --tokens 2B --gpu H100
```

## GPU and Triton behavior

The production GDN-2 path uses `fla-core==0.5.2` and its Triton `chunk_gdn2` CUDA kernel. Moving from T4 to H100/H200 changes GPU architecture, so Triton compiles a Hopper-targeted kernel on first use. The launcher persists `TRITON_CACHE_DIR` in `small-llm-cache`, keyed by compute capability.

The first platform migration stays on FP16. BF16 or Blackwell should be qualified separately rather than being combined with this trajectory.
