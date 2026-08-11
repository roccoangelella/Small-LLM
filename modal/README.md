# Modal training

`modal/launch.py` is the canonical Modal entry point for new single-GPU Small-LLM pretraining runs. It keeps the accepted model/optimizer/checkpoint contract while moving GPU execution, dataset attachment, checkpoint durability, and Triton cache persistence to Modal.

## One-time setup

```bash
python -m pip install 'modal>=1.1,<2'
modal setup
modal volume create small-llm-data
modal volume create small-llm-runs
modal volume create small-llm-cache
modal secret create small-llm-training \
  WANDB_API_KEY="$WANDB_API_KEY" \
  HF_TOKEN="$HF_TOKEN" \
  SMALL_LLM_HF_REPO_ID="$SMALL_LLM_HF_REPO_ID"
```

`WANDB_ENTITY` may be added to the same Secret when needed. `HF_TOKEN` and `SMALL_LLM_HF_REPO_ID` remain available to repository tooling; Modal pretraining checkpoints themselves live in `small-llm-runs` rather than the legacy dataset-keyed Hugging Face checkpoint namespace.

## Reblock the existing 2B corpus for Modal

Do **not** redownload or retokenize the 2B corpus. The verified `20m-2b-dataset-001` records are already fixed context+1 uint16 sequences. For the 100M / 2B Modal trajectory, derive the block-64 corpus locally:

```bash
python -m dataset.reblock \
  --source-dir /local/path/to/20m-2b-dataset-001 \
  --output-dir /local/path/to/modal-2b-b64-dataset-001
```

The reblocker verifies the source dataset, copies the train and validation byte streams in exact order, groups those same sequence bytes into 64-sequence prepared blocks, writes fresh checksums/manifests, and verifies the target `modal-2b-b64` qualification plan. It does not download source documents, tokenize, repack text, reassign splits, or rerun mixture scheduling.

The original block-16 corpus remains unchanged for Kaggle/reproduction. The derived Modal profile is:

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

The reblocker emits a compatibility `drive_manifest.json` whose opaque durable-object IDs are bound to the `small-llm-data` Modal Volume rather than Google Drive. Upload the complete derived directory before training:

```bash
modal volume put small-llm-data \
  /local/path/to/modal-2b-b64-dataset-001 \
  /datasets/modal-2b-b64-dataset-001
```

The launcher searches the read-only `/data` mount and requires exactly one manifest matching the selected token/profile contract. Use `--dataset-dir` when more than one matching copy exists. The first run performs a full schema-v2 verification and records the manifest identities; resumes skip that scan unless the identity changes.

## Launch

Recommended 100M / 2B production command:

```bash
modal run --detach modal/launch.py \
  --model 100M \
  --tokens 2B \
  --gpu H100
```

`H100` intentionally permits Modal's compatible automatic H200 upgrade. Use `--gpu 'H100!'` only when a strict H100 is required for benchmarking.

The human-facing arguments preserve the Kaggle shape where it matters:

```text
--model SIZE
--tokens SIZE
--dataset-dir PATH
--max-steps-this-session N
--dry-run
```

Modal-specific controls are:

```text
--gpu GPU                 default: H100
--microbatch-size N       default: 0 = benchmark 16, 32, 48, 64 and freeze fastest safe result
--precision fp16          first migration is deliberately FP16-only
```

Examples:

```bash
# Resolve configuration without renting a GPU.
modal run modal/launch.py --model 100M --tokens 2B --dry-run

# Force microbatch 32 while still running the first-GPU qualification probe.
modal run --detach modal/launch.py \
  --model 100M --tokens 2B --gpu H100 --microbatch-size 32

# Bounded debugging segment.
modal run modal/launch.py \
  --model 100M --tokens 2B --gpu H100 --max-steps-this-session 250
```

## Current 100M / 2B training contract

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

Microbatching does **not** change the optimizer batch after the block-64 dataset is frozen. A microbatch of 16 runs four forward/backward slices per optimizer update; 32 runs two; 48 runs a 48+16 pair; 64 runs one full-block forward/backward. The first H100/H200 probe determines which is actually fastest and safe.

Changing from the historical block-16 corpus to block 64 does change the 100M run's optimizer batch. It does not change sequence bytes or token-space WSD boundaries. ADR 0041 explicitly authorizes that efficiency tradeoff for this new trajectory.

## GPU and Triton behavior

The production GDN-2 path uses `fla-core==0.5.2` and its Triton `chunk_gdn2` CUDA kernel. Moving from T4 to H100/H200 changes GPU architecture, so Triton compiles a Hopper-targeted kernel on first use. The launcher runs a short real forward/backward training probe before the first optimizer trajectory and persists `TRITON_CACHE_DIR` in `small-llm-cache`, keyed by compute capability.

Do not treat the historical T4 qualification as proof for a new GPU. The first Modal probe is the hardware gate. It requires finite loss/gradients and no more than 90% reserved VRAM, then chooses the fastest successful candidate from 16, 32, 48, and 64 unless one was explicitly requested. With the current activation-heavy MHA/LM-head path, 48 or 64 may fail on an 80 GB H100; that is an expected probe outcome, not a reason to cap the search in advance.

The first platform migration stays on FP16 so the GPU/provider move and the explicitly authorized batch geometry are the controlled changes. BF16 should be qualified separately before becoming a production default.

## Resume and interruption behavior

Modal Functions can be interrupted. The launcher is reentrant:

1. checkpoints are written under `small-llm-runs/<run-id>/checkpoints/`;
2. every candidate resume checkpoint is verified with the repository's local checkpoint manifest;
3. the newest valid `step-XXXXXXXX` is selected automatically;
4. the existing trainer restores model, optimizer, WSD scheduler, FP16 scaler, RNG, counters, and data cursor;
5. Modal retries the same function input in a fresh container after infrastructure failures or the 24-hour function boundary.

Run with `--detach` for long trajectories. Manually rerunning the identical command is also safe. A run freezes its source commit, precision, model geometry, dataset identity, and selected microbatch; incompatible drift is rejected.

Do not run the same W&B/run identity concurrently on Kaggle and Modal.

## Adding later model sizes

The launcher is one profile-driven entry point, not one script per parameter count. Today:

```text
20M  -> trainer model-size smoke
100M -> trainer model-size substantive
```

When a later under-1B geometry is accepted, add its shared model/trainer preset and one entry to `modal/profiles.py::MODEL_PRESETS`; the Modal infrastructure, dataset discovery, microbatch qualification, checkpointing, and resume path stay unchanged.

Operational support in the launcher is not scientific authorization to start a particular experiment.
