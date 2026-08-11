---
status: current
last_reviewed: 2026-08-11
---

# Modal training launcher

The canonical new-training entry point is `modal/launch.py`. It is a provider adapter around the existing `dataset`, `model`, and `trainer` packages; it does not define a second scientific trainer. All operator commands for the Modal lane run from the VPS.

## VPS setup

Use the project `.venv` and install the two operator CLIs there:

```bash
source .venv/bin/activate
uv pip install kaggle 'modal>=1.1,<2'
modal setup
modal volume create small-llm-data
modal volume create small-llm-runs
modal volume create small-llm-cache
modal secret create small-llm-training \
  WANDB_API_KEY="$WANDB_API_KEY" \
  HF_TOKEN="$HF_TOKEN" \
  SMALL_LLM_HF_REPO_ID="$SMALL_LLM_HF_REPO_ID"
```

Kaggle is only the remote source for the already-published 2B finite dataset. Authenticate the VPS Kaggle CLI with `KAGGLE_API_TOKEN` or an official token file. No Kaggle notebook participates in the Modal workflow.

## Prepare the 2B Modal corpus from the VPS

The 100M / 2B run uses the `modal-2b-b64` dataset profile rather than editing the historical block-16 corpus. The supported one-command preparation path is:

```bash
python modal/prepare_dataset.py
```

This command discovers the authenticated user's existing Kaggle dataset `small-llm-20m-2b-dataset-001`, downloads it to the VPS only if a verified cached source is absent, verifies production run ID `20m-2b-dataset-001`, performs the byte-preserving block-64 transformation through `dataset.reblock`, verifies the derivative, and uploads it from the VPS to Modal Volume `small-llm-data`.

Fixed VPS paths:

```text
~/small-llm-data/kaggle/small-llm-20m-2b-dataset-001
~/small-llm-data/modal-2b-b64-dataset-001
```

Fixed Modal destination:

```text
/datasets/modal-2b-b64-dataset-001
```

The helper is stage-idempotent. Rerunning normally skips a verified download, a verified reblock, and a matching completed upload. Repair flags are `--force-download`, `--force-reblock`, and `--force-upload`; `--no-upload` stops after VPS preparation.

The original Kaggle dataset remains unchanged. The derived profile is:

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

Dry-run resolution does not rent a GPU. It should resolve the dataset profile to `modal-2b-b64` and the automatic microbatch set to `16,32,48,64`.

## Production shape

```bash
modal run --detach modal/launch.py \
  --model 100M \
  --tokens 2B \
  --gpu H100
```

`H100` is the default and permits the compatible H200 automatic upgrade. The default microbatch value is `0`, meaning benchmark 16, 32, 48, and 64 on the first GPU and freeze the fastest candidate that is finite and stays at or below 90% reserved VRAM.

Use an explicit microbatch only when reproducing a qualified run:

```bash
modal run --detach modal/launch.py \
  --model 100M --tokens 2B --gpu H100 --microbatch-size 32
```

The prepared optimizer block contains 64 sequences. A full block is approximately 131,072 target tokens. Microbatch 16 executes four slices per optimizer update; 32 executes two; 48 executes 48+16; 64 executes one full-block pass.

For the exact existing 2B training stream the reblocked plan has 15,259 optimizer updates, with a final 48-sequence block. Token-space WSD boundaries remain 100,007,936 warmup tokens, 1,499,987,968 stable tokens, and 399,998,976 decay tokens.

## First-run gates

The launcher:

1. requires a clean controlling Git checkout and records its exact commit;
2. discovers exactly one matching finite dataset on the read-only data Volume;
3. performs a full schema-v2 verification once per manifest identity;
4. derives the one-pass WSD plan from the dataset manifest;
5. records GPU name, memory, compute capability, PyTorch, and CUDA runtime;
6. uses a compute-capability-specific persistent Triton cache;
7. runs a short real trainer forward/backward qualification at microbatch 16, 32, 48, and 64 unless one value was explicitly requested;
8. rejects failed/OOM/non-finite candidates and freezes the fastest safe measured result;
9. freezes source commit, model/data identity, precision, block geometry, and selected microbatch;
10. starts online W&B training with 250-update validation/checkpoint cadence.

A 48 or 64 failure on an 80 GB H100 is an expected capacity measurement, not a launcher failure as long as at least one candidate passes. An H200 may admit a larger candidate.

## Checkpointing and W&B

Every 250 successful optimizer updates the trainer writes a verified joint checkpoint under `small-llm-runs/<run-id>/checkpoints/`; the final update is checkpointed as well. The Modal run Volume is the canonical exact-resume checkpoint transport for this trajectory. Legacy Hugging Face dataset-keyed checkpoint publication remains disabled inside the live trainer because that namespace can collide when the same finite corpus is reused by different model sizes.

W&B runs online in project `Small-LLM` with stable run ID `100m-2b-data-001`. Resumes use the same W&B identity and `must` resume semantics after a local durable checkpoint exists.

## Hugging Face checkpoint/model publication

ADR 0044 requires the 100M / 2B artifact to exist on Hugging Face as well as the Modal run Volume. Publication is handled by `modal/publish_hf.py`, which verifies the checkpoint first and stores it under a model/run-specific namespace rather than the legacy dataset-only namespace.

To copy the latest currently verified checkpoint to the configured private Hugging Face model repository while training is still running:

```bash
git pull --ff-only
modal run modal/publish_hf.py --model 100M --tokens 2B
```

The checkpoint is uploaded under:

```text
models/100m-2b-data-001/step-XXXXXXXX/
```

and the pointer/identity metadata is written to:

```text
models/100m-2b-data-001/artifact.json
```

The command is safe to rerun. It publishes the latest verified checkpoint available in `small-llm-runs`; it never reads a partially written checkpoint.

After training completes, require and publish the exact final checkpoint with:

```bash
git pull --ff-only
modal run modal/publish_hf.py --model 100M --tokens 2B --require-complete
```

The final publication command fails closed unless the latest verified checkpoint step equals the qualification plan's full 15,259-step target. This post-run publication does not change, restart, or resume the frozen training trajectory.

## Resume

The run Volume stores `step-XXXXXXXX` joint checkpoints. On every fresh Modal container the launcher verifies candidate `local_manifest.json` files and chooses the newest checkpoint whose block cursor agrees with its step number. The existing trainer then restores model, optimizer, WSD scheduler, FP16 scaler, RNG, counters, and data cursor.

Modal automatic retries use this same path. Manual recovery is the identical launch command from the frozen source commit. The historical Kaggle 20M trajectory has a different W&B identity; do not create a second concurrent Modal invocation of the same 100M run identity.

## Hardware migration policy

The historical FLA acceptance is T4/SM75 evidence. H100/H200 uses a different target architecture, so the first bounded probe is required even though recurrence semantics are unchanged. Keep FP16 for the first platform migration. Treat BF16 or Blackwell as separate follow-up qualifications rather than combining them with this run.

## Adding a future model

Add the accepted geometry to the shared model/trainer preset surface, then register its nominal parameter label in `modal/profiles.py::MODEL_PRESETS`. Token budgets map to finite dataset profiles independently of model size. The block-64 profile is specifically authorized by ADR 0041 for the new Modal 100M / 2B trajectory; later batch changes require their own qualification/decision.
