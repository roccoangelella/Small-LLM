---
status: current
last_reviewed: 2026-08-11
---

# Modal training launcher

The canonical new-training entry point is `modal/launch.py`. It is a provider adapter around the existing `dataset`, `model`, and `trainer` packages; it does not define a second scientific trainer.

## Setup

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

## Prepare the 2B Modal corpus

The 100M / 2B run uses the `modal-2b-b64` dataset profile rather than editing the historical block-16 corpus. Derive it byte-for-byte from the verified `20m-2b-dataset-001` directory:

```bash
python -m dataset.reblock \
  --source-dir /local/path/to/20m-2b-dataset-001 \
  --output-dir /local/path/to/modal-2b-b64-dataset-001
```

This is local I/O only: no source redownload, retokenization, packing, split assignment, or mixture scheduling. The output is reverified and contains 64-sequence prepared blocks with a new dataset identity.

Upload the complete derived directory once:

```bash
modal volume put small-llm-data \
  /local/path/to/modal-2b-b64-dataset-001 \
  /datasets/modal-2b-b64-dataset-001
```

Keep the original `20m-2b-dataset-001` unchanged for Kaggle/reproduction.

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

Every 250 successful optimizer updates the trainer writes a verified joint checkpoint under `small-llm-runs/<run-id>/checkpoints/`; the final update is checkpointed as well. The Modal run Volume is the canonical durable checkpoint transport for this trajectory. Legacy Hugging Face dataset-keyed checkpoint publication remains disabled to avoid cross-model namespace collisions.

W&B runs online in project `Small-LLM` with stable run ID `100m-2b-data-001`. Resumes use the same W&B identity and `must` resume semantics after a local durable checkpoint exists.

## Resume

The run Volume stores `step-XXXXXXXX` joint checkpoints. On every fresh Modal container the launcher verifies candidate `local_manifest.json` files and chooses the newest checkpoint whose block cursor agrees with its step number. The existing trainer then restores model, optimizer, WSD scheduler, FP16 scaler, RNG, counters, and data cursor.

Modal automatic retries use this same path. Manual recovery is the identical launch command from the frozen source commit. Do not launch one run identity concurrently on Kaggle and Modal.

## Hardware migration policy

The historical FLA acceptance is T4/SM75 evidence. H100/H200 uses a different target architecture, so the first bounded probe is required even though recurrence semantics are unchanged. Keep FP16 for the first platform migration. Treat BF16 or Blackwell as separate follow-up qualifications rather than combining them with this run.

## Adding a future model

Add the accepted geometry to the shared model/trainer preset surface, then register its nominal parameter label in `modal/profiles.py::MODEL_PRESETS`. Token budgets map to finite dataset profiles independently of model size. The block-64 profile is specifically authorized by ADR 0041 for the new Modal 100M / 2B trajectory; later batch changes require their own qualification/decision.
