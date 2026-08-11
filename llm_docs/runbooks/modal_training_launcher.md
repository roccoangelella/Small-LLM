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

Upload an already-verified finite dataset once, for example:

```bash
modal volume put small-llm-data \
  /local/path/to/20m-2b-dataset-001 \
  /datasets/20m-2b-dataset-001
```

## Dry run

```bash
modal run modal/launch.py --model 100M --tokens 2B --dry-run
```

Dry-run resolution does not rent a GPU.

## Production shape

```bash
modal run --detach modal/launch.py \
  --model 100M \
  --tokens 2B \
  --gpu H100
```

`H100` is the default and permits Modal's H200 automatic upgrade. The default microbatch value is `0`, meaning benchmark 4, 8, and 16 on the first run and freeze the fastest candidate that is finite and stays at or below 90% reserved VRAM.

Use an explicit microbatch only when reproducing a qualified run:

```bash
modal run --detach modal/launch.py \
  --model 100M --tokens 2B --gpu H100 --microbatch-size 16
```

The current optimizer block contains 16 sequences. A larger microbatch cannot increase execution parallelism without changing the prepared-block/optimizer contract.

## First-run gates

The launcher:

1. requires a clean controlling Git checkout and records its exact commit;
2. discovers exactly one matching finite dataset on the read-only data Volume;
3. performs a full schema-v2 verification once per manifest identity;
4. derives the one-pass WSD plan from the dataset manifest;
5. records GPU name, memory, compute capability, PyTorch, and CUDA runtime;
6. uses a compute-capability-specific persistent Triton cache;
7. runs a short real trainer forward/backward microbatch qualification;
8. freezes source commit, model/data identity, precision, and selected microbatch;
9. starts online W&B training with 250-update validation/checkpoint cadence.

## Resume

The run Volume stores `step-XXXXXXXX` joint checkpoints. On every fresh Modal container the launcher verifies candidate `local_manifest.json` files and chooses the newest checkpoint whose block cursor agrees with its step number. The existing trainer then restores model, optimizer, WSD scheduler, FP16 scaler, RNG, counters, and data cursor.

Modal automatic retries use this same path. Manual recovery is the identical launch command from the frozen source commit. Do not launch one run identity concurrently on Kaggle and Modal.

## Hardware migration policy

The historical FLA acceptance is T4/SM75 evidence. H100/H200 uses a different target architecture, so the first bounded probe is required even though recurrence semantics are unchanged. Keep FP16 for the first platform migration. Treat BF16 or Blackwell as separate follow-up qualifications rather than combining them with the provider move.

## Adding a future model

Add the accepted geometry to the shared model/trainer preset surface, then register its nominal parameter label in `modal/profiles.py::MODEL_PRESETS`. Token budgets map to the existing finite dataset profiles independently of model size, so an accepted corpus does not need to be rebuilt merely because parameter count changes.
