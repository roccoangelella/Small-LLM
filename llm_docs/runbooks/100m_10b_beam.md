# 100M / 10B Beam runbook

_Last reviewed: 2026-08-14_

This is the active operator procedure for the ADR-0071 full fresh trajectory.
HF remains the authoritative dataset and checkpoint backend; the Beam cache
Volume supplies preseeded dataset bytes to avoid paid GPU downloads.

## Frozen identity

```text
training run ID: 100m-10b-data-001
dataset run ID: modal-10b-b64-dataset-001
GPU: one Beam serverless RTX5090
updates: 76,294
target tokens: 10,000,007,168
WSD: 3,815 warmup / 57,220 stable / 15,259 decay
microbatch: 4 (live-qualified; 8 exceeded RTX5090 VRAM)
precision: fp16 autocast with FP32 master parameters
```

The complete corpus is already verified in HF and mirrored to
`beam://small-llm-cache/datasets/modal-10b-b64-dataset-001`. Do not restart
`beam/vps_dataset_producer.py` and do not use the paid producer path in
`beam/launch.py`.

## Preflight

The checkout must be clean and the locked environment installed:

```bash
uv sync --locked
uv run python beam/vps_train.py \
  --model 100M \
  --tokens 10B \
  --gpu RTX5090 \
  --dry-run
```

The payload must name `100m-10b-data-001`, `hf_rolling_shards`, RTX5090,
auto-microbatch candidates `8,12,16`, and `remaining plan`. Beam secrets
`WANDB_API_KEY`, `HF_TOKEN`, and `SMALL_LLM_HF_REPO_ID` must already exist.

## Full launch

Run from the repository root in a persistent tmux session:

```bash
uv run python beam/vps_train.py \
  --model 100M \
  --tokens 10B \
  --gpu RTX5090 \
  --microbatch-size 4
```

Do not pass `--max-steps-this-session`. The wrapper uses only the external VPS
dataset feed, then performs CPU import, checkpoint-aligned staging, and
fresh-container visibility gates before allocating the GPU. The requested
microbatch 4 still runs the four-update finite-loss, gradient, memory, and
throughput qualification before continuing directly into production. Do not
return to auto 8/12/16 on this device: live microbatch 8 exceeded VRAM.

## Concurrent approximately-5B Kaggle evaluation

Training does not stop for this evaluation. Remote live checkpoints publish
every 500 successful updates with rolling latest-only retention. Capture:

```text
checkpoint: step-00038000
consumed targets: 4,980,736,000
```

Start the Kaggle copy/evaluation as soon as the HF latest pointer names that
checkpoint, before `step-00038500` replaces it. Record the exact checkpoint,
source commit, dataset identity, and decoding/eval manifests in the resulting
evidence. This is an intermediate stable-phase checkpoint, not a terminal 5B
WSD model, and its result does not pause or terminate the Beam trajectory.

## Resume

Rerun the same uncapped microbatch-4 command from active source commit
`1f9dff920ecc45ce2fdb43fd875514a18391273d` or the exact source commit recorded by a later checkpoint. Commit
`42b0376` is accepted only as the one-time infrastructure-migration parent for
the verified step-250 checkpoint. CPU staging realigns to the next unconsumed
block before a new GPU allocation. Never change the microbatch, precision,
dataset identity, or run ID on resume.

Beam sets `SMALL_LLM_CHECKPOINT_FSYNC=0`. Checkpoint files and manifests still
use staging plus atomic rename and are hash-verified on restore, but the runtime
does not issue local-disk power-loss barriers against Beam's distributed
Volume. Do not remove this provider-specific setting: the first step-250 save
completed its final rename and then blocked indefinitely on the parent
directory `fsync`.
