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
microbatch qualification: 8, 12, 16; fastest safe result freezes
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
  --gpu RTX5090
```

Do not pass `--max-steps-this-session`. The wrapper uses only the external VPS
dataset feed, then performs CPU import, checkpoint-aligned staging, and
fresh-container visibility gates before allocating the GPU. The first GPU
allocation probes real forward/backward execution at microbatch 8, 12, and 16
and continues directly with the fastest safe result.

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

Rerun the same uncapped command from the exact source commit recorded by the
latest checkpoint. CPU staging realigns to the next unconsumed block before a
new GPU allocation. Never change the microbatch, precision, dataset identity,
or run ID on resume.

