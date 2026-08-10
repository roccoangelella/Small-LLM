# 20M Model / 100M-Token Experiment Runbook

_Last updated: 2026-08-10 Europe/Rome_

The 20M/100M experiment is completed historical evidence. This runbook records its fixed identities and the **current reproduction command surface**. Separate 100M launcher/publisher wrappers have been removed; use `kaggle/launch.py`.

## Fixed identity

```text
profile: 20m-100m-data-scaling-v1
dataset run ID: 20m-100m-dataset-001
W&B run ID: 20m-100m-data-004
launch commit: 8e3cd9cb149facc5fa28e8108a70304c1f8c1c15
model parameters: 20,637,592
fresh seed: 17
context: 2,048
training microbatch: 4
local checkpoint cadence: 250 updates
held-out validation cadence: 250 updates
verified remote publication cadence: 250 updates
```

Dataset production identity:

```text
target accepted source tokens: 100,000,000
minimum: 90,000,000
maximum: 110,000,000
producer durable checkpoint cadence: 20,000,000 source tokens
context length: 2,048
sequences per optimizer block: 16
target shard size: 8 MiB
remote durability: required
```

## VPS environment

Use the ignored repository `.env`; reference values remain in:

```text
kaggle/100m-publish.env.example
```

Required publication credentials include `KAGGLE_API_TOKEN`, Google Drive authorization, and the Drive folder ID. `KAGGLE_USERNAME` may derive the handle, or set:

```env
SMALL_LLM_KAGGLE_DATASET_HANDLE=owner/small-llm-20m-100m-dataset-001
```

The unified Python launcher replaces the removed shell wrapper and automatically re-executes publication through `uv` with Python 3.13, `.env`, and `kaggle/requirements-100m-publish.txt`.

## Build, verify, and privately publish

```bash
cd /path/to/Small-LLM
git switch main
git pull --ff-only
python kaggle/launch.py publish --model 20M --tokens 100M
```

The path preserves deterministic build/resume, full local verification, exact qualification-plan derivation, private Kaggle publication, fresh round-trip download, byte-identical verification, and denied anonymous access.

Rerun the identical command after interruption. Publication resume is automatic; do not pass `--resume`.

Successful publication evidence remains:

```text
/data/small-llm/20m-100m-ops/build-and-push-summary.json
/data/small-llm/20m-100m-ops/kaggle-publish-state.json
```

## Kaggle training / exact resume

Configure an NVIDIA T4 notebook with Internet enabled, attach the exact verified private 100M dataset, and provide:

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
python kaggle/launch.py train --model 20M --tokens 100M
```

The profile preserves the original 100M fresh-start behavior: on a fresh namespace the microbatch-1 versus microbatch-4 T4 gate runs before real training. Verified resume restores the exact model, optimizer, WSD scheduler, FP16 scaler, RNG state, data/block cursor, and consumed-token cursor.

For a deliberate bounded diagnostic only:

```bash
python kaggle/launch.py train --model 20M --tokens 100M --max-steps-this-session 250
```

## Canonical completed result

```text
optimizer updates: 3,053
consumed training target tokens: 100,018,176
final validation loss: 4.252758495143203
final validation perplexity: 70.29906475797992
final checkpoint: step-00003053
```

The late-run throughput collapse belongs to the historical adaptive PyTorch GDN-2 execution path; it motivated the later FLA backend qualification. Reproduction through the current command surface still uses the frozen launch commit recorded above for this historical profile.
