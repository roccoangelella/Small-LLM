# 20M Model / 100M-Token Experiment Runbook

_Last updated: 2026-08-06 10:25 Europe/Rome_

This runbook uses one VPS command to build, verify, and privately publish the fixed 100M-token dataset, followed by one Kaggle command that attempts the complete remaining finite one-pass schedule. Verified remote checkpoints preserve exact resume when Kaggle interrupts the invocation.

## Part A — Configure the VPS once

From the Small-LLM repository, copy the template values into the ignored `.env` file:

```env
KAGGLE_API_TOKEN=<token from Kaggle settings>
KAGGLE_USERNAME=<your Kaggle owner slug>
SMALL_LLM_GOOGLE_OAUTH_TOKEN=.secrets/google-drive-authorized-user.json
SMALL_LLM_DRIVE_FOLDER_ID=<existing qualified Drive folder ID>
```

Instead of `KAGGLE_USERNAME`, an exact handle may be used:

```env
SMALL_LLM_KAGGLE_DATASET_HANDLE=owner/small-llm-20m-100m-dataset-001
```

Reference template:

```text
kaggle/100m-publish.env.example
```

Default paths:

```text
mixture weights: /data/climbmix-mixture-calibration/climbmix_code_free_weights.json
producer output: /data/small-llm/20m-100m-dataset-001
operations/evidence: /data/small-llm/20m-100m-ops
```

Optional path overrides are documented in the environment template.

## Part B — Build and privately publish with one command

```bash
cd /path/to/Small-LLM
git switch main
git pull --ff-only
bash kaggle/build_and_push_100m.sh
```

The command:

1. loads `.env` without printing secrets;
2. pins Python 3.13 and `kagglehub==1.0.2`;
3. starts `dataset.qualification_100m` when no producer output exists;
4. automatically resumes an interrupted producer directory;
5. skips production when the fixed completed manifest already exists;
6. runs a literal full local shard scan;
7. derives and verifies `qualification_plan.json`;
8. stages exactly:
   - `manifest.json`
   - `drive_manifest.json`
   - `qualification_plan.json`
   - `train/`
   - `validation/`
9. refuses a Kaggle handle already readable anonymously;
10. uploads with `kagglehub.dataset_upload` as a private dataset;
11. downloads the complete Kaggle dataset back to the VPS;
12. requires a byte-identical tree, another full scan, and denied anonymous access;
13. records a verified receipt and avoids duplicate versions on identical reruns.

The producer remains fixed at:

```text
run ID: 20m-100m-dataset-001
accepted-source-token target: 100,000,000
minimum: 90,000,000
hard maximum: 110,000,000
context length: 2,048
sequences per optimizer block: 16
target shard size: 8 MiB
producer durable checkpoint cadence: 20,000,000 source tokens
remote durability: required
```

Successful publication requires:

```text
/data/small-llm/20m-100m-ops/build-and-push-summary.json
  status: completed or already_published

/data/small-llm/20m-100m-ops/kaggle-publish-state.json
  status: verified
```

Rerun the same command after an interruption. The producer and publisher are idempotent. Use `--force-upload` only when an intentional new Kaggle version is required.

## Part C — Configure the Kaggle notebook

Notebook settings:

```text
Accelerator: NVIDIA T4
Internet: On
Attached input: the private small-llm-20m-100m-dataset-001 dataset
```

Required Kaggle secrets:

```text
GITHUB_TOKEN
WANDB_API_KEY
HF_TOKEN
SMALL_LLM_HF_REPO_ID
```

Optional:

```text
WANDB_ENTITY
```

Attach the exact private Kaggle Dataset version used by the qualification plan. Training reads immutable local shards; it does not stream the dataset from Drive or Hugging Face.

## Part D — Run or resume training

```bash
cd /kaggle/working/Small-LLM
git switch main
git pull --ff-only
python kaggle/run_20m_100m.py
```

Do not pass `--launch-commit`. The wrapper pins a verified worktree commit containing the bounded-validation hotfix.

Operational defaults:

```text
training microbatch: 4 sequences
validation microbatch: 1 sequence
local checkpoint cadence: 250 updates
held-out validation cadence: 250 updates
verified remote publication cadence: 250 updates
W&B run ID: 20m-100m-data-004
allocator safeguard: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
repository default session cap: none within the finite qualification plan
```

The first fresh invocation performs the microbatch-1 versus microbatch-4 T4 training gate and starts from seed 17 only if microbatch 4 passes. The launcher then requests every remaining update in the exact finite one-pass plan rather than stopping at update 749.

Validation is independent of the training microbatch. It runs under `torch.inference_mode()` one sequence at a time, releases optimizer gradients before evaluation, and clears unused CUDA allocator cache before and after evaluation. Do not increase the validation microbatch until a controlled T4 memory probe demonstrates safe headroom.

At every 250-update boundary, the trainer runs held-out validation, writes the local joint checkpoint, and publishes a verified remote checkpoint. If Kaggle interrupts the run, rerun the identical command. It restores the latest verified private Hugging Face checkpoint, checks the attached Drive-manifest identity, and resumes the same model, optimizer, scheduler, scaler, RNG, data cursor, and finite schedule.

The optional underlying flag remains available for an intentionally bounded diagnostic:

```bash
python kaggle/run_20m_100m.py --max-steps-this-session 250
```

Do not use that override for the normal corrected run.

Completion requires:

```text
status: completed
remaining_steps: 0
```

## Evidence locations

VPS dataset publication:

```text
/data/small-llm/20m-100m-ops/logs/
/data/small-llm/20m-100m-ops/kaggle-dataset/
/data/small-llm/20m-100m-ops/kaggle-roundtrip/
/data/small-llm/20m-100m-ops/kaggle-publish-state.json
/data/small-llm/20m-100m-ops/build-and-push-summary.json
```

Kaggle training:

```text
/kaggle/working/small-llm-20m-100m-data-scaling/evidence-<UTC timestamp>/
/kaggle/working/small_llm_20m_100m_data_scaling_summary.json
```

Incident record:

```text
llm_docs/evidence/20m_100m/validation_oom_step_500_2026-08-06.md
```

## First corrected-run checks

At update 250 require:

```text
validation completed without CUDA OOM
local checkpoint: step-00000250
verified remote publication: step-00000250
```

At update 500 require the same three events for `step-00000500`. Passing update 749 without a launcher exit confirms that the old artificial session stop is no longer active.

## Stop conditions

Do not proceed if:

- dataset production, local verification, upload, round-trip verification, or privacy verification fails;
- the local and Drive manifests disagree;
- the uploaded tree differs from the staged tree;
- Kaggle training finds zero or multiple matching datasets;
- microbatch 4 fails throughput, numerical, overflow, or memory gates;
- validation still approaches unsafe T4 memory or emits a CUDA OOM;
- a restored checkpoint disagrees with the attached dataset;
- W&B refuses the configured run identity;
- a scheduled boundary exits without a verified remote publication;
- the trainer attempts to consume beyond the exact finite one-pass plan.

A failed gate is evidence to review, not permission to silently change the experiment.
