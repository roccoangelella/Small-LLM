# 20M Model / 100M-Token Experiment Runbook

_Last updated: 2026-08-05 15:34 Europe/Rome_

This runbook uses one VPS command to build, verify, and privately publish the fixed 100M-token dataset, followed by one repeated Kaggle command for segmented exact training.

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

## Part C — Configure each Kaggle account

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

Every account must attach the same private Kaggle Dataset version. Training reads the attached immutable shards locally; it does not stream the dataset from Drive or Hugging Face.

## Part D — Run or resume training

```bash
cd /kaggle/working/Small-LLM
git switch main
git pull --ff-only
python kaggle/run_20m_100m.py
```

The wrapper pins the evidence-producing worktree to:

```text
43190cb72443a2de290dc8e6f2c54f29d8dff501
```

Do not pass `--launch-commit`.

The first invocation performs the microbatch-1 versus microbatch-4 T4 gate and starts from seed 17 only if microbatch 4 passes. Each invocation executes at most 749 additional optimizer updates and requires an explicit final remote checkpoint.

After a successful segment, run the identical command on the next Kaggle account. It restores the verified private Hugging Face checkpoint, checks the attached Drive-manifest identity, and resumes the same model, optimizer, scheduler, scaler, RNG, data cursor, and W&B run.

Continue until:

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

## Stop conditions

Do not proceed if:

- dataset production, local verification, upload, round-trip verification, or privacy verification fails;
- the local and Drive manifests disagree;
- the uploaded tree differs from the staged tree;
- Kaggle training finds zero or multiple matching datasets;
- microbatch 4 fails throughput, numerical, overflow, or memory gates;
- a restored checkpoint disagrees with the attached dataset;
- W&B refuses exact resume;
- a segment exits without a verified final remote publication.

A failed gate is evidence to review, not permission to silently change the experiment.
