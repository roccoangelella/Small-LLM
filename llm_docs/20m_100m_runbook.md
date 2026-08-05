# 20M Model / 100M-Token Experiment Runbook

_Last updated: 2026-08-05 15:24 Europe/Rome_

This runbook builds the fixed 100M-token finite dataset on the VPS, attaches the completed immutable shards to Kaggle, and repeatedly invokes one pinned Kaggle entry point until the exact one-pass training plan is complete.

## Part A — Produce the 100M dataset on the VPS

### 1. Prepare a clean checkout

```bash
cd /path/to/Small-LLM
git switch main
git pull --ff-only
git status --short
git rev-parse HEAD
```

The working tree must be clean.

### 2. Prepare the existing Drive credentials

The ignored `.env` file must contain the already-qualified values:

```env
SMALL_LLM_GOOGLE_OAUTH_TOKEN=.secrets/google-drive-authorized-user.json
SMALL_LLM_DRIVE_FOLDER_ID=<existing-dataset-shards-folder-id>
```

### 3. Select new output and evidence directories

```bash
export SMALL_LLM_REPO=/path/to/Small-LLM
export WEIGHTS_FILE=/data/climbmix-mixture-calibration/climbmix_code_free_weights.json
export DATASET_DIR=/data/small-llm-20m-100m-dataset-001
export OPS_DIR=/data/small-llm-20m-100m-ops
mkdir -p "$OPS_DIR/logs"
```

`DATASET_DIR` must be new for the first invocation. Do not reuse or mutate the 10M qualification directory.

### 4. Run the fixed fail-closed producer

```bash
cd "$SMALL_LLM_REPO"
set -o pipefail
uv run \
  --env-file .env \
  --with-requirements dataset/requirements-remote.txt \
  python -m dataset.qualification_100m \
  --weights-file "$WEIGHTS_FILE" \
  --output-dir "$DATASET_DIR" \
  2>&1 | tee "$OPS_DIR/logs/dataset-build.log"
status=${PIPESTATUS[0]}
echo "$status" > "$OPS_DIR/dataset-build.exit-code"
test "$status" -eq 0
```

The wrapper fixes and refuses overrides for:

```text
run ID: 20m-100m-dataset-001
accepted-source-token target: 100,000,000
minimum: 90,000,000
hard maximum: 110,000,000
context length: 2,048
sequences per block: 16
target shard size: 8 MiB
producer durable checkpoint cadence: 20,000,000 source tokens
remote durability: required
```

Resume an interrupted producer with the same command plus `--resume`.

### 5. Run a literal full scan

```bash
set -o pipefail
uv run \
  --with-requirements dataset/requirements-remote.txt \
  python -m dataset.main verify \
  --output-dir "$DATASET_DIR" \
  --full-scan \
  2>&1 | tee "$OPS_DIR/logs/dataset-verify.log"
status=${PIPESTATUS[0]}
echo "$status" > "$OPS_DIR/dataset-verify.exit-code"
test "$status" -eq 0
```

### 6. Derive and bind the exact one-pass trainer plan

```bash
set -o pipefail
uv run python -m dataset.qualification_100m_report \
  --dataset-dir "$DATASET_DIR" \
  --drive-manifest "$DATASET_DIR/drive_manifest.json" \
  --output "$DATASET_DIR/qualification_plan.json" \
  2>&1 | tee "$OPS_DIR/logs/qualification-plan.log"
status=${PIPESTATUS[0]}
echo "$status" > "$OPS_DIR/qualification-plan.exit-code"
test "$status" -eq 0
```

Record at least:

```bash
sha256sum \
  "$DATASET_DIR/manifest.json" \
  "$DATASET_DIR/drive_manifest.json" \
  "$DATASET_DIR/qualification_plan.json" \
  | tee "$OPS_DIR/dataset-identities.sha256"
```

## Part B — Publish the completed directory as a private Kaggle Dataset

Create a private Kaggle Dataset containing the complete `DATASET_DIR` without changing names, bytes, or layout. It must contain at least:

```text
manifest.json
drive_manifest.json
qualification_plan.json
train/
validation/
```

Training uses these attached local shards. It does not download the training dataset from Drive or Hugging Face during optimizer steps.

## Part C — Configure Kaggle

Notebook settings:

```text
Accelerator: NVIDIA T4
Internet: On
Attached input: private 20m-100m-dataset-001 dataset
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

The checkpoint repository may be the same private Hugging Face repository used previously because checkpoint objects are namespaced under the new dataset run ID.

## Part D — Run the single pinned Kaggle entry point

Clone or update the repository using the existing private-repository procedure, then run:

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

### First invocation

The launcher:

1. requires a T4;
2. finds exactly one attached dataset matching the fixed profile;
3. performs a full scan and regenerates the exact plan;
4. confirms no remote checkpoint already exists;
5. runs the microbatch-1 versus microbatch-4 qualification;
6. starts from seed 17 with microbatch 4 only if every gate passes;
7. executes at most 749 optimizer updates;
8. validates, saves, and publishes an explicit final segment checkpoint.

### Later invocations

Run the identical command again:

```bash
cd /kaggle/working/Small-LLM
git switch main
git pull --ff-only
python kaggle/run_20m_100m.py
```

The launcher restores the private remote `latest` checkpoint, verifies its complete tree and embedded Drive-manifest identity, resumes the same W&B run, and continues from the next unconsumed block. It never repeats the microbatch probe after a verified training checkpoint exists.

Continue until the summary reports:

```text
status: completed
remaining_steps: 0
```

## Evidence locations

Per-session evidence:

```text
/kaggle/working/small-llm-20m-100m-data-scaling/evidence-<UTC timestamp>/
```

Latest summary:

```text
/kaggle/working/small_llm_20m_100m_data_scaling_summary.json
```

W&B run identity:

```text
project: Small-LLM
run ID: 20m-100m-data-001
```

Remote checkpoint namespace:

```text
run/20m-100m-dataset-001/
```

## Stop conditions

Do not proceed if any of the following occurs:

- the producer or full scan exits nonzero;
- the local and Drive manifests disagree;
- Kaggle finds zero or multiple matching datasets;
- microbatch 4 fails throughput, numerical, overflow, or memory gates;
- a remote checkpoint disagrees with the attached dataset identity;
- the checkpoint step and consumed-block cursor disagree;
- W&B refuses exact resume;
- a segment exits without an explicit final verified remote publication.

A failed microbatch gate is evidence to review, not permission to silently alter the experiment.