# 20M Model / 2B-Token Data-Scaling Runbook

_Last updated: 2026-08-10 Europe/Rome_

This is the active runbook for the fresh seed-17 approximately-20M-parameter GDN-2 hybrid on the separately identified approximately-2B-token finite dataset. It does **not** continue the completed 500M checkpoint or the abandoned 1B setup. CUDA training uses the qualified mixed FLA GDN-2 backend from update 1.

The canonical command surface is [`unified_kaggle_launcher.md`](unified_kaggle_launcher.md). Do not invent a separate publication, fresh-start, or resume command for this profile.

## 1. Fixed experiment identity

```text
profile: 20m-2b-data-scaling-v1
dataset run ID: 20m-2b-dataset-001
W&B run ID: 20m-2b-data-001
accepted-source-token target: 2,000,000,000
minimum: 1,800,000,000
hard maximum: 2,200,000,000
producer durable checkpoint cadence: 80,000,000 source tokens
context length: 2,048
sequences per optimizer block: 16
target shard size: 8 MiB
source revision: 5eaa64b9c0c85b7f56af01d7dffdb0795816b12b
tokenizer: existing GPT-2 token IDs
programming cluster 11: excluded
fresh initialization seed: 17
training microbatch: 4
```

At 20,637,592 learned parameters, the nominal point is approximately 96.9 accepted source tokens per parameter.

The exact train target-token count, optimizer-update count, and WSD phase boundaries come from the completed verified manifest. Do not hard-code a nominal step count.

## 2. Data transport

Do **not** stream Nemotron-ClimbMix during the Kaggle GPU job.

Use:

```text
pinned source
    -> VPS deterministic finite build
    -> local immutable uint16 shards
    -> verified Google Drive durability mirror
    -> private Kaggle publication
    -> fresh round-trip byte verification
    -> exact attached Kaggle dataset
    -> GPU training from Kaggle-local input
```

Do not start training before local verification, Drive agreement, private Kaggle publication, round-trip verification, and denied anonymous access all pass.

## 3. VPS environment

Profile template:

```text
kaggle/2b-publish.env.example
```

Required values:

```env
KAGGLE_API_TOKEN=<token from Kaggle settings>
KAGGLE_USERNAME=<your Kaggle owner slug>
SMALL_LLM_GOOGLE_OAUTH_TOKEN=.secrets/google-drive-authorized-user.json
SMALL_LLM_DRIVE_FOLDER_ID=<existing qualified Drive parent folder ID>
```

Instead of `KAGGLE_USERNAME`, an exact handle may be set:

```env
SMALL_LLM_2B_KAGGLE_DATASET_HANDLE=owner/small-llm-20m-2b-dataset-001
```

Optional path overrides:

```env
SMALL_LLM_2B_WEIGHTS_FILE=/data/climbmix-mixture-calibration/climbmix_code_free_weights.json
SMALL_LLM_2B_DATASET_DIR=/data/small-llm/20m-2b-dataset-001
SMALL_LLM_2B_OPS_DIR=/data/small-llm/20m-2b-ops
SMALL_LLM_KAGGLE_READY_TIMEOUT_SECONDS=900
```

Default paths:

```text
mixture weights: /data/climbmix-mixture-calibration/climbmix_code_free_weights.json
producer output: /data/small-llm/20m-2b-dataset-001
operations/evidence: /data/small-llm/20m-2b-ops
```

Do not bypass the production disk preflight.

## 4. Build, verify, and privately publish

From the VPS:

```bash
cd /path/to/Small-LLM
git switch main
git pull --ff-only
python kaggle/launch.py publish --model 20M --tokens 2B
```

After a VPS or network interruption, rerun the **identical command**. Publication resume is automatic; do not pass `--resume`.

The publication path must preserve the complete profile-specific gate sequence:

```text
fixed production identity
deterministic build/resume
full local verification
exact 2B qualification-plan derivation
Google Drive durability agreement
private Kaggle publication
fresh Kaggle round-trip download
byte-identical tree verification
denied anonymous access
```

Successful publication requires:

```text
/data/small-llm/20m-2b-ops/build-and-push-summary.json
  status: completed or already_published

/data/small-llm/20m-2b-ops/kaggle-publish-state.json
  status: verified
```

## 5. Configure the Kaggle notebook

```text
Accelerator: NVIDIA T4
Internet: On
Attached input: the exact private small-llm-20m-2b-dataset-001 version that passed VPS round-trip verification
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

Do not attach another matching scaling dataset if it could make dataset discovery ambiguous.

## 6. Launch or resume training

From the Kaggle clone:

```bash
cd /kaggle/working/Small-LLM
git switch main
git pull --ff-only
python kaggle/launch.py train --model 20M --tokens 2B
```

Resume is automatic and fail-closed. Rerun the exact same command after interruption. The selected backend checks only the `20m-2b-dataset-001` checkpoint namespace and requires the restored checkpoint's Drive manifest to match the attached 2B dataset.

This profile is fresh with respect to earlier scaling runs. If no verified 2B checkpoint exists, training starts from seed 17. It must never initialize from the completed 500M checkpoint.

Fixed training contract:

```text
model parameters: 20,637,592
architecture: gdn2_hybrid
context: 2,048
precision: FP16 autocast with FP32 master parameters
optimizer: hybrid Muon + AdamW
learning rate: 3e-4
weight decay: 0.1
schedule: one-pass WSD from the verified 2B qualification plan
training microbatch: 4
saved/configured GDN chunk: 32
CUDA GDN-2 backend: mixed FLA
FLA internal chunk: 64
local checkpoint cadence: 250 successful updates
held-out validation cadence: 250 successful updates
verified remote publication cadence: 250 successful updates
W&B run ID: 20m-2b-data-001
W&B run name: 20M model on 2B tokens
normal session cap: none within the finite plan
```

A verified resume restores:

```text
model weights
optimizer state
WSD scheduler position
FP16 scaler
RNG state
data/block cursor
consumed-token cursor
```

For a deliberate bounded diagnostic only:

```bash
python kaggle/launch.py train --model 20M --tokens 2B --max-steps-this-session 250
```

Do not use that override for normal training.

Completion requires:

```text
status: completed
remaining_steps: 0
```

## 7. Inspect command resolution without executing

Before a VPS or Kaggle operation, the profile can be resolved without importing the backend:

```bash
python kaggle/launch.py publish --model 20M --tokens 2B --dry-run
python kaggle/launch.py train --model 20M --tokens 2B --dry-run
```

The result must resolve to the registered 20M/2B publisher and training backend.

## 8. Final evaluation

After completion, run the same frozen post-pretraining evaluation bundle used for the earlier 20M checkpoints. Preserve at minimum:

```text
final held-out loss/perplexity
eval_core_v1 fast/full results
teacher-forced confidence/rank diagnostics
fixed free-generation prompt outputs and degeneration metrics
training loss-versus-token curve
throughput/overflow/memory telemetry
exact dataset/manifest/checkpoint identities
```

Compare the 100M, 500M, and 2B points directly. The abandoned 1B profile has no experimental result and is not a datapoint.

## Stop conditions

Stop rather than silently changing the experiment if:

- source or dataset identity changes;
- local and Drive manifests disagree;
- publication or round-trip verification fails;
- Kaggle finds zero or multiple matching 2B datasets;
- the trainer does not use microbatch 4;
- `fla-core==0.5.2` cannot be resolved/imported on CUDA;
- strict checkpoint restore fails;
- a restored checkpoint disagrees with the attached 2B manifest;
- W&B does not use `20m-2b-data-001`;
- numerical, validation, memory, or durability gates fail;
- a 250-update durability boundary is incomplete;
- the trainer attempts to consume beyond the exact finite one-pass plan.

A failed gate is evidence to inspect, not permission to mutate the frozen profile.
