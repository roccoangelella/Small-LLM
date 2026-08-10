# 20M Model / 2B-Token Data-Scaling Runbook

_Last updated: 2026-08-10 Europe/Rome_

This run is a fresh seed-17 trajectory for the approximately-20M-parameter GDN-2 hybrid on a separately identified approximately-2B-token finite dataset. It does **not** continue the completed 500M checkpoint or the abandoned 1B setup. CUDA training uses the qualified mixed FLA GDN-2 backend from update 1.

## 1. Data transport decision

Do **not** stream Nemotron-ClimbMix from Hugging Face during the Kaggle GPU training job.

Build the complete deterministic finite dataset on the VPS, preserve its Google Drive durability mirror, privately publish it to Kaggle, verify a round-trip download, then attach that exact private dataset to the Kaggle notebook.

This keeps network/source ingestion off the GPU critical path and preserves the existing hashed manifest, exact block schedule, deterministic cursor, and fail-closed checkpoint/resume identity.

The payload remains modest at this scale: uint16 storage is two bytes per token ID, so two billion stored IDs are roughly 4 GB before validation/EOD/manifest overhead.

## 2. Fixed 2B dataset contract

```text
profile: 20m-2b-data-scaling-v1
run ID: 20m-2b-dataset-001
accepted-source-token target: 2,000,000,000
minimum: 1,800,000,000
hard maximum: 2,200,000,000
producer durable checkpoint cadence: 80,000,000 source tokens
context length: 2,048
sequences per optimizer block: 16
target shard size: 8 MiB
remote durability: required
source revision: 5eaa64b9c0c85b7f56af01d7dffdb0795816b12b
tokenizer: existing GPT-2 token IDs, reused verbatim
cluster policy: unchanged; explicit programming cluster 11 excluded
```

At 20,637,592 learned parameters, the nominal 2B point is approximately 96.9 accepted source tokens per parameter.

The exact train target tokens, optimizer-update count, and WSD phase token boundaries are derived from the completed verified manifest. Do not hard-code a nominal step count. A rough expectation is approximately 61.0k optimizer updates at 32,768 target tokens per full block.

## 3. VPS environment

Copy relevant values from the existing `.env`; the profile-specific template is:

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

Do not use `--allow-unsafe-low-disk` for this build. Let the production disk preflight decide whether the VPS has adequate capacity.

## 4. Build, verify, and privately publish from the VPS

```bash
cd /path/to/Small-LLM
git switch main
git pull --ff-only
bash kaggle/build_and_push_2b.sh
```

The command is resumable. After a VPS or network interruption, rerun the exact same command without deleting the production directory; the suite selects the production `--resume` path automatically.

The publication suite must complete all inherited gates: fixed production identity, deterministic production/resume, full local verification, exact 2B qualification-plan derivation, Google Drive durability agreement, private Kaggle publication, fresh Kaggle round-trip download, byte-identical tree verification excluding Kaggle transport archives, and denied anonymous access.

Successful publication requires:

```text
/data/small-llm/20m-2b-ops/build-and-push-summary.json
  status: completed or already_published

/data/small-llm/20m-2b-ops/kaggle-publish-state.json
  status: verified
```

Do not start GPU training before those gates pass.

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

Do not attach the 100M, 500M, or abandoned 1B dataset alongside the intended 2B dataset if that creates multiple matching or ambiguous manifests.

## 6. Run or resume the 2B training

```bash
cd /kaggle/working/Small-LLM
git switch main
git pull --ff-only
python kaggle/run_20m_2b.py
```

Do not pass `--launch-commit`. The wrapper pins an immutable launch worktree containing the 2B qualification modules and the T4-qualified `fla-core==0.5.2` mixed GDN-2 implementation.

### Fixed training identities/defaults

```text
model parameters: 20,637,592
architecture: gdn2_hybrid
fresh initialization seed: 17
context: 2,048
precision: FP16 autocast, FP32 master parameters
optimizer: hybrid Muon + AdamW
learning rate: 3e-4
weight decay: 0.1
schedule: one-pass WSD derived from the 2B qualification plan
training microbatch: 4
saved/configured GDN chunk: 32
CUDA GDN-2 backend: mixed FLA, internal chunk 64
local checkpoint cadence: 250 updates
held-out validation cadence: 250 updates
verified remote publication cadence: 250 updates
W&B run ID: 20m-2b-data-001
W&B run name: 20M model on 2B tokens
repository default session cap: none within the finite plan
```

This is a **fresh** run. The first invocation must not initialize from the 500M checkpoint. If the 2B checkpoint namespace already contains a verified checkpoint from a prior interrupted 2B session, normal exact resume is expected.

### Resume behavior

On later invocations, the launcher checks only the `20m-2b-dataset-001` remote checkpoint namespace. A verified checkpoint restores model weights, optimizer state, WSD scheduler position, FP16 scaler, RNG state, data/block cursor, and consumed-token cursor. The restored checkpoint's Drive manifest must match the attached 2B dataset.

At every 250-successful-update boundary require held-out validation, a local checkpoint, and verified remote checkpoint publication.

For a deliberate bounded diagnostic only:

```bash
python kaggle/run_20m_2b.py --max-steps-this-session 250
```

Do not use that override for normal training.

Completion requires:

```text
status: completed
remaining_steps: 0
```

## 7. Final evaluation

After completion, run the same frozen post-pretraining evaluation bundle used for earlier 20M checkpoints. Preserve at minimum final held-out loss/perplexity, `eval_core_v1` fast/full results, teacher-forced confidence/rank diagnostics, fixed free-generation prompt outputs and degeneration metrics, training loss-versus-token curve, throughput/overflow/memory telemetry, and exact dataset/manifest/checkpoint identities.

Compare the 100M, 500M, and 2B points directly. The abandoned 1B profile has no experimental result and must not be treated as a datapoint.

## Stop conditions

Do not proceed if source/dataset identity changes, local and Drive manifests disagree, Kaggle finds zero or multiple matching 2B datasets, the trainer does not use microbatch 4, `fla-core==0.5.2` cannot be resolved/imported on CUDA, checkpoint strict restore fails, a restored checkpoint disagrees with the attached 2B manifest, W&B does not use `20m-2b-data-001`, numerical/validation/memory gates fail, a 250-update durability boundary is incomplete, or the trainer attempts to consume beyond the exact finite one-pass plan.

A failed gate is evidence to inspect, not permission to silently alter the experiment.
