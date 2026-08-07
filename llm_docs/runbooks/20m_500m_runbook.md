# 20M Model / 500M-Token Final Probe Runbook

_Last updated: 2026-08-07 10:11 Europe/Rome_

This is the final planned data-scaling probe for the approximately-20M-parameter GDN-2 hybrid. It preserves the proven 100M production and training process while binding a separate approximately-500M-token finite dataset and a fresh seed-17 training run.

The 500M training run must not resume the 100M model checkpoint. It receives its own one-pass WSD schedule derived from the completed 500M manifest. Dataset preparation may run in parallel with the tail of the 100M training run because source access, local paths, Drive run folder, Kaggle dataset handle, W&B identity, and checkpoint namespace are distinct.

## Part A — No whole ClimbMix download is required

Do not download the Nemotron-ClimbMix source corpus locally first.

The production path already implements deterministic source streaming correctly:

- the source repository is pinned to immutable revision `5eaa64b9c0c85b7f56af01d7dffdb0795816b12b`;
- only metadata is listed up front to freeze the exact root source-file work plan;
- source content is fetched through deterministic HTTP byte-range requests;
- complete JSONL record boundaries are recovered around range edges;
- the selected GPT-2 token IDs are materialized into the fixed local binary training cache;
- production state is checkpointed and is crash-safe/resumable;
- finalized shards are mirrored to Google Drive under the run-specific folder `20m-500m-dataset-001`.

This source-range streaming build is safe to start immediately on the VPS. Training itself must still consume the completed finite Kaggle dataset rather than reading a live Hugging Face stream. That separation preserves a hashed manifest, exact one-pass schedule, deterministic data cursor, and exact checkpoint/resume identity.

## Part B — Fixed 500M dataset contract

```text
profile: 20m-500m-data-scaling-v1
run ID: 20m-500m-dataset-001
accepted-source-token target: 500,000,000
minimum: 450,000,000
hard maximum: 550,000,000
producer durable checkpoint cadence: 20,000,000 source tokens
context length: 2,048
sequences per optimizer block: 16
target shard size: 8 MiB
remote durability: required
source revision: 5eaa64b9c0c85b7f56af01d7dffdb0795816b12b
tokenizer: existing GPT-2 token IDs, reused verbatim
cluster policy: unchanged from production corpus policy
```

At 20,637,592 model parameters, the nominal 500M source-token point is approximately 24.2 accepted source tokens per parameter.

The exact number of train target tokens and optimizer updates is not guessed in advance. `qualification_plan.json` derives it from the completed manifest after validation/EOD packing. A rough expectation is about 15.2k optimizer updates at 32,768 target tokens per full block.

## Part C — VPS configuration

Use the existing `.env` credentials. The 500M-specific template is:

```text
kaggle/500m-publish.env.example
```

Required values remain:

```env
KAGGLE_API_TOKEN=<token from Kaggle settings>
KAGGLE_USERNAME=<your Kaggle owner slug>
SMALL_LLM_GOOGLE_OAUTH_TOKEN=.secrets/google-drive-authorized-user.json
SMALL_LLM_DRIVE_FOLDER_ID=<existing qualified Drive parent folder ID>
```

The same Drive parent folder is safe: the Drive backend creates a distinct child folder named by `run_id`.

Instead of `KAGGLE_USERNAME`, a dedicated exact handle may be set:

```env
SMALL_LLM_500M_KAGGLE_DATASET_HANDLE=owner/small-llm-20m-500m-dataset-001
```

Do not reuse `SMALL_LLM_KAGGLE_DATASET_HANDLE` from the 100M publication. The 500M wrapper intentionally ignores that old generic override so it cannot accidentally publish a new version onto the 100M Kaggle dataset.

Optional path overrides:

```env
SMALL_LLM_500M_WEIGHTS_FILE=/data/climbmix-mixture-calibration/climbmix_code_free_weights.json
SMALL_LLM_500M_DATASET_DIR=/data/small-llm/20m-500m-dataset-001
SMALL_LLM_500M_OPS_DIR=/data/small-llm/20m-500m-ops
SMALL_LLM_KAGGLE_READY_TIMEOUT_SECONDS=900
```

Default local paths:

```text
mixture weights: /data/climbmix-mixture-calibration/climbmix_code_free_weights.json
producer output: /data/small-llm/20m-500m-dataset-001
operations/evidence: /data/small-llm/20m-500m-ops
```

Because the final binary dataset is roughly on the order of 1 GiB, leave several GiB free for the production tree, publication staging, and complete Kaggle round-trip copy even though the producer's own disk preflight is smaller.

## Part D — Build and privately publish with one command

```bash
cd /path/to/Small-LLM
git switch main
git pull --ff-only
bash kaggle/build_and_push_500m.sh
```

This command performs the same gates as the 100M publication suite:

1. load `.env` without printing secrets;
2. start `dataset.qualification_500m`, or resume it if production state exists;
3. fetch the pinned source only by deterministic HTTP byte ranges;
4. require remote Drive durability throughout production;
5. perform a literal full local shard scan;
6. derive the exact one-pass `qualification_plan.json` with `dataset.qualification_500m_report`;
7. stage only `manifest.json`, `drive_manifest.json`, `qualification_plan.json`, `train/`, and `validation/`;
8. refuse a publicly readable Kaggle target;
9. upload the fixed dataset privately;
10. download the complete Kaggle dataset back to the VPS;
11. require byte-identical tree identity, another full shard scan, and denied anonymous access;
12. record an idempotent verified publication receipt.

Rerun the same command after a VPS/network interruption. Do not delete the production directory; `--resume` is selected automatically.

Successful publication requires:

```text
/data/small-llm/20m-500m-ops/build-and-push-summary.json
  status: completed or already_published

/data/small-llm/20m-500m-ops/kaggle-publish-state.json
  status: verified
```

## Part E — Finish the 100M checkpoint before starting 500M training

Dataset production may happen now, but do not launch the 500M model training until the current 100M run has:

1. completed its finite schedule;
2. published its final verified checkpoint;
3. run the frozen final validation/evaluation bundle and prompt diagnostics;
4. preserved its result bundle as the 100M comparison point.

The 500M model is a fresh pretraining run. It is not initialized from the 100M final checkpoint.

## Part F — Configure the Kaggle notebook

```text
Accelerator: NVIDIA T4
Internet: On
Attached input: the private small-llm-20m-500m-dataset-001 dataset
```

Required Kaggle secrets are unchanged:

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

Attach the exact private Kaggle dataset version that passed the VPS round-trip verification. Training reads immutable local Kaggle input shards; it does not stream training data live from Hugging Face or Google Drive.

## Part G — Run or resume the fresh 500M training

```bash
cd /kaggle/working/Small-LLM
git switch main
git pull --ff-only
python kaggle/run_20m_500m.py
```

Do not pass `--launch-commit`. The wrapper pins the implementation worktree at:

```text
7c726ab51e4f3ed221d164e2596816da6d54c5cc
```

Fixed identities/defaults:

```text
model parameters: 20,637,592
architecture: gdn2_hybrid
fresh initialization seed: 17
context: 2,048
precision: FP16
optimizer: hybrid Muon + AdamW
learning rate: 3e-4
weight decay: 0.1
schedule: one-pass WSD derived from the 500M qualification plan
training microbatch: 4 sequences after the existing 1-vs-4 gate
validation microbatch: 1 sequence
local checkpoint cadence: 250 updates
held-out validation cadence: 250 updates
verified remote publication cadence: 250 updates
W&B run ID: 20m-500m-data-001
W&B run name: 20M model on 500M tokens
repository default session cap: none within the finite plan
```

Each invocation requests every remaining update in the finite plan. If Kaggle/runtime limits interrupt it, rerun the identical command. The launcher restores only the latest verified checkpoint under the distinct 500M run identity and checks the attached 500M Drive-manifest identity before resuming model, optimizer, scheduler, scaler, RNG, data cursor, and WSD position.

For a deliberate bounded diagnostic only:

```bash
python kaggle/run_20m_500m.py --max-steps-this-session 250
```

Do not use that override for the normal run.

Completion requires:

```text
status: completed
remaining_steps: 0
```

## Part H — Final comparison

After completion, run the same frozen evaluation bundle used for the 100M checkpoint. The key scientific comparison is the same 20M architecture and recipe at approximately 100M versus 500M source-token exposure.

At minimum preserve:

```text
final held-out loss and perplexity
fixed eval_core_v1 fast/full results
teacher-forced confidence/rank diagnostics
fixed free-generation prompt outputs and degeneration metrics
training loss-versus-token curve
throughput, overflow, and memory telemetry
exact dataset/manifest/checkpoint identities
```

Do not interpret lower perplexity alone as successful generation. The purpose of this final probe is specifically to learn whether the 20M model's free-generation behavior and other capabilities materially change after more than 20 source tokens per parameter.

## Stop conditions

Do not proceed if:

- the producer cannot resume its deterministic saved work plan;
- source identity/revision changes;
- local and Drive manifests disagree;
- a finalized shard is not remotely durable;
- full local verification fails;
- Kaggle upload or byte-identical round-trip verification fails;
- the target Kaggle dataset is anonymously readable;
- Kaggle finds zero or multiple matching 500M datasets;
- a 100M dataset is attached instead of the 500M identity;
- microbatch 4 fails the existing speed/numerical/memory gate;
- validation or FP16 safety gates fail;
- a restored checkpoint disagrees with the attached 500M manifest;
- W&B does not use `20m-500m-data-001`;
- a scheduled durability boundary lacks verified remote publication;
- the trainer attempts to consume beyond the exact finite one-pass plan.

A failed gate is evidence to inspect, not permission to silently alter the experiment.
