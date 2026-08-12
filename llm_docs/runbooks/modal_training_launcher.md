---
status: current
last_reviewed: 2026-08-12
---

# Modal training launcher

The canonical new-training and resume entry point is `modal/launch.py`. It is a provider adapter around the existing `dataset`, `model`, and `trainer` packages; it does not define a second scientific trainer. All operator commands for the Modal lane run from the VPS.

## VPS setup

Use the project `.venv` and install the two operator CLIs there:

```bash
source .venv/bin/activate
uv pip install kaggle 'modal>=1.1,<2'
modal setup
modal volume create small-llm-data
modal volume create small-llm-runs
modal volume create small-llm-cache
modal secret create small-llm-training \
  WANDB_API_KEY="$WANDB_API_KEY" \
  HF_TOKEN="$HF_TOKEN" \
  SMALL_LLM_HF_REPO_ID="$SMALL_LLM_HF_REPO_ID"
```

The secret can be created from an existing root environment file without copying values manually:

```bash
set -a && source /root/.env && set +a && \
modal secret create small-llm-training \
  WANDB_API_KEY="$WANDB_API_KEY" \
  HF_TOKEN="$HF_TOKEN" \
  SMALL_LLM_HF_REPO_ID="$SMALL_LLM_HF_REPO_ID"
```

`SMALL_LLM_HF_REPO_ID` must identify the private model repository dedicated to this checkpoint/artifact workflow. ADR 0046 uses rolling cleanup plus Hugging Face branch-history squashing to keep resume storage bounded, so do not point it at a repository whose historical Git commit graph must be preserved.

Kaggle is only the remote source for the already-published 2B finite dataset. Authenticate the VPS Kaggle CLI with `KAGGLE_API_TOKEN` or an official token file. No Kaggle notebook participates in the Modal workflow.

## Prepare the 2B Modal corpus from the VPS

The 100M / 2B run uses the `modal-2b-b64` dataset profile rather than editing the historical block-16 corpus. The supported one-command preparation path is:

```bash
python modal/prepare_dataset.py
```

This command discovers the authenticated user's existing Kaggle dataset `small-llm-20m-2b-dataset-001`, downloads it to the VPS only if a verified cached source is absent, verifies production run ID `20m-2b-dataset-001`, performs the byte-preserving block-64 transformation through `dataset.reblock`, verifies the derivative, and uploads it from the VPS to Modal Volume `small-llm-data`.

Fixed VPS paths:

```text
~/small-llm-data/kaggle/small-llm-20m-2b-dataset-001
~/small-llm-data/modal-2b-b64-dataset-001
```

Fixed Modal destination:

```text
/datasets/modal-2b-b64-dataset-001
```

The helper is stage-idempotent. Rerunning normally skips a verified download, a verified reblock, and a matching completed upload. Repair flags are `--force-download`, `--force-reblock`, and `--force-upload`; `--no-upload` stops after VPS preparation.

When changing Modal accounts/workspaces while keeping the same VPS cache, use `--force-upload` until the upload-marker workspace-identity bug is removed: the current local marker can otherwise describe an upload performed against the previous workspace.

The original Kaggle dataset remains unchanged. The derived profile is:

```text
profile: modal-2b-b64
dataset run ID: modal-2b-b64-dataset-001
context: 2,048
sequences per optimizer block: 64
target shard size: 32 MiB
train target tokens: 1,999,994,880
optimizer updates: 15,259
final train block: 48 sequences
```

## Dry run

```bash
modal run modal/launch.py --model 100M --tokens 2B --dry-run
```

Dry-run resolution does not rent a GPU. It should resolve the dataset profile to `modal-2b-b64` and the automatic microbatch set to `16,32,48,64`.

## Production and resume command

```bash
modal run --detach modal/launch.py \
  --model 100M \
  --tokens 2B \
  --gpu H100
```

The same command starts a fresh trajectory, resumes from the current workspace's verified Modal checkpoint, or restores from the configured Hugging Face checkpoint transport when the new workspace has no verified local checkpoint.

`H100` is the default and permits the compatible H200 automatic upgrade. On a genuinely fresh trajectory the default microbatch value is `0`, meaning benchmark 16, 32, 48, and 64 and freeze the fastest candidate that is finite and stays at or below 90% reserved VRAM. On a restored trajectory the frozen microbatch is recovered from durable run metadata rather than reprobed.

Use an explicit microbatch only when reproducing a qualified run:

```bash
modal run --detach modal/launch.py \
  --model 100M --tokens 2B --gpu H100 --microbatch-size 32
```

The prepared optimizer block contains 64 sequences. A full block is approximately 131,072 target tokens. Microbatch 16 executes four slices per optimizer update; 32 executes two; 48 executes 48+16; 64 executes one full-block pass.

For the exact existing 2B training stream the reblocked plan has 15,259 optimizer updates, with a final 48-sequence block. Token-space WSD boundaries remain 100,007,936 warmup tokens, 1,499,987,968 stable tokens, and 399,998,976 decay tokens.

## Startup and resume gates

The launcher:

1. requires a clean controlling Git checkout and records its exact commit;
2. discovers exactly one matching finite dataset on the read-only data Volume;
3. performs a full schema-v2 verification once per manifest identity;
4. derives the one-pass WSD plan from the dataset manifest;
5. looks for the newest verified local `small-llm-runs/<run-id>/checkpoints/step-XXXXXXXX` checkpoint;
6. if no verified local checkpoint exists, reads `run/<run-id>/latest.json` from the configured private Hugging Face repository and restores that checkpoint only after both local and published manifests verify;
7. for migration of the pre-ADR-0046 100M / 2B run only, falls back to `models/<run-id>/artifact.json` plus its referenced checkpoint and verifies that legacy checkpoint before installing it locally;
8. recovers the frozen execution microbatch from restored metadata, or runs the 16/32/48/64 qualification only for a genuinely fresh trajectory;
9. freezes model/data identity, precision, block geometry, microbatch and checkpoint transport metadata;
10. starts or resumes the stable W&B run identity.

A 48 or 64 failure on an 80 GB H100 is an expected capacity measurement only for a fresh qualification, not a launcher failure as long as at least one candidate passes. An H200 may admit a larger candidate.

## Checkpoint durability and Hugging Face transport

ADR 0046 defines two durability layers:

```text
same-workspace durability:
  Modal Volume small-llm-runs
  every 250 successful optimizer updates + final

cross-workspace durability:
  private Hugging Face model repository
  every 500 successful optimizer updates + final
  namespace: run/100m-2b-data-001/
  retention: latest resumable checkpoint only, with history squash
```

At an HF boundary the trainer first creates a normal verified joint checkpoint. The two-phase publisher uploads its complete checkpoint tree, reads the uploaded bytes back for SHA-256 verification, writes the checkpoint manifest, and moves `run/100m-2b-data-001/latest.json` only after the snapshot is valid. Rolling cleanup then removes superseded run checkpoint folders, squashes branch history, and reads the current latest pointer back before training continues.

This replaces the former external ten-minute `modal/publish_hf.py` backup loop. Do **not** run that loop for live checkpoint durability.

A same-workspace retry can therefore lose at most the work after the newest 250-update Modal boundary. A different Modal account/workspace can restore from the newest 500-update HF boundary. The exact trainer checkpoint contains model, optimizer, WSD scheduler, FP16 scaler, RNG state, counters, and data cursor.

W&B runs online in project `Small-LLM` with stable run ID `100m-2b-data-001`. Any actual checkpoint resume uses the same W&B identity with `must` resume semantics.

## Migrating the already-running 100M / 2B trajectory to a new Modal account

The first new-workspace launch can bootstrap from the old Hugging Face publication layout created by `modal/publish_hf.py`:

```text
models/100m-2b-data-001/artifact.json
models/100m-2b-data-001/step-XXXXXXXX/
```

If those objects still exist, the new launcher verifies and installs that checkpoint, reuses the recorded microbatch, resumes the same W&B run, and establishes the new rolling `run/100m-2b-data-001/...` transport at the next remote boundary. The infrastructure-only source-commit migration is recorded; subsequent rolling-HF restores again require the launcher checkout to match the transport checkpoint's source commit.

If the old HF checkpoint was deleted and the old Modal workspace is also inaccessible, no launcher can reconstruct the missing trainer state. One surviving verified copy must be republished or migrated first.

## Final human-facing Hugging Face artifact

ADR 0044 remains in force independently of the rolling resume transport. After training completes, materialize the exact final model/checkpoint under the human-facing `models/...` namespace with:

```bash
git pull --ff-only
modal run modal/publish_hf.py --model 100M --tokens 2B --require-complete
```

The final publication command fails closed unless the latest verified local checkpoint step equals the qualification plan's full 15,259-step target. It writes the final checkpoint under:

```text
models/100m-2b-data-001/step-XXXXXXXX/
```

and identity metadata under:

```text
models/100m-2b-data-001/artifact.json
```

Do not use this command in an external periodic loop; live cross-workspace durability is now integrated in `modal/launch.py`.

## Hardware migration policy

The historical FLA acceptance is T4/SM75 evidence. H100/H200 uses a different target architecture, so the first bounded probe was required even though recurrence semantics are unchanged. Keep FP16 for this platform trajectory. Treat BF16 or Blackwell as separate follow-up qualifications rather than combining them with this run.

## Adding a future model

Add the accepted geometry to the shared model/trainer preset surface, then register its nominal parameter label in `modal/profiles.py::MODEL_PRESETS`. Token budgets map to finite dataset profiles independently of model size. The block-64 profile is specifically authorized by ADR 0041 for the 100M / 2B trajectory; later batch changes require their own qualification/decision. Future Modal trajectories should keep model/run-specific HF checkpoint identities rather than returning to dataset-only checkpoint namespaces.
