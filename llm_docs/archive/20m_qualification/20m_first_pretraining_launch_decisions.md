# Approximately-20M First Pretraining Launch Decisions

_Last updated: 2026-08-04_

## Purpose

This file is the consolidated decision record for the first approximately-20M
integrated engineering qualification.  It is not a model-quality experiment.

## Model and optimizer

```text
model parameters: 20,637,592
architecture: gdn2_hybrid
context length: 2,048
precision: FP16
GDN-2 chunk size: 32
initialization: normal
optimizer: hybrid whole-matrix Muon + AdamW
base LR: 3e-4
AdamW betas: 0.9 / 0.95
AdamW epsilon: 1e-8
AdamW weight decay: 0.1
Muon momentum: 0.95
Muon LR multiplier: 1.0
Muon target direction RMS: 0.18
Muon weight decay: 0.1
global gradient clip: 1.0
seed: 17
```

Pure AdamW remains the later matched control.  Seed 17 is sufficient for this
engineering qualification but not for single-seed model-quality claims.

## Dataset scope and lifecycle

The qualification uses a new finite dataset, not the accepted 512-sequence
operational pilot.

```text
target accepted source tokens: 10,000,000
minimum accepted source tokens: 9,000,000
hard maximum accepted source tokens: 11,000,000
context length: 2,048
sequences per block: 16
target shard size: 8 MiB = 8,388,608 bytes
durable dataset checkpoint cadence: 2,000,000 source tokens
remote durability: required
passes: 1
implicit wraparound: forbidden
```

Build only through:

```bash
uv run --env-file .env python -m dataset.qualification_20m \
  --weights-file <approved-exact-weight-file.json> \
  --output-dir <new-output-directory> \
  --run-id 20m-qualification-dataset-001
```

The wrapper rejects changes to fixed geometry and rejects local-only operation.
The dataset is completed and fully verified before the trainer starts.
Producer/trainer overlap remains a separate later operational test.

## Exact values derived from the 10M build

The 10M run is intentionally used to determine the exact values that cannot be
known from the source-token target alone:

- exact accepted, train, and validation source tokens;
- exact stored and loss-bearing target tokens;
- exact ordered train and validation block IDs;
- exact train and validation block counts;
- exact shard count, sizes, and hashes;
- exact one-pass optimizer-update count;
- exact dataset, schema, work-plan, weight, local-manifest, and Drive-manifest
  identities.

After the verified manifest exists, freeze the longer-run token horizons:

```text
warmup updates: max(16, 5% of planned updates)
decay updates: final 20% of planned updates
stable updates: all remaining updates
minimum LR ratio: 0.1
```

Approximately 305 updates is only an estimate and is never copied blindly into
the launch command.

## Validation

The document-identity validation split remains deterministic.  Accepted and
excluded clusters, production cluster weights, and the exact mixture scheduler
remain unchanged.  Validation is not rebalanced, oversampled, or edited after
the build.

After completion, freeze:

- dataset and Drive-manifest hashes;
- complete ordered validation block IDs;
- validation token and document counts by cluster;
- deterministic generation prompts and their SHA-256.

Validation at this scale is a functional health signal, not a strong estimate
of model quality.

## Checkpoints and recovery

Provisionally approved cadence:

```text
local joint checkpoint: every 25 successful updates
validation: every 50 successful updates
remote joint-checkpoint publication: every 50 successful updates
```

Recurring overhead is measured on the exact T4 path.  Validation and remote
publication may move to every 100 updates only after evidence; local checkpoint
cadence does not widen automatically.

The first empty-environment recovery prefetches two consecutive train shards,
starting with the shard containing the next unconsumed block.

Remote publication is synchronous and fail-closed.  The CLI writes the local
atomic checkpoint first, publishes and verifies it in the private Hugging Face
repository, and only then advances the remote latest pointer.  Publication
failure leaves the local checkpoint intact and fails the command.

## W&B telemetry

The first T4 qualification uses project:

```text
Small-LLM
```

The API key is provided through the ignored `.env` file as `WANDB_API_KEY` and
is never added to launch configuration, logs, checkpoints, or the repository.

Successful-step telemetry includes:

- loss, LR, block ID, token counts, throughput;
- GradScaler scale, retries, cumulative overflows;
- global and per-role pre-clipping gradient norms;
- clipping events;
- CUDA allocated and reserved memory;
- data-wait and compute time;
- validation, checkpoint, and remote-publication timings;
- exact code, model, optimizer-routing, and manifest identities;
- optimizer-direction RMS by Muon/AdamW role;
- actual post-cast effective update RMS by role;
- actual effective update-to-weight ratio by role;
- named per-Muon-matrix direction RMS and actual update-to-weight ratio.

Only one parameter is cloned at a time to measure its actual post-update delta;
no full-model telemetry copy is retained.  Statistics are exposed only after a
GradScaler-accepted update and are not checkpoint state.

## Exact-commit tests without GitHub Actions

GitHub Actions is optional and not required for this qualification.  The user
selected manual exact-commit evidence in Kaggle:

1. record the final commit SHA;
2. check out that SHA in detached-HEAD mode;
3. run the complete offline suite;
4. save the full log and numeric exit code under `/kaggle/working`;
5. run the corrected T4 harness;
6. save its JSON report and require successful exit;
7. only then start the trainer preflight.

## Qualification ladder

1. Build and verify the fixed finite dataset.
2. Freeze manifest-derived schedule values and validation block IDs.
3. Run exact-commit offline tests on Kaggle.
4. Run the corrected T4 model harness.
5. Run a 20-successful-update constant-LR trainer preflight with W&B.
6. Run an uninterrupted baseline and same-hardware A/A repeatability control.
7. Freeze empirical thresholds before the longer segment.
8. Qualify actual-process local interruption/resume.
9. Qualify remote publication and two-shard empty-environment recovery.
10. Run the complete one-pass finite-dataset segment.
11. Run validation and deterministic generation from trainer checkpoints.
12. Authorize the approximately-100M comparison only after the report passes.
