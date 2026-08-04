# Approximately-20M Training Readiness

_Last updated: 2026-08-04_

## Purpose

The approximately-20M model is an end-to-end engineering qualification.  It
must prove the selected architecture, hybrid optimizer, FP16 execution,
schema-v2 dataset, telemetry, checkpoint/resume, validation, generation, remote
publication, and empty-environment recovery on one NVIDIA T4.

It is not a model-quality or architecture-ranking experiment.

## Fixed execution profile

```text
model parameters: 20,637,592
architecture: gdn2_hybrid
context length: 2,048
precision: FP16
GDN-2 backend: ordinary PyTorch chunkwise
GDN-2 chunk size: 32
initialization: normal
optimizer: hybrid whole-matrix Muon + AdamW
microbatch size: 1
sequences per atomic block: 16
approximate target tokens per full update: 32,768
seed: 17
```

The selected scalar baseline is:

```text
base LR: 3e-4
AdamW beta1 / beta2: 0.9 / 0.95
AdamW epsilon: 1e-8
AdamW weight decay: 0.1
Muon momentum: 0.95
Muon LR multiplier: 1.0
Muon target direction RMS: 0.18
Muon weight decay: 0.1
global gradient clipping norm: 1.0
```

The short preflight uses constant LR.  The longer one-pass segment uses the
implemented token-count warmup/stable/cosine-decay scheduler.

## Fixed finite dataset profile

The accepted operational 10M pilot remains immutable evidence with 512
sequences per block.  The trainer qualification uses a new dataset with:

```text
target source tokens: 10,000,000
minimum source tokens: 9,000,000
hard maximum source tokens: 11,000,000
context length: 2,048
sequences per block: 16
target shard size: 8 MiB = 8,388,608 bytes
durable checkpoint cadence: 2,000,000 source tokens
remote durability: required
passes: 1
implicit wraparound: forbidden
```

Build it only through:

```bash
uv run --env-file .env python -m dataset.qualification_20m \
  --weights-file <approved-exact-weight-file.json> \
  --output-dir <new-output-directory> \
  --run-id 20m-qualification-dataset-001
```

The entry point rejects changes to the fixed token bounds, context, block size,
shard size, durability cadence, and remote-required policy.

## Values intentionally derived from the completed build

The 10M source-token target is known now.  These exact values are not known
until the producer completes and verifies its manifest:

- accepted source-token total;
- train and validation source-token totals;
- train and validation stored/target-token totals;
- complete ordered train and validation block IDs;
- number and sizes of immutable shards;
- exact one-pass optimizer-update count;
- dataset, schema, source, weight, local-manifest, and Drive-manifest hashes.

The longer-run schedule is calculated from those exact block/token values:

```text
warmup updates: max(16, 5% of planned updates)
decay updates: final 20% of planned updates
stable updates: all remaining updates
minimum LR ratio: 0.1
```

Approximate counts such as 305 updates are planning estimates only.

## Implemented telemetry

W&B project `Small-LLM` receives successful-step metrics using
`trainer/global_step` as the common axis.

Implemented signals include:

- loss, LR, block ID, committed tokens, throughput;
- current GradScaler scale, retries, cumulative overflow events;
- global and per-role pre-clipping gradient norms;
- clipping flag;
- allocated and reserved CUDA memory;
- data-wait and compute time;
- validation duration, tokens, loss, and perplexity;
- local checkpoint duration and size;
- remote publication duration and final-boundary flag;
- exact Git, model, trainer, optimizer-routing, and manifest identities;
- pre-update weight RMS by optimizer role;
- optimizer-direction RMS by optimizer role;
- effective update RMS including LR and decoupled decay;
- effective update-to-weight ratios by optimizer role;
- named per-Muon-matrix direction RMS and update-to-weight ratios.

The update statistics are cleared before each GradScaler candidate and exposed
only after a successful optimizer update.  They are not checkpoint state.

## Test evidence without GitHub Actions

GitHub Actions is not required for this qualification.  The exact-commit gate
runs manually in Kaggle:

1. check out the exact final commit in detached-HEAD mode;
2. run the complete offline suite;
3. save its full log and exit code under `/kaggle/working`;
4. run the corrected T4 harness;
5. save its JSON report and require successful exit.

A future workflow may automate this, but its absence does not block the current
manual qualification.

## Current readiness

### Ready now

- build the correctly configured finite qualification dataset;
- run the complete CPU/offline suite on Kaggle against the exact commit;
- rerun the corrected T4 model harness;
- inspect W&B during a short trainer preflight once the dataset exists.

### Not yet authorized

The complete one-pass 10M trainer segment is not authorized until:

1. the new dataset is complete and fully verified locally and on Drive;
2. exact schedule values and validation block IDs are frozen from its manifest;
3. the 20-update constant-LR preflight passes;
4. empirical warning/failure thresholds are frozen from preflight and A/A
   evidence;
5. local interruption/resume passes;
6. remote publication and two-shard empty-environment recovery pass.

These are runtime evidence gates.  Dataset scale, shard size, model profile,
optimizer, seed, pass count, telemetry, and test-execution venue are no longer
open decisions.
