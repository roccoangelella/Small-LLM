# Project Status

_Last updated: 2026-08-04_

## Current phase

The project is preparing the finite dataset and exact-commit Kaggle evidence for
the first approximately-20M-parameter integrated engineering qualification.
Architecture selection is not being reopened at this stage.

The qualification remains an engineering test, not a model-quality claim.  It
must prove the selected model, hybrid optimizer, FP16 path, schema-v2 data,
checkpoint/resume, validation, W&B telemetry, remote publication, and recovery
contracts on one NVIDIA T4.

The approximately-100M architecture comparison and complete 90B dataset build
remain unauthorized until the 20M qualification ladder passes.

## Accepted dataset evidence

The authenticated 10M operational pilot passed on 2026-08-02 using the approved
exact weight SHA-256:

```text
76e82e22760adcac59c7294fe9bac11358f5a8b7a26035aae64c3f2e6fa1acb7
```

Accepted evidence includes:

```text
run ID: climbmix-pilot-001
accepted source tokens: 10,000,662
source documents: 14,136
immutable local shards: 7
matching durable Drive objects: 7
```

It passed real Drive durability, actual producer-process interruption,
deterministic resume, schema-v2 full verification, completed-resume
idempotence, and fail-closed acceptance verification.

That dataset used 512 sequences per block and remains immutable operational
evidence.  It is not the trainer qualification dataset because it would provide
only about ten optimizer updates.

## Fixed model and optimizer

The selected approximately-20M model has 20,637,592 parameters and uses:

```text
architecture: [GDN-2, GDN-2, GDN-2, full gated MHA] repeated
context length: 2,048
precision: FP16 model execution
GDN-2 backend: ordinary PyTorch chunkwise
GDN-2 chunk size: 32
initialization: normal
seed: 17
```

The corrected T4 harness previously passed recurrent/chunkwise parity for
chunks 16, 32, and 64.  Full-model FP16 passed for chunks 16 and 32; chunk 64
produced non-finite values.  Chunk 32 remains the trusted T4 FP16 path.

The first integrated optimizer is the implemented fail-closed hybrid:

```text
ordinary feature-transform matrices: whole-matrix Muon
embedding, norms, biases, dynamics, structured filters: AdamW
base LR: 3e-4
AdamW betas: 0.9 / 0.95
AdamW epsilon: 1e-8
AdamW weight decay: 0.1
Muon momentum: 0.95
Muon LR multiplier: 1.0
Muon target direction RMS: 0.18
Muon weight decay: 0.1
global gradient clipping: 1.0
```

Pure AdamW remains the later matched control.

## Fixed finite qualification dataset

The new training dataset is now fixed to:

```text
target accepted source tokens: 10,000,000
minimum accepted source tokens: 9,000,000
hard maximum accepted source tokens: 11,000,000
context length: 2,048
sequences per block: 16
target tokens per full optimizer update: approximately 32,768
target shard size: 8 MiB = 8,388,608 bytes
durable dataset checkpoint cadence: 2,000,000 source tokens
remote durability: required
passes: 1
silent wraparound: forbidden
```

The dedicated fail-closed entry point is:

```bash
uv run --env-file .env python -m dataset.qualification_20m \
  --weights-file <approved-weight-file.json> \
  --output-dir <new-output-directory> \
  --run-id 20m-qualification-dataset-001
```

It rejects changes to the fixed scientific/storage identity and rejects
`--allow-local-only`.

The completed verified manifest, not an approximation, determines the exact
train/validation target tokens, ordered block IDs, shard identities, one-pass
update count, and warmup/stable/decay token horizons.

## Trainer and telemetry implementation

The trainer currently provides:

- exact schema-v2 shard reading and geometry validation;
- one prepared block per atomic optimizer update;
- internal microbatching without mid-block acknowledgement;
- hybrid Muon + AdamW and pure AdamW;
- FP32, FP16, and BF16 execution;
- bounded GradScaler overflow retry and gradient clipping;
- constant and token-count warmup/stable/cosine-decay schedules;
- complete model, optimizer, scheduler, scaler, RNG, counter, and data-cursor
  checkpoint state;
- deterministic local resume primitives;
- validation and deterministic generation primitives;
- synchronous fail-closed two-phase publication to a private Hugging Face
  checkpoint repository;
- W&B telemetry under project `Small-LLM`.

Successful-step telemetry now includes:

- loss, LR, committed tokens, throughput, and block ID;
- GradScaler scale, retries, and cumulative overflows;
- global and per-optimizer-role pre-clipping gradient norms;
- clipping events;
- CUDA peak allocated and reserved memory;
- data wait separate from compute time;
- validation, local checkpoint, and remote publication timing;
- exact code, model, trainer, optimizer-routing, and manifest identities;
- Muon and AdamW pre-update weight RMS;
- optimizer-direction RMS;
- effective update RMS including LR and decoupled weight decay;
- effective update-to-weight ratios;
- named per-Muon-matrix direction RMS and update-to-weight ratios.

The optimizer statistics are runtime evidence only and do not change checkpoint
identity or state.

## Test-evidence policy

GitHub Actions is optional and is not required for the first qualification.
The user chose to run exact-commit evidence manually on Kaggle.

Before any trainer preflight, the Kaggle notebook must:

1. record and check out the exact final commit in detached-HEAD mode;
2. run the complete offline test suite;
3. persist the complete log and numeric exit code under `/kaggle/working`;
4. run the corrected T4 qualification harness;
5. persist its JSON report and require successful exit.

This concrete exact-commit evidence replaces, rather than weakens, an automated
workflow requirement.

## Qualification sequence

Current order:

1. run the complete offline suite on the current implementation commit;
2. build the fixed qualification dataset;
3. fully verify local shards, manifest, and Drive objects;
4. freeze the exact manifest-derived schedule and validation block list;
5. run a 20-successful-update constant-LR T4 preflight with W&B;
6. run an uninterrupted baseline and same-hardware A/A control;
7. freeze empirical throughput, memory, overflow, clipping, update-statistic,
   loss, and resume thresholds before the longer segment;
8. qualify actual-process local interruption/resume;
9. qualify remote publication and empty-environment restore with two-shard
   prefetch;
10. run the complete one-pass finite-dataset segment;
11. run validation and deterministic generation from trainer checkpoints;
12. record the final report and only then authorize the 100M comparison.

## Current readiness verdict

The code and decisions needed to produce the correct 10M/16-sequence dataset
and observe the first trainer preflight are now present.

The longer one-pass run is not yet authorized because runtime evidence remains:

- execute the tests on the final commit;
- build and verify the new qualification dataset;
- derive exact schedule values from its manifest;
- pass the 20-update T4 preflight and freeze empirical thresholds;
- pass local and remote recovery stages.

These are execution gates, not unresolved architecture or dataset-scope choices.
