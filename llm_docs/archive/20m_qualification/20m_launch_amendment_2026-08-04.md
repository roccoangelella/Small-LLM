# 20M Qualification Launch Amendment — 2026-08-04

This amendment records the latest user decisions and supersedes conflicting
"still open" statements in earlier readiness and status documents.

## Finite qualification dataset — fixed and implemented

The approximately-20M engineering qualification trains for one pass over a new,
separately built finite dataset.  The accepted operational pilot with 512
sequences per block is not reused as the training dataset.

The fixed qualification identity is:

```text
target accepted source tokens: 10,000,000
minimum accepted source tokens: 9,000,000
hard maximum accepted source tokens: 11,000,000
context length: 2,048
sequences per prepared block: 16
target shard size: 8 MiB = 8,388,608 bytes
durable dataset checkpoint cadence: 2,000,000 accepted source tokens
remote durability: required
local-only escape hatch: forbidden
```

The 2M durability cadence reuses the cadence already qualified by the
authenticated 10M operational pilot.  The 8 MiB shard target is specific to
this small qualification build; it does not change the future large-corpus
1 GiB default.

The fail-closed entry point is:

```bash
uv run --env-file .env python -m dataset.qualification_20m \
  --weights-file <approved-exact-weight-file.json> \
  --output-dir <new-qualification-dataset-directory> \
  --run-id 20m-qualification-dataset-001
```

The wrapper appends the fixed values and rejects attempts to override target,
minimum, maximum, context, block size, shard size, checkpoint cadence, or
remote-required operation.

## What the 10M build determines

The source-token target is fixed before the build, but the exact training
geometry is deliberately read from the completed verified manifest.  The build
therefore establishes:

- exact accepted source tokens;
- exact train and validation source-token counts;
- exact stored and loss-bearing target-token counts;
- exact train and validation document counts;
- exact ordered train and validation block IDs;
- exact number and sizes of immutable shards;
- exact number of one-pass optimizer updates;
- exact dataset, schema, work-plan, weight, local-manifest, and Drive-manifest identities.

Only after those values exist do we calculate and freeze the longer-run token
schedule:

```text
warmup updates: max(16, 5% of planned updates)
decay updates: final 20% of planned updates
stable updates: all remaining updates
minimum LR ratio: 0.1
```

The illustrative value of about 305 updates is not a launch input.

## Optimizer telemetry — implemented for the selected hybrid run

The successful-step record now includes the previously missing update
statistics.  For each optimizer role it records:

- pre-update weight RMS;
- optimizer-direction RMS before LR and decoupled weight decay;
- effective update RMS including LR and decoupled weight decay;
- effective update-to-weight ratio;
- number of parameter tensors and scalar elements.

For every Muon-routed matrix it additionally records:

- the named matrix optimizer-direction RMS, which should reflect the configured
  `0.18` normalization target;
- the named matrix effective update-to-weight ratio.

The statistics are derived from the FP32 optimizer arithmetic without cloning
the complete model.  They are cleared before every GradScaler candidate and
published only when the scaler accepts the optimizer update.  They are runtime
telemetry, not checkpoint state.

W&B receives these nested values automatically under the training metric
namespace, together with the already implemented loss, throughput, gradient,
clipping, scaler, overflow, memory, data-wait, validation, checkpoint, remote
publication, and exact-identity telemetry.

## GitHub Actions — not required for this qualification

The user chose not to depend on GitHub Actions for the first qualification.
This is acceptable.  GitHub Actions is an optional automation convenience, not
a correctness requirement.

The exact-commit evidence gate is instead executed interactively in the Kaggle
T4 notebook:

1. record the final launch commit SHA;
2. check out that exact commit in detached-HEAD mode;
3. run the complete offline suite;
4. persist the full test log and numeric exit code under `/kaggle/working`;
5. run the corrected T4 hardware harness and persist its JSON report;
6. do not start the trainer unless both commands exit successfully.

This manual evidence is valid because it binds concrete outputs to the exact
commit and target hardware.  A future decision may add GitHub Actions without
changing the scientific protocol.

## Readiness after this amendment

The code and decisions required to build the correct qualification dataset are
present.  The repository is not yet authorized for the longer one-pass trainer
segment until the following runtime gates occur:

1. execute the complete offline suite on the new exact commit;
2. build the qualification dataset through `dataset.qualification_20m`;
3. fully verify its local files and Drive objects;
4. freeze the manifest-derived schedule values and validation block list;
5. run the short 20-update constant-LR T4 preflight with W&B;
6. review the newly available optimizer-update telemetry and freeze empirical
   thresholds;
7. qualify local resume and remote empty-environment recovery.

No GitHub workflow run is required for those gates.
