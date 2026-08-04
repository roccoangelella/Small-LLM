# Project Status

_Last updated: 2026-08-04_

## Current phase

The fixed finite dataset for the first approximately-20M-parameter integrated
engineering qualification has been built, durably mirrored, fully scanned, and
accepted. The project now moves to exact-commit Kaggle T4 qualification.

Architecture selection is not being reopened. The qualification is engineering
evidence, not a model-quality claim. The approximately-100M architecture
comparison and complete 90B dataset build remain unauthorized until the full
20M qualification ladder passes.

## Fixed model and optimizer

```text
parameters: 20,637,592
architecture: [GDN-2, GDN-2, GDN-2, full gated MHA] repeated
context length: 2,048
precision: FP16
GDN-2 backend: ordinary PyTorch chunkwise
GDN-2 chunk size: 32
initialization: normal
seed: 17
```

Primary optimizer:

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

## Accepted qualification dataset

The fixed 10M/16-sequence qualification dataset completed on the VPS:

```text
run ID: 20m-qualification-dataset-001
accepted source tokens: 10,000,662
train source tokens: 9,991,872
validation source tokens: 8,790
train shards: 6
validation shards: 1
train sequences: 4,886
validation sequences: 5
train blocks / one-pass optimizer updates: 306
stored uint16 tokens: 10,021,659
manifest SHA-256: 1e5ee8f372b77b6728288610dbe7cce74d833be21e53d1538bc5a890229b18bb
Drive manifest SHA-256: fbb29ee0d0102658e1274e39d6647cf56a6dcb685e0f566b1736847dcc4fbe84
```

Accepted evidence now includes:

- focused dedicated-verifier tests: `3 passed`;
- schema-v2 structural and per-shard SHA-256 verification;
- exact local-to-Drive shard identity and remote durability;
- literal token-by-token scan of all 10,021,659 stored `uint16` tokens;
- no out-of-vocabulary IDs and no geometry problems;
- exact per-cluster source-token total equal to 10,000,662;
- verifier exit code `0`, `passed=true`, `complete=true`, `full_scan=true`, and
  `problems=[]`;
- regenerated qualification plan with unchanged identities and schedule.

The dataset is accepted for private Kaggle packaging. It is an engineering
qualification dataset and should not be used as strong model-quality or broad
mixture-coverage evidence.

## Exact one-pass plan

```text
schedule: WSD
passes: 1
steps: 306
full-block target tokens: 32,768
warmup: 16 updates / 524,288 target tokens
stable: 228 updates / 7,471,104 target tokens
decay: 62 updates / 2,011,136 target tokens
minimum LR ratio: 0.1
validation blocks: 1
train target tokens: 10,006,528
```

The final training block is partial. Silent data wraparound remains forbidden.

## Trainer and telemetry implementation

The trainer provides exact schema-v2 shard reading, one prepared block per
atomic update, internal microbatching, FP16 GradScaler retry, clipping, WSD and
constant token schedules, complete local checkpoint state, deterministic resume,
validation and generation, private Hugging Face checkpoint publication, and W&B
telemetry.

Successful-step telemetry includes loss, LR, committed tokens, throughput,
block ID, scaler state and overflows, global and branch gradient norms, clipping,
CUDA allocation/reservation, data wait, checkpoint/validation/publication time,
exact code/model/data identities, and Muon/AdamW weight/direction/effective-update
RMS and update-to-weight ratios including named Muon matrices.

## Evidence policy

GitHub Actions is optional. Exact-commit evidence will be run manually on
Kaggle and persisted under `/kaggle/working` with complete logs, JSON reports,
and numeric exit codes.

Before trainer preflight, Kaggle must:

1. record the final commit and check it out in detached-HEAD mode;
2. run the complete offline test suite;
3. persist the complete test log and exit code;
4. run the corrected T4 qualification harness;
5. persist its report and require successful exit;
6. verify the attached private dataset and reproduce the exact plan.

## Remaining qualification sequence

1. Package and attach the accepted dataset as a private Kaggle Dataset.
2. Freeze the exact implementation commit for Kaggle evidence.
3. Run the complete offline suite on that exact commit.
4. Run the corrected T4 harness on that exact commit.
5. Run a 20-successful-update constant-LR preflight with W&B.
6. Freeze empirical loss, throughput, memory, overflow, clipping, and optimizer
   update-statistic thresholds.
7. Run uninterrupted baseline and same-hardware A/A control.
8. Qualify actual-process local interruption/resume.
9. Qualify remote publication and empty-environment two-shard restore.
10. Authorize and run the complete 306-update one-pass segment.
11. Run final validation and deterministic generation.
12. Record the final report before considering the 100M comparison.

## Current readiness verdict

**Dataset gate: passed.**

**Ready next:** private Kaggle packaging and exact-commit suite/T4 evidence.

**Not yet authorized:** the complete 306-update training segment, because the
20-update preflight, empirical threshold freeze, and local/remote recovery gates
remain outstanding.
