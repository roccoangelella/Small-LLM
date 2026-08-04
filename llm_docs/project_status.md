# Project Status

_Last updated: 2026-08-04_

## Current phase

The finite approximately-20M engineering qualification dataset is built,
durably mirrored, fully scanned, and accepted. Exact-commit Kaggle T4 evidence,
the 20-update integrated preflight, and the two-run 50-update same-T4
repeatability measurement have passed.

The project is now in **threshold interpretation and recovery qualification**.
Architecture selection is not being reopened. The complete 306-update one-pass
segment, approximately-100M architecture comparison, and complete 90B dataset
build remain unauthorized until checkpoint interpretation, local
interruption/resume, and remote empty-environment recovery pass.

Detailed evidence is recorded in:

```text
llm_docs/20m_kaggle_preflight_results.md
llm_docs/20m_repeatability_results.md
```

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

Pure AdamW remains the later matched control. No optimizer, LR, or clipping
change is authorized from the qualification evidence without a separately
recorded one-variable decision.

## Accepted qualification dataset

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

Accepted evidence includes schema-v2 structural and per-shard SHA-256
verification, exact local-to-Drive identity, literal token-by-token scanning of
all stored tokens, no vocabulary or geometry problems, exact cluster accounting,
and exact plan regeneration from the private Kaggle mount.

The dataset is engineering qualification data, not strong model-quality or broad
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

## Passed exact-commit Kaggle gates

The evidence-producing worktree was clean and detached at:

```text
launch commit: 45d1da4a1ac3f18cf6ce02b8439672f10e2c8b4c
GPU: Tesla T4
```

Passed gates:

```text
offline suite: 229 passed, 1 expected live-remote skip
corrected T4 harness: passed
dataset full scan: passed
exact qualification plan reproduction: passed
20-update trainer preflight: passed
50-update reference segment: passed
50-update same-T4 A/A segment: passed
```

The corrected T4 harness retained the primary candidate:

```text
architecture: gdn2_hybrid
backend: pytorch_chunkwise
chunk size: 32
precision: FP16
FP16 overflows: 0
```

## Twenty-update preflight summary

```text
training loss: 10.845867 -> 9.573909
validation loss: 9.240405 on 10,240 target tokens
mean throughput: 1,066.12 target tokens/s
maximum reserved CUDA memory: 2,868 MiB
GradScaler: stable at 65,536
overflow events / retries: 0 / 0
checkpoint: step-00000020
```

Gradient clipping occurred on all 20 updates, creating the optimizer-stability
review flag that motivated the longer repeatability measurement.

## Fifty-update repeatability result

The uninterrupted reference and independent A/A repeat each consumed exactly
50 blocks / 1,638,400 target tokens using the same WSD prefix, seed,
initialization, model, optimizer, data order, and T4.

Final status:

```text
status: passed_repeatability_measurement
authorization: threshold_review_only
evidence directory: /kaggle/working/small-llm-repeatability-controller/small-llm-repeatability-20260804T145817Z
summary: /kaggle/working/small_llm_repeatability_summary.json
```

### Exact metric repeatability

```text
compared numerical values: 10,650
differing numerical values: 0
maximum absolute difference: 0.0
maximum relative difference: 0.0
numeric trajectory exact: true
discrete trajectory exact: true
validation exact: true
```

No recorded training-trajectory nondeterministic floor was observed. Loss,
gradient norms, clipping decisions, LR, FP16 state, optimizer telemetry,
counters, block order, and validation matched exactly.

Both runs produced:

```text
training loss, update 1: 10.845867
training loss, update 50: 8.090633
validation loss: 7.915478
GradScaler: 65,536 throughout
overflow events / retries: 0 / 0
maximum allocated CUDA memory: 2,510,114,816 bytes
maximum reserved CUDA memory: 3,007,315,968 bytes
```

Runtime varied materially between the runs while the math stayed exact:

```text
reference mean throughput: 903.77 target tokens/s
repeat mean throughput: 1,000.97 target tokens/s
```

Runtime thresholds must therefore use distributions rather than exact equality.

### Clipping interpretation

Gradient clipping occurred on all 50 updates in both runs. The clipping review
flag remains active because the frequency exceeds the protocol's provisional
band.

However, the longer evidence does not show runaway norm growth:

```text
first-10 median pre-clip norm: 1.399718
last-10 median pre-clip norm: 1.385033
final pre-clip norm: 1.344303
maximum pre-clip norm: 2.680975
```

The pattern is exactly repeatable, bounded over this window, finite, and
accompanied by improving loss and stable FP16/optimizer telemetry. The recipe is
reproducibly clipping-dependent, but the evidence does not justify a silent LR
or clipping change.

### Checkpoint byte-level mismatch

The expected step-25 and step-50 checkpoints existed and passed structural
verification in both runs, but complete checkpoint-tree hashes were not
byte-identical.

Because all compared training and validation telemetry was exactly identical,
this is currently classified as unresolved serialization or run-metadata
nondeterminism rather than demonstrated model-state divergence.

Before the resume gate, comparison must distinguish:

1. semantic model, optimizer, scheduler, scaler, RNG, counter, and cursor state;
2. expected run-specific metadata;
3. raw serialization byte differences.

The resume test must fail closed on semantic state mismatch. Because A/A metric
trajectories were exact, the default expectation for the post-resume trajectory
is exact equality unless checkpoint analysis identifies a justified
serialization-only exception.

## Remaining qualification sequence

1. Preserve the preflight and repeatability evidence directories and W&B runs.
2. Inspect the step-25 and step-50 checkpoint trees and identify the exact source
   of byte-level differences.
3. Freeze empirical warning/failure thresholds for loss, gradient norms,
   clipping, FP16 state, optimizer telemetry, memory, and runtime distributions.
4. Decide explicitly whether universal but bounded clipping requires a
   one-variable diagnostic before recovery qualification.
5. Run an actual-process interruption at the update-25 checkpoint boundary and
   resume from local state.
6. Compare resumed semantic state and post-resume trajectory against the
   uninterrupted reference.
7. Qualify private remote publication and empty-environment restore, including
   verified two-shard prefetch and exact next-block continuation.
8. Authorize and run the complete 306-update one-pass segment only after all
   preceding gates pass.
9. Run final validation and deterministic generation, then record the final
   qualification report before considering the approximately-100M comparison.

## Current readiness verdict

**Dataset gate: passed.**

**Exact-commit offline/T4/mounted-data gates: passed.**

**Twenty-update integrated trainer preflight: passed.**

**Fifty-update same-T4 metric repeatability: passed with exact recorded
trajectory equality.**

**Ready next:** checkpoint-difference analysis, empirical threshold freeze, and
local process-kill/resume qualification.

**Not yet authorized:** the complete 306-update one-pass segment, because
checkpoint semantic interpretation, clipping policy, local interruption/resume,
and remote empty-environment recovery remain outstanding.
