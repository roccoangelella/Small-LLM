# Approximately-20M Same-T4 Repeatability Results

_Last updated: 2026-08-04_

## Verdict

The 50-update uninterrupted reference run and the independent 50-update same-T4 A/A repeat completed successfully on 2026-08-04.

```text
status: passed_repeatability_measurement
authorization: threshold_review_only
scope: same-T4 50-update reference and A/A WSD-prefix repeatability measurement
launch commit: 45d1da4a1ac3f18cf6ce02b8439672f10e2c8b4c
controller commit: 557dd4530d4955f32aab230a2d9c7a760f36e5e1
GPU: Tesla T4, 15,360 MiB
started UTC: 2026-08-04T14:58:17.229462+00:00
finished UTC: 2026-08-04T15:58:41.747692+00:00
evidence directory: /kaggle/working/small-llm-repeatability-controller/small-llm-repeatability-20260804T145817Z
summary: /kaggle/working/small_llm_repeatability_summary.json
```

This is a repeatability-measurement pass. It does not authorize the complete 306-update segment. Threshold interpretation, local interruption/resume, and remote empty-environment recovery remain required.

## Identity and data gates

The launcher used a clean detached worktree at the frozen launch commit and selected the accepted private Kaggle dataset by its exact identities:

```text
manifest SHA-256: 1e5ee8f372b77b6728288610dbe7cce74d833be21e53d1538bc5a890229b18bb
Drive manifest SHA-256: fbb29ee0d0102658e1274e39d6647cf56a6dcb685e0f566b1736847dcc4fbe84
run ID: 20m-qualification-dataset-001
```

The literal full dataset scan and exact 306-update plan regeneration both returned exit code `0` before either trainer run began.

## Run configuration

Both runs used the same:

```text
steps: 50
schedule: WSD prefix using the frozen 306-update token horizons
warmup tokens: 524,288
stable tokens: 7,471,104
decay tokens: 2,011,136
minimum LR ratio: 0.1
seed: 17
architecture: gdn2_hybrid
GDN-2 chunk size: 32
precision: FP16
optimizer: hybrid Muon + AdamW
checkpoint cadence: updates 25 and 50
validation: update 50
```

Each run consumed exactly 1,638,400 target tokens.

W&B run IDs:

```text
reference: 20m-t4-aa-20260804-145830-reference
repeat: 20m-t4-aa-20260804-145830-repeat
```

## Exact trajectory comparison

The strongest finding is exact telemetry repeatability:

```text
compared numerical values: 10,650
differing numerical values: 0
maximum absolute difference: 0.0
maximum relative difference: 0.0
numeric trajectory exact: true
discrete trajectory exact: true
validation exact: true
```

The two runs therefore matched exactly for all compared non-runtime numerical metrics and all discrete training identities, including block order, counters, clipping decisions, overflow state, loss, gradient norms, optimizer telemetry, LR, and validation.

No measurable training-trajectory nondeterministic floor was observed in the recorded metrics on this T4/software path.

## Loss and validation

Both runs produced the same values:

```text
training loss, update 1: 10.845867
training loss, update 50: 8.090633
minimum observed training loss: 8.090633
maximum observed training loss: 10.854832
validation loss: 7.915478
validation perplexity: 2,739.35
validation target tokens: 10,240
```

The loss trajectory remained finite and improved substantially over the 50-update prefix. These values remain engineering observations, not a model-quality claim.

## FP16, memory, and optimizer stability

Both runs matched exactly on the non-runtime stability metrics:

```text
GradScaler minimum / maximum: 65,536 / 65,536
overflow events: 0
overflow retries: 0
maximum allocated CUDA memory: 2,510,114,816 bytes
maximum reserved CUDA memory: 3,007,315,968 bytes
```

The optimizer telemetry was numerically identical across the two runs. No non-finite branch statistic or trajectory divergence was observed.

## Gradient clipping interpretation

Gradient clipping occurred on all 50 successful updates in both runs:

```text
clipped updates: 50 / 50
clipping fraction: 1.0
minimum pre-clip norm: 1.267016
maximum pre-clip norm: 2.680975
first-10 median: 1.399718
last-10 median: 1.385033
final norm: 1.344303
post-startup median: 1.723966
post-startup 95th percentile: 2.606312
```

This exceeds the protocol's provisional clipping-frequency failure band, so the clipping review flag remains active. However, the longer run changed the interpretation of the 20-update concern:

- the gradient norms did not continue increasing;
- the last-10 median was slightly lower than the first-10 median;
- the final norm was lower than both medians;
- loss continued improving;
- FP16 and optimizer statistics remained stable;
- the entire pattern was exactly reproduced in the A/A run.

The recipe is therefore reproducibly clipping-dependent but not showing runaway gradient growth over this 50-update window. No LR or clipping change is authorized from this result alone. Threshold review must decide whether a one-variable diagnostic is required before recovery testing or before the complete segment.

## Runtime observations

Runtime metrics were intentionally excluded from exact trajectory comparison. They differed between the two runs:

```text
reference mean throughput: 903.77 target tokens/s
repeat mean throughput: 1,000.97 target tokens/s
reference elapsed trainer time: 1,949.26 s
repeat elapsed trainer time: 1,657.20 s
```

The training math remained exact despite this wall-clock variation. Runtime thresholds should use distributions rather than exact equality.

## Checkpoint finding

All four expected checkpoints existed and passed structural verification:

```text
reference: step 25 and step 50
repeat: step 25 and step 50
```

However, complete checkpoint-tree hashes were not byte-identical:

```text
step 25 exact: false
step 50 exact: false
```

The checkpoint file counts matched, while byte sizes differed slightly. Because all compared training metrics and validation values were exactly identical, this does not currently demonstrate model-state divergence. It does demonstrate that complete serialized checkpoint trees contain some byte-level nondeterminism or run-specific metadata.

Before using bitwise checkpoint identity as a recovery requirement, the differing files and fields must be isolated. The next comparison should distinguish:

1. semantic tensor/state equality;
2. deterministic trainer counters, cursor, scaler, optimizer and RNG state;
3. expected run-specific or timestamp metadata;
4. raw archive or serialization byte differences.

The local resume gate must compare semantic state and the exact post-resume trajectory, while continuing to fail closed on any real state mismatch.

## Stage evidence

All stages returned exit code `0`.

```text
trainer-reference-50.log SHA-256: 6096f033c4c238a81c10297e1d6283893f89b7c9316a9c0886eb00bf8abf86ef
verify-checkpoints-reference.log SHA-256: 59adf787ab093bbbfa62b4ed7a9aeabf7ac5544d84f7ccccbb6c4022817b7327
trainer-repeat-50.log SHA-256: 840885b043a1331accb853a00ebaa9994e55c6e963464abdd6939f2b262ba667
verify-checkpoints-repeat.log SHA-256: 88a4bf3cfc1b60085d8095dbb86031fc61705c1474b43888cd39b724ba3a6e10
```

## Required next sequence

1. Freeze the observed exact metric-repeatability result and runtime distributions.
2. Inspect the step-25 and step-50 checkpoint trees to identify the source of byte-level differences and define semantic checkpoint comparison.
3. Freeze empirical warning/failure thresholds for loss, gradient norms, clipping, FP16 state, optimizer telemetry, memory, and throughput.
4. Decide explicitly whether universal but bounded clipping requires a one-variable diagnostic before recovery qualification.
5. Run the actual-process interruption test at the update-25 checkpoint boundary and resume from local state.
6. Compare resumed semantic state and the post-resume trajectory against the uninterrupted reference; because A/A metrics were exact, the default trajectory expectation is exact equality unless checkpoint analysis identifies a justified serialization-only exception.
7. Qualify private remote publication and empty-environment restore with verified next-block continuation and two-shard prefetch.
8. Authorize the complete 306-update one-pass segment only after the remaining gates pass.
