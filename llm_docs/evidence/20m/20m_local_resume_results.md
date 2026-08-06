# Approximately-20M Local Interruption and Resume Results

_Last updated: 2026-08-04_

## Verdict

The actual-process update-25 interruption and local resume qualification completed successfully on 2026-08-04.

```text
status: passed_local_interruption_resume
authorization: remote_recovery_only
resume class: exact_local_resume
launch commit: 45d1da4a1ac3f18cf6ce02b8439672f10e2c8b4c
controller commit: 47a9b5056581c03d7c3c190d8c0c3fa158484ad1
GPU: Tesla T4, 15,360 MiB
started UTC: 2026-08-04T16:29:21.883920+00:00
finished UTC: 2026-08-04T17:24:56.412144+00:00
evidence directory: /kaggle/working/small-llm-local-resume-controller/small-llm-local-resume-20260804T162921Z
summary: /kaggle/working/small_llm_local_resume_summary.json
```

This passes the local recovery gate. It does not by itself authorize the complete 306-update run; private remote publication and empty-environment recovery remain required.

## Actual process interruption

The second trainer process emitted and completed `step-00000025`. The controller independently verified the checkpoint, then sent `SIGTERM` to the actual trainer process group.

```text
signal: SIGTERM
exit code: 143
expected non-zero exit: true
forced SIGKILL: false
process group gone: true
last consumed block: 24
next block after resume: 25
```

The checkpoint remained unchanged and hash-valid after termination.

## Exact resumed trajectory

A fresh process loaded `step-00000025` and executed updates 26 through 50. The uninterrupted reference and combined interrupted/resumed trajectory matched exactly.

```text
compared numerical values: 10,650
differing numerical values: 0
maximum absolute difference: 0.0
maximum relative difference: 0.0
numeric trajectory exact: true
discrete trajectory exact: true
validation exact: true
```

Loss, learning rate, gradient norms, clipping decisions, optimizer telemetry, FP16 scaler state, overflow counters, data order, block IDs, committed-token counters, and validation all matched.

## Semantic checkpoint equality

Raw checkpoint-tree hashes differed, as in the A/A test, but decoded semantic state was exact.

At both step 25 and step 50:

```text
tensors compared: 383
tensor elements compared: 54,184,616
semantic differences: 0
semantic exact: true
```

The comparison covered checkpoint JSON plus model, optimizer, scheduler, scaler, RNG, counters, and all serialized trainer state. This demonstrates that the raw tree-hash mismatch is serialization-byte variability rather than a training-state difference.

## Stable training observations

Both paths reproduced:

```text
training loss: 10.845867 -> 8.090633
validation loss: 7.915478
GradScaler: 65,536 throughout
FP16 overflow events: 0
overflow retries: 0
gradient clipping: 50 / 50 updates
```

The user had already accepted the bounded, exactly repeatable universal clipping pattern for this frozen 20M qualification recipe.

## Resulting authorization

The local interruption/resume gate is complete. The next and final pre-training qualification gate is:

1. publish a verified joint checkpoint to the private remote checkpoint repository;
2. start with no prior local checkpoint or data cache;
3. restore the checkpoint from the remote pointer;
4. download and hash-verify the two required Drive train shards;
5. continue from the exact next block;
6. match a local continuation trajectory and semantic checkpoint state exactly.

Only after that gate passes may the complete 306-update one-pass run be explicitly authorized.