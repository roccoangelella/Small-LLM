# Approximately-20M Kaggle T4 Preflight Results

_Last updated: 2026-08-04_

## Verdict

The exact-commit Kaggle T4 gates and the 20-successful-update constant-LR trainer
preflight completed successfully on 2026-08-04.

This is an **execution and integration pass**, with authorization limited to
post-preflight review and the remaining repeatability/recovery stages. It is not
a model-quality claim and does not authorize the complete 306-update one-pass
segment.

```text
summary status: passed_preflight
authorization: post_preflight_review_only
launch commit: 45d1da4a1ac3f18cf6ce02b8439672f10e2c8b4c
W&B run ID: 20m-t4-preflight-001
evidence directory: /kaggle/working/small-llm-qualification-controller/small-llm-qualification-20260804T135359Z
summary: /kaggle/working/small_llm_qualification_summary.json
started UTC: 2026-08-04T13:53:59.634233+00:00
finished UTC: 2026-08-04T14:07:10.115073+00:00
```

## Environment and identity

```text
GPU: Tesla T4, 15,360 MiB reported by nvidia-smi
compute capability: 7.5
Python: 3.13.14
PyTorch: 2.13.0+cu130
CUDA runtime: 13.0
cuDNN: 92000
controller commit: ac111dc6912cf1d0b4459ee071c095b30f21422b
launch worktree: detached, clean, exact commit 45d1da4
```

The attached private dataset was selected by exact identity:

```text
manifest SHA-256: 1e5ee8f372b77b6728288610dbe7cce74d833be21e53d1538bc5a890229b18bb
Drive manifest SHA-256: fbb29ee0d0102658e1274e39d6647cf56a6dcb685e0f566b1736847dcc4fbe84
run ID: 20m-qualification-dataset-001
```

## Gate results

All evidence-producing commands returned exit code `0`.

```text
offline suite: 229 passed, 1 expected live-remote skip
corrected T4 harness: passed
dataset full scan: passed=true, complete=true, problems=[]
qualification plan regeneration: exact accepted plan reproduced
trainer preflight: 20 successful updates, exit code 0
```

Stage log identities:

```text
offline-tests.log SHA-256: b5889c8476039b4a99717e0bca58502095d980db8b1b00584ccbec8f095fad47
t4-qualification.log SHA-256: 5bd020caebf393fef0a0b4cec87f8c52f0428eb821b3dc2e63fccdd1765c9569
dataset-full-scan.log SHA-256: 23dd508f96f39b24d9fbae9cec8efea840fdb06514da3fc1ee741363823b79be
qualification-plan.log SHA-256: 878f56e5f10bceccd27de18b84a9f15e8aafae7b0ddaa97a8d94d73a5bde51c6
trainer-preflight-20.log SHA-256: e23cfd896dbc8f53613b8b42e1fd9b1ec4d337f64753bb1b857e5f64c5d35e90
```

## T4 harness finding

The corrected harness qualified the frozen primary execution path:

```text
architecture: gdn2_hybrid
backend: pytorch_chunkwise
chunk size: 32
precision: FP16
context: 2,048
batch size: 1
status: pass
FP16 overflows: 0
mean measured step: 1,819.60 ms
throughput: 1,125.52 target tokens/s
peak allocated: 2,346.19 MiB
peak reserved: 2,456 MiB
```

The harness recommendation was `candidate`, not an automatic architecture or
hyperparameter change. The primary GDN-2 hybrid remains frozen.

## Twenty-update trainer preflight

The trainer consumed exactly 20 prepared blocks and 655,360 target tokens using
constant LR `3e-4`, normal initialization, hybrid Muon + AdamW, FP16, chunk size
32, microbatch size 1, and seed 17.

### Loss and validation

```text
training loss, update 1: 10.845867
training loss, update 20: 9.573909
minimum observed training loss: 9.573909
validation blocks: 1
validation target tokens: 10,240
validation loss: 9.240405
validation perplexity: 10,305.21
```

The short loss trajectory was finite and decreased throughout the preflight.
The validation value is an integration observation only; five validation
sequences and 20 updates do not support a model-quality conclusion.

### FP16 and overflow behavior

```text
GradScaler scale: 65,536 on every successful update
overflow events: 0
overflow retries: 0
exhausted retries: 0
```

### Throughput, memory, and input pipeline

Across all 20 updates:

```text
mean throughput: 1,066.12 target tokens/s
range: 1,026.47 to 1,075.82 target tokens/s
mean update wall time: 30.739 s
maximum allocated CUDA memory: 2,393.83 MiB
maximum reserved CUDA memory: 2,868 MiB
mean data wait: 0.00376 s
```

After excluding the first three startup updates, throughput had median
`1,068.14` target tokens/s, MAD `3.01`, and a 5th–95th percentile interval of
approximately `1,063.47`–`1,073.84`. Data wait was negligible relative to step
time.

### Optimizer-update statistics

Muon remained tightly normalized:

```text
Muon optimizer-direction RMS: approximately 0.18 throughout
Muon effective update/weight ratio, update 1: 0.00309444
Muon effective update/weight ratio, update 20: 0.00309331
```

AdamW effective update/weight ratios decreased without becoming non-finite:

```text
AdamW decay branch, update 1 to 20: 0.0147166 -> 0.0112170
AdamW no-decay branch, update 1 to 20: 0.000132995 -> 0.000085085
```

No branch-wide growth or non-finite optimizer statistic was observed.

### Gradient clipping finding

This is the main item requiring further qualification:

```text
clipped successful updates: 20 / 20
pre-clip global gradient norm range: 1.3763 to 2.7359
pre-clip global gradient norm at update 20: 2.7359
post-startup median pre-clip norm: 2.2147
post-startup 95th percentile: 2.6871
```

The loss improved while gradients and optimizer statistics remained finite, so
this did not trigger a hard correctness failure. However, 100% clipping exceeds
the protocol's provisional clipping warning/failure frequencies. The standard
recipe therefore has an **optimizer-stability review flag**. The full one-pass
run must not begin until the uninterrupted reference and A/A runs establish
whether the rising pre-clip norm is repeatable, bounded, and compatible with a
stable longer trajectory.

Do not silently change LR or clipping norm. Any diagnostic or replacement recipe
must alter one variable at a time and be recorded as a new decision.

## Checkpoint result

```text
checkpoint ID: step-00000020
checkpoint byte size: 216,852,669
checkpoint save time: 2.094 s
checkpoint path: /kaggle/working/small-llm-qualification-controller/small-llm-qualification-20260804T135359Z/checkpoints-preflight/step-00000020/checkpoint.json
```

## Non-blocking environment warnings

`uv` fell back from hardlinks to copies because cache and environment paths were
on different filesystems. PyTorch also warned that NumPy was not installed.
Neither warning caused a test or trainer failure, but adding NumPy to the model
extra and explicitly using `UV_LINK_MODE=copy` would remove noise in future
qualification logs. Such housekeeping must not change the frozen launch code
used for comparison runs unless a new exact commit is intentionally frozen.

## Required next sequence

1. Preserve the complete evidence directory and W&B run.
2. Run an uninterrupted reference segment of at least 50 successful updates
   from a known initial state using the precisely matched WSD prefix.
3. Run a second uninterrupted same-hardware A/A segment from the same initial
   state, seed, block order, and recipe.
4. Quantify the T4 nondeterministic floor and freeze warning/failure thresholds
   for loss, throughput, memory, overflow, clipping, gradient norms, and
   optimizer update statistics.
5. If clipping remains nearly universal or pre-clip norms continue growing,
   stop and run a separately labeled one-variable diagnostic before changing
   the frozen recipe.
6. Run an actual-process interruption at the planned local checkpoint boundary,
   normally update 25, then resume and compare with the uninterrupted reference
   using the A/A tolerance.
7. Qualify private remote publication and empty-environment restore, including
   exact next-block continuation and two-shard prefetch.
8. Only after all preceding gates pass, authorize the complete 306-update
   one-pass qualification segment.
