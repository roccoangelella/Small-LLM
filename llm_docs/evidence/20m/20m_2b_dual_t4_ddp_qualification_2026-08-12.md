# 20M / 2B exact-batch dual-T4 DDP qualification — 2026-08-12

## Scope

This evidence records the disposable Kaggle qualification that compared the existing single-T4 20M/2B optimizer update with a two-T4 `DistributedDataParallel` execution that preserves the exact 16-sequence global optimizer block.

The run used the real attached `20m-2b-dataset-001` schema-v2 blocks and did not publish production checkpoints or W&B state.

## Qualified runtime

```text
GPU: 2 x Tesla T4 / SM75
PyTorch: 2.10.0+cu128
CUDA runtime: 12.8
Triton: 3.6.0
fla-core: 0.5.2
precision: FP16 autocast with FP32 master parameters
optimizer: hybrid Muon + AdamW
global optimizer block: 16 sequences
microbatch: 4
```

The qualification used the same bounded six-representative-config Triton autotune policy on both execution paths. Cold compile/autotune work was excluded by one warmup block.

## Exact-batch DDP contract

```text
world size: 2
sequences per rank: 8
microbatches per rank: 2
first local microbatch: DDP no_sync()
second local microbatch: synchronized backward
gradient normalization: world_size * local_loss_sum / global_target_tokens
```

This compensates PyTorch DDP's gradient averaging and keeps one optimizer update scientifically equivalent to the serial 16-sequence block.

## Result

The report returned:

```text
status: passed
loss_parity: true
gradient_parity: true
parameter_parity: true
optimizer_parity: true
throughput: true
```

Warmed throughput across four measured blocks:

```text
single T4 median:       20,183.496142272958 target tok/s
dual T4 DDP median:    34,292.22117304134 target tok/s
median speedup:        1.6990228517059847x
promotion threshold:   1.60x
```

Measured warmed block times were approximately 1.616–1.627 seconds on one T4 and 0.951–0.969 seconds under two-T4 DDP.

Numerical deltas after the five-block trajectory were:

```text
maximum loss delta:                 4.76837158203125e-06
maximum gradient relative delta:    7.535586156002224e-06
parameter relative L2:              2.7796250767485542e-05
parameter maximum absolute delta:   0.0005217967554926872
optimizer relative L2:              0.00021572932322973847
optimizer maximum absolute delta:   1.425156369805336e-05
overflow retries:                   0 on both paths
```

All values were within the predeclared qualification thresholds. The speedup exceeded the 1.60x promotion gate by about 0.099x absolute.

## Interpretation

The result qualifies exact-batch two-T4 DDP as a materially faster Kaggle execution backend for this training geometry while preserving the intended optimizer batch, loss normalization, and trajectory within the established numerical tolerances.

The disposable harness did not exercise an asymmetric FP16 overflow. Production adoption therefore adds a stronger requirement than the harness itself: all ranks must synchronize forward/non-finite status before either optimizer replica can step, and checkpoints must serialize the unwrapped model so execution topology is not embedded in checkpoint keys.

The user subsequently authorized production adoption for Kaggle only. Modal remains a separate single-H100 execution path.
