# Approximately-20M Training Readiness

_Last updated: 2026-08-03_

## Purpose

The approximately-20M model is an end-to-end engineering qualification model. It is meant to prove that the selected architecture, optimizer, schema-v2 data path, checkpoint contract, interruption/resume behavior, validation, generation, and remote recovery all work together on the target NVIDIA T4.

It is not large enough to support meaningful architecture-quality claims. Passing this stage authorizes the approximately-100M comparison; it does not validate the final large-run recipe.

The detailed execution stages, instrumentation, threshold derivation, and pass/fail rules are in `20m_qualification_protocol.md`.

## Decisions now fixed

### Trusted GDN-2 execution

The corrected T4 qualification passed mathematical parity for chunk sizes 16, 32, and 64, but full-model FP16 execution passed only at chunks 16 and 32. Chunk 64 produced non-finite values under autocast.

The trusted approximately-20M T4 path therefore uses:

```text
architecture: gdn2_hybrid
precision: fp16
GDN-2 backend: ordinary PyTorch chunkwise
GDN-2 chunk size: 32
initialization candidate: normal
```

The model family's general default remains chunk 64 for non-FP16 or diagnostic use. The trainer CLI resolves trusted `gdn2_hybrid + fp16` runs to chunk 32 and rejects another chunk unless the operator explicitly marks the run diagnostic.

### Optimizer architecture

The first integrated approximately-20M run uses the documented hybrid optimizer, not pure AdamW:

```text
ordinary feature-transform matrices: whole-matrix Muon
embeddings, norms, biases, dynamics, and structured temporal filters: AdamW
```

The selected implementation uses:

```text
Nesterov momentum: 0.95
Newton-Schulz arithmetic: FP32
iterations: 8 aggressive + 2 stabilizing
aggressive coefficients: (3.4445, -4.7750, 2.0315)
stabilizing coefficients: (2.0, -1.5, 0.5)
Muon target update RMS: 0.18
Muon weight decay: 0.1
AdamW weight decay: 0.1
shared token-count schedule
Muon LR multiplier: 1.0
```

The pure-AdamW path remains mandatory as the matched control, but it is not the default launch optimizer.

### Parameter routing

Muon receives complete logical two-dimensional matrices only:

- every SwiGLU gate, up, and down projection;
- MHA Q, K, V, output-gate, and output projections;
- GDN-2 Q, K, V, erase, write, decay, output-gate, and output projections.

AdamW receives the currently classified exception roles:

- the tied token embedding / prediction matrix;
- every RMSNorm scale;
- every bias;
- GDN-2 `A_log` and `dt_bias`;
- GDN-2 depthwise Q/K/V convolution kernels.

Routing fails closed. Every trainable parameter must be assigned exactly once, and any future parameter requires an explicit Muon or AdamW classification. A new unrecognized parameter aborts optimizer construction rather than silently entering a generic group.

### Initial T4 update geometry

The first approximately-20M training-qualification dataset uses:

```text
context_length: 2,048
sequences_per_block: 16
microbatch_size: 1
target tokens per optimizer update: approximately 32,768
```

This is a project decision, not an example or inherited default.

At the measured short-run GDN-2 throughput of approximately 1,291 target tokens/s, a 32,768-token update is expected to take roughly 25 seconds before data and checkpoint overhead. A 10M-training-token dataset would provide approximately 305 updates. That is enough update cadence to inspect loss, gradient clipping, FP16 scaler behavior, Muon statistics, checkpointing, interruption, and resume while remaining practical on a single T4.

The value `16` is specific to the first T4 qualification profile. It does **not** replace the dataset-production CLI default of 512 sequences per block. Production storage/buffering geometry and training update geometry remain separate decisions.

The finite training dataset must be built with `--sequences-per-block 16`. Its manifest records that geometry as dataset identity. The trainer launch must also pass `--sequences-per-block 16`, causing a dataset built with a different block size to fail closed rather than silently changing the effective batch.

After the initial profile passes, the first batch-growth comparison is 16 versus 32 sequences per block. Growth is not automatic and requires a separate decision based on stability, throughput, and optimizer statistics.

### Standard hyperparameter baseline

The approximately-20M engineering qualification uses a conservative standard baseline rather than a broad hyperparameter search:

```text
base learning rate: 3e-4
AdamW beta1 / beta2: 0.9 / 0.95
AdamW epsilon: 1e-8
AdamW weight decay: 0.1
Muon momentum: 0.95
Muon LR multiplier: 1.0
Muon target update RMS: 0.18
Muon weight decay: 0.1
global gradient clipping norm: 1.0
seed: 17
```

The short failure-detection preflight uses constant `3e-4`.

The longer qualification uses the token scheduler already implemented by the trainer:

```text
warmup: max(16 updates, 5% of planned updates)
stable: all remaining updates before final decay
cosine decay: final 20% of planned updates
minimum LR ratio: 0.1
```

Exact warmup/stable/decay token horizons are derived after the finite dataset source-token envelope is selected and the verified manifest establishes the planned update count. Those exact integers are then frozen in the launch configuration and checkpoint identity.

This baseline is structurally consistent with public post-2025 Muon recipes, especially DeepSeek-V4's hybrid optimizer, `0.9/0.95` AdamW betas, `0.1` weight decay, `0.95` Muon momentum, `0.18` update RMS, and warmup/stable/cosine structure. Kimi K3 supports cosine scheduling and Muon-family optimization but does not publish a complete small-model numeric recipe. Kimi K3 per-head Muon is not adopted because it would be a separate optimizer-mechanics experiment.

### Checkpoint and evaluation cadence

The provisionally approved cadence is:

```text
local joint checkpoint: every 25 successful updates
validation: every 50 successful updates
remote joint-checkpoint publication: every 50 successful updates
```

This remains conditional on measured overhead. The intended aggregate recurring overhead budget is at most 5% of training wall-clock time, subject to confirmation and freezing on the exact T4 path.

If validation plus remote publication exceeds the frozen budget, those operations may move to every 100 updates. Local recovery checkpoint cadence does not widen automatically.

## Terminology: finite qualification dataset

Earlier notes used the phrase **bounded cache**, which was easy to misread. The preferred term is now **finite qualification dataset**.

It means:

- the dataset producer has an explicit accepted-source-token target, minimum, and hard maximum;
- it stops at a defined completion point far below the full 90B production target;
- accepted text is tokenized and packed into immutable schema-v2 shard files;
- the trainer consumes those prepared files without retokenizing the source.

It does not refer to context length, optimizer batch size, epoch count, circular buffering, or automatic repetition.

Source-token target and number of passes are independent:

```text
training tokens consumed
≈ usable target tokens in the finite dataset × trainer passes
```

The first qualification should use one pass unless another decision explicitly authorizes repetition. The trainer must not silently wrap to the beginning of the dataset.

The source-token target/minimum/maximum for the new finite training dataset are not fixed yet. The previously discussed 10M/9M/11M envelope remains a proposal.

## Why the accepted operational 10M dataset is not the training dataset

The accepted authenticated 10M run is valid evidence for source reading, exact mixture scheduling, immutable shards, Drive durability, interruption, resume, schema verification, and idempotence.

It used the dataset CLI default:

```text
sequences_per_block = 512
context_length = 2,048
```

The trainer's atomic contract says that one prepared block is one optimizer update. Microbatching splits the block only to fit memory; it does not change the effective token batch or allow a checkpoint in the middle.

Ignoring the small number of padding tokens:

```text
target tokens per optimizer update
= sequences_per_block × context_length
= 512 × 2,048
= 1,048,576
```

The accepted dataset contains about 10M train tokens, so it provides only about ten optimizer updates. At the current ordinary-PyTorch GDN-2 throughput, each update would also be very long. That is poor geometry for debugging scaler behavior, clipping, schedules, checkpoint cadence, and loss trajectory.

The accepted dataset is therefore not defective. It was built for operational dataset acceptance. Training qualification needs a second finite dataset with the same approved source, tokenizer, weights, and schema but `sequences_per_block=16`.

The manifest records block geometry as part of dataset identity. The trainer must not pretend that a 512-sequence block is a set of independently acknowledgeable smaller blocks because that would change the durable update/checkpoint contract after the data was built.

## Update-geometry reference

At context 2,048:

| Sequences per block | Approx. target tokens/update | Approx. updates in 10M train tokens | Status |
|---:|---:|---:|---|
| 8 | 16,384 | 610 | smaller debugging candidate |
| 16 | 32,768 | 305 | **selected initial T4 profile** |
| 32 | 65,536 | 152 | first later growth comparison |
| 64 | 131,072 | 76 | not selected initially |
| 512 | 1,048,576 | 9–10 | operational-dataset default, not training geometry |

## Threshold policy

The user approved an empirical, fail-closed qualification method rather than arbitrary final limits.

### Hard gates

Hard correctness gates are derived from the system contract and are never relaxed by statistics. Examples include:

- finite loss, gradients, updates, parameters, optimizer state, scheduler state, and validation/generation values;
- complete and exclusive optimizer routing;
- exact source, tokenizer, schema, dataset, model, optimizer, schedule, and weight-file identities;
- no skipped, duplicated, reordered, or prematurely acknowledged prepared block;
- checkpoint integrity and atomic-block boundaries;
- exact next-block and counter restoration;
- verified remote checkpoint and shard objects;
- termination of the real trainer process group during interruption tests.

### Empirical gates

Hardware, overhead, optimizer-distribution, and numerical-resume thresholds are derived from:

1. the standard short T4 preflight;
2. an uninterrupted reference segment;
3. an A/A repeatability control on the same T4;
4. a controlled local interruption/resume comparison;
5. remote publication and empty-environment recovery.

The baseline run must pass all hard gates before it may define acceptable distributions.

Provisional targets include:

```text
steady-state throughput: >= 90% of frozen baseline median
data wait: < 5% of wall-clock training time
recurring checkpoint/validation/blocking-publication overhead: <= 5%
memory headroom: >= 10%, preferably >= 1.5 GiB
post-warmup skipped candidate updates: <= 1%
clipping frequency > 20%: warning
sustained clipping frequency > 50%: failure
```

Final measurement windows, robust statistics, overflow behavior, Muon/AdamW update distributions, loss-runaway detector, and numerical resume tolerance are calculated from the preflight evidence and committed before the longer qualification segment.

The complete protocol is in `20m_qualification_protocol.md`.

## Instrumentation required

The current trainer records loss, base LR, gradient norm, throughput, overflow retries, and peak allocated memory. Before threshold calibration it must also record:

- current GradScaler scale;
- overflow and skipped-update events;
- whether clipping occurred;
- per-optimizer-branch gradient norms;
- Muon update RMS and update-to-weight statistics;
- AdamW update-to-weight statistics;
- CUDA reserved memory;
- data wait separately from compute time;
- checkpoint save/load duration and byte size;
- validation duration and token count;
- remote publication blocking and completion times;
- exact git, model, optimizer-routing, dataset, schema, and weight identities in one run summary.

## Engineering choices still open

### Finite dataset scope

- source-token target, minimum, and hard maximum;
- expected one-pass optimizer-update count;
- shard size;
- prepared-block queue and cache head-start settings;
- whether the first dataset is completed before training or later qualified in live producer/trainer overlap mode.

The initial `sequences_per_block=16` choice is fixed and is no longer open.

### Validation and recovery details

- exact fixed validation slice size and block IDs;
- definition of the best checkpoint;
- number of Drive shards prefetched during empty-environment restoration;
- exact asynchronous/deferred publication implementation.

### Empirical qualification outputs

- final warning/failure values calculated from T4 measurements;
- exact resume tolerance derived from A/A repeatability;
- whether the engineering qualification needs more than seed `17`.

The optimizer family, routing, Newton-Schulz recipe, learning rate, AdamW/Muon scalar hyperparameters, schedule ratios, clipping norm, FP16 model execution, FP32 optimizer arithmetic, seed, and 32,768-token update geometry are selected.

## Required launch sequence

The implementation and documentation changes do not themselves constitute a passed training run. The remaining sequence is:

1. complete the missing instrumentation;
2. run the complete offline suite on the exact launch commit;
3. select the finite qualification dataset source-token envelope and operational settings;
4. build and fully verify a separate dataset with `context_length=2048` and `sequences_per_block=16`;
5. derive exact schedule token horizons from the verified manifest;
6. launch the trainer with explicit dataset-geometry and recipe identity assertions;
7. run the short constant-LR GDN-2 chunk-32 FP16 hybrid-Muon preflight;
8. run the uninterrupted reference and A/A repeatability segments;
9. freeze the empirical threshold table in project memory;
10. terminate the actual trainer process and qualify local resume;
11. publish a verified joint checkpoint and qualify empty-environment recovery;
12. run the longer one-pass qualification segment;
13. run validation and deterministic generation checks from trainer-produced checkpoints;
14. record the complete report and update `project_status.md`.

The approximately-100M architecture comparison and complete 90B dataset production remain unauthorized until this ladder passes.