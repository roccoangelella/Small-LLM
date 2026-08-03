# Approximately-20M Training Readiness

_Last updated: 2026-08-03_

## Purpose

The approximately-20M model is an end-to-end engineering qualification model. It is meant to prove that the selected architecture, optimizer, schema-v2 data path, checkpoint contract, interruption/resume behavior, validation, and generation all work together on the target T4.

It is not large enough to support meaningful architecture-quality claims. Passing this stage authorizes the approximately-100M comparison; it does not validate the final model recipe.

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

The model family's general default remains chunk 64 for non-FP16 or diagnostic use. The trainer CLI now resolves trusted `gdn2_hybrid + fp16` runs to chunk 32 and rejects another chunk unless the operator explicitly marks the run diagnostic.

### Optimizer architecture

The first integrated approximately-20M run uses the documented hybrid optimizer, not pure AdamW:

```text
ordinary feature-transform matrices: whole-matrix Muon
embeddings, norms, biases, dynamics, and structured temporal filters: AdamW
```

The first implementation uses the recorded DeepSeek-V4-style whole-matrix Muon mechanics:

```text
Nesterov momentum: 0.95
Newton-Schulz arithmetic: FP32
iterations: 8 aggressive + 2 stabilizing
aggressive coefficients: (3.4445, -4.7750, 2.0315)
stabilizing coefficients: (2.0, -1.5, 0.5)
qualification target update RMS: 0.18
Muon weight decay: 0.1
AdamW weight decay: 0.1
shared token-count schedule
configurable Muon LR multiplier
```

The pure-AdamW path remains mandatory as the matched control, but it is not the default launch optimizer.

### Parameter routing

Muon receives complete logical two-dimensional matrices only:

- every SwiGLU gate, up, and down projection;
- MHA Q, K, V, output-gate, and output projections;
- GDN-2 Q, K, V, erase, write, decay, output-gate, and output projections.

AdamW receives:

- the tied token embedding / prediction matrix;
- every RMSNorm scale;
- every bias;
- GDN-2 `A_log` and `dt_bias`;
- GDN-2 depthwise Q/K/V convolution kernels;
- any future parameter not explicitly admitted to Muon.

Routing fails closed. Every trainable parameter must be assigned exactly once, and a new unrecognized parameter aborts optimizer construction rather than silently entering a generic group.

## Why the accepted 10M cache is not the training cache

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

The accepted cache contains about 10M train tokens, so it provides only about ten optimizer updates. At the current ordinary-PyTorch GDN-2 throughput, each update would also be very long. That is poor geometry for debugging scaler behavior, clipping, schedules, checkpoint cadence, and loss trajectory.

The cache is therefore not defective. It was built for operational dataset acceptance, and its block geometry matches that purpose. Training qualification needs a second bounded cache with the same approved source, tokenizer, weights, and schema but a smaller `sequences_per_block`.

The manifest records block geometry as part of dataset identity. The trainer must not pretend that a 512-sequence block is a set of independently acknowledgeable smaller blocks, because that would change the durable update/checkpoint contract after the data was built.

## Block-size examples for the later decision

At context 2,048:

| Sequences per block | Approx. target tokens/update | Approx. updates in 10M train tokens |
|---:|---:|---:|
| 8 | 16,384 | 610 |
| 16 | 32,768 | 305 |
| 32 | 65,536 | 152 |
| 64 | 131,072 | 76 |
| 512 | 1,048,576 | 9–10 |

These are geometry calculations, not a recommendation frozen by this document. The actual block choice must balance update frequency, T4 step time, checkpoint granularity, data overhead, and the desired effective batch.

## Engineering choices deliberately left open

These choices are now written down rather than being hidden in CLI defaults. They require a separate discussion before the bounded training cache and final launch command are approved.

### Data/update geometry

- `sequences_per_block` for the training-qualification cache;
- total source-token target and expected number of optimizer updates;
- shard size and whether it should differ from the operational pilot;
- prepared-block queue and cache head-start settings for live producer/trainer overlap.

### Optimizer and schedule values

- base AdamW learning rate;
- Muon LR multiplier and whether target update RMS stays at `0.18`;
- WSD warmup, stable, and decay token horizons;
- minimum LR ratio;
- whether effective batch grows in stages during the same run or across separate runs.

The optimizer family, routing, Newton-Schulz recipe, clipping default `1.0`, FP16 model execution, FP32 Muon arithmetic, and seed `17` are already selected. The values above are tuning choices, not architecture choices.

### Checkpoint and evaluation policy

- checkpoint cadence in successful optimizer updates or committed target tokens;
- validation cadence and number of validation blocks;
- generation prompts and pass/fail checks;
- definition of the best checkpoint;
- remote joint-checkpoint publication cadence;
- number of Drive shards prefetched during empty-VPS restoration.

### Acceptance thresholds

- maximum allowed scaler reductions or overflow retries;
- acceptable clipping frequency;
- minimum sustained tokens/s and maximum data-wait fraction;
- peak allocated/reserved memory margin;
- local-resume and empty-VPS trajectory tolerances;
- minimum number of uninterrupted and resumed updates;
- whether the qualification result needs more than seed `17`.

### Instrumentation completion

The current trainer records loss, base LR, gradient norm, throughput, overflow retries, and peak allocated memory. Before the longer qualification segment it should also record:

- current GradScaler scale;
- whether clipping occurred;
- per-optimizer-branch gradient norms;
- Muon update-to-weight and update RMS statistics;
- CUDA reserved memory;
- data wait separately from compute time;
- checkpoint save/load duration and byte size;
- exact git, model, optimizer-routing, dataset, schema, and weight identities in one run summary.

## Required launch sequence

The implementation and documentation changes in this file do not themselves constitute a passed training run. The remaining execution sequence is:

1. run the complete offline test suite on the exact launch commit;
2. build and fully verify the bounded training-qualification cache after its block geometry is selected;
3. run a short GDN-2 chunk-32 FP16 hybrid-Muon preflight;
4. run a longer local segment, save a joint checkpoint, terminate the actual trainer process, and resume;
5. publish a joint checkpoint and continue from an empty environment using the prefetched Drive cache window;
6. run validation and generation checks from trainer-produced checkpoints;
7. record the complete report and update `project_status.md`.

The approximately-100M architecture comparison and complete 90B dataset production remain unauthorized until this ladder passes.
