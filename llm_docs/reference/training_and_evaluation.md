# Training, Evaluation, and Checkpointing

_Last updated: 2026-07-31_

## Fixed constraints

- Initial accelerator: one NVIDIA T4.
- Initial context: 2,048 input tokens.
- Likely microbatch for larger trials: 1, with gradient accumulation.
- Train from random initialization.
- Checkpoint only at completed optimizer-step boundaries.
- Compare hybrid and all-MHA models with matched tokenizer, data mixture, parameter budget, training tokens, optimizer setup, data ordering, and evaluation protocol as closely as possible.

The intended first pass overlaps source streaming, preparation, local caching, Drive mirroring, and model training. With sufficient buffering, the T4 should be the steady-state bottleneck; data preparation must not starve it.

## Experiment ladder

1. Run the approximately 20M smoke model for implementation correctness.
2. Connect it to the schema-v2 dataset consumer and joint-checkpoint contract.
3. Validate forward pass, backward pass, generation, interruption, resume, and migration.
4. Benchmark memory, throughput, and mixer kernels on the T4.
5. Train the approximately 100M hybrid and a parameter-matched all-MHA baseline.
6. Scale only after stable loss, throughput, memory, and checkpoint evidence.

Smoke runs answer only whether the system works. The approximately 100M run is the first scale intended to compare architecture behavior.

## Required instrumentation

Every run should record:

- exact total and per-component parameter counts;
- optimizer step and source/training token counters;
- train and validation loss;
- gradient norm and clipping events;
- learning rate;
- loss-scaler state and overflow events when using FP16;
- peak allocated and reserved GPU memory;
- tokens per second and step time;
- data-wait time;
- mixer-specific kernel timings;
- GDN recurrent-state memory;
- MHA activation memory;
- checkpoint duration and size;
- save/load verification results;
- git commit, configuration hash, dataset hashes, tokenizer identity, schema version, and approved mixture-weight hash.

## Fixed-window joint checkpoint contract

Training must be pausable and safely resumable, including migration between VPS providers. A checkpoint binds exact trainer state to exact dataset-pipeline state.

### Trainer state

A complete trainer checkpoint must include:

- model weights;
- optimizer state;
- learning-rate scheduler state;
- FP16 scaler state when applicable;
- optimizer step and token counters;
- gradient-accumulation position;
- Python, framework, CUDA, and data-order RNG states;
- evaluation state.

### Pipeline state

The corresponding pipeline state must include:

- last consumed and last durable block IDs;
- validation state;
- durable source/work-plan cursor;
- queue, scheduler, rolling-mixture, and packer state;
- pending prepared sequences;
- finalized shard state;
- exact Google Drive manifest snapshot;
- hashes for configuration, source, code, tokenizer, schema, and approved mixture weights.

### Publication order

```text
finish optimizer step and pause consumption
→ finalize referenced shard tails
→ upload and verify Drive shards
→ atomically finalize local joint checkpoint
→ upload and read-back verify versioned private-Hub checkpoint
→ publish latest pointer
→ conditionally update best
→ resume training
```

No silent skip, unknown duplicate range, or model/data-cursor mismatch is acceptable. Bitwise-identical arithmetic after migration is best effort unless hardware and software environments match exactly.

## Baseline comparison contract

The all-MHA baseline should match the hybrid as closely as possible in:

- tokenizer and vocabulary;
- total parameters;
- depth or total compute, with unavoidable differences documented;
- SwiGLU FFN type and width;
- RMSNorm placement;
- RoPE configuration;
- training tokens;
- global token batch and optimizer setup;
- data ordering;
- checkpoint procedure;
- evaluation datasets and metrics.

Final claims must not be based on unmatched token counts, unmatched parameter counts, or different data orderings.

## Open training decisions

The following require explicit decisions and controlled tests:

- optimizer and optimizer-state strategy;
- learning-rate schedule, peak LR, warmup, and minimum LR;
- initialization and depth-dependent residual scaling;
- global token batch and gradient-accumulation schedule;
- precision policy on T4;
- gradient clipping;
- weight decay and parameter exclusions;
- checkpoint cadence;
- evaluation datasets and metrics;
- definition of the `best` checkpoint and metric direction;
- token budget for each architecture trial;
- whether repeated presentations up to 2T tokens are justified;
- post-training and instruction-tuning procedure;
- reasoning datasets and teacher model;
- final compute availability and release policy.

## Evaluation principle

Architecture comparisons must change one important variable at a time. Quality, stability, memory, throughput, and checkpoint behavior all matter. A configuration is not accepted merely because its nominal parameter count is attractive.
