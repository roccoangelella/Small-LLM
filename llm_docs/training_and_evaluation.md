# Training and Evaluation

_Last updated: 2026-07-31_

## Current fixed constraints

- Initial accelerator: one NVIDIA T4.
- Initial context: 2,048 input tokens.
- Likely microbatch: 1 for larger trials, with gradient accumulation.
- Train from random initialization.
- Use the same tokenizer, data mixture, parameter budget, training tokens, optimizer setup, and evaluation protocol when comparing hybrid and all-MHA architectures.
- Checkpoint only at completed optimizer-step boundaries.

## Experiment ladder

1. Run the approximately 20M smoke model for implementation correctness.
2. Connect it to the schema-v2 dataset consumer and joint checkpoint contract.
3. Validate interruption, resume, generation, and migration.
4. Benchmark the approximately 100M hybrid and a parameter-matched all-MHA baseline.
5. Scale only after stable loss, throughput, and memory evidence.

## Required instrumentation

Every run should record:

- exact total and per-component parameters;
- optimizer step and source/training token counters;
- train and validation loss;
- gradient norm and clipping events;
- learning rate;
- loss-scaler state and overflow events when using FP16;
- peak allocated and reserved GPU memory;
- tokens per second and step time;
- data-wait time;
- mixer-specific kernel timings;
- checkpoint duration and size;
- git commit, configuration hash, dataset hashes, and approved mixture-weight hash.

## Open training decisions

The following still require explicit decisions and controlled tests:

- optimizer and optimizer-state strategy;
- learning-rate schedule, peak LR, warmup, and minimum LR;
- initialization and depth-dependent residual scaling;
- global token batch and gradient-accumulation schedule;
- precision policy on T4;
- gradient clipping;
- weight decay and parameter exclusions;
- checkpoint cadence;
- evaluation datasets and metrics;
- definition of `best` checkpoint;
- token budget for each architecture trial;
- whether repeated presentations up to 2T tokens are justified;
- post-training and instruction-tuning procedure.

## Evaluation principle

Early smoke runs answer only whether the system works. The approximately 100M run is the first scale intended to compare architecture behavior. Final claims must not be based on unmatched token counts, unmatched parameter counts, or different data orderings.
