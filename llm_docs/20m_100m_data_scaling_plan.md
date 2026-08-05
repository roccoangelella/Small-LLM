# 20M Model / 100M-Token Data-Scaling Experiment

_Last updated: 2026-08-05 14:52 Europe/Rome_

## Decision

The next authorized experiment keeps the completed approximately-20M model family and increases the finite training dataset from approximately 10M to approximately 100M accepted source tokens.

The current experiment does **not** authorize a 1B-, 10B-, or 100B-token run and does not enlarge the model. Future dataset-size experiments should normally use approximately logarithmic boundaries such as 10M, 100M, 1B, 10B, and the final approximately-90B production scale, but every later boundary remains a separate decision.

## Scientific variable

The model, tokenizer, source corpus identity, source revision, accepted/excluded cluster policy, mixture weights, initialization family, seed, architecture, optimizer, context length, sequences per optimizer block, precision, and one-pass policy remain unchanged from the accepted 20M/10M qualification.

The intended scientific change is:

```text
accepted-source-token target: 10,000,000 -> 100,000,000
minimum accepted source tokens: 9,000,000 -> 90,000,000
maximum accepted source tokens: 11,000,000 -> 110,000,000
```

The model remains:

```text
parameters: 20,637,592
model size: smoke
architecture: gdn2_hybrid
pattern: [GDN-2, GDN-2, GDN-2, full gated MHA] x 2
context length: 2,048
GDN-2 chunk size: 32
initialization: normal
precision: FP16
seed: 17
```

The optimizer remains:

```text
hybrid whole-matrix Muon + AdamW
base LR: 3e-4
AdamW betas: 0.9 / 0.95
AdamW epsilon: 1e-8
weight decay: 0.1
Muon momentum: 0.95
Muon LR multiplier: 1.0
Muon target direction RMS: 0.18
Muon weight decay: 0.1
global gradient clipping norm: 1.0
WSD warmup/stable/cosine-decay policy
minimum LR ratio: 0.1
```

## Dataset and shard policy

The 100M finite dataset remains an immutable schema-v2 shard set. It is not converted into one monolithic training file and it is not streamed from Google Drive or Hugging Face during optimizer steps.

```text
context length: 2,048
stored tokens per sequence: 2,049
sequences per optimizer block: 16
target tokens per full optimizer update: 32,768
target shard size: 8 MiB
source-production durable checkpoint cadence: 20,000,000 source tokens
remote durability: required
passes: 1
implicit wraparound: forbidden
```

The source-production durable checkpoint cadence is scaled by 10x from the 10M build so the 100M producer retains the same approximate number of durable production boundaries. The immutable shard size stays at 8 MiB.

For Kaggle training, the complete approximately-200-MB uint16 dataset is attached and mounted once under `/kaggle/input`. The trainer reads local immutable shards sequentially. Google Drive remains the durable mirror and recovery source, but ordinary training does not download a shard over the network on every use.

## Increased microbatch execution

The effective optimizer batch is unchanged at one 16-sequence prepared block, or approximately 32,768 target tokens per full update.

The candidate execution grouping is:

```text
microbatch size: 4
microbatches per full block: 4
optimizer updates per block: 1
```

This changes only how the same block is split into forward/backward calls. It does not change the number of sequences or target tokens contributing to an optimizer update.

Because FP16 accumulation order and GPU kernels can differ with microbatch size, microbatch 4 is not silently assumed equivalent. The single Kaggle launcher must first run a fixed-prefix A/B qualification using microbatch 1 and microbatch 4 from the same seed and initialization. The full 100M run may start only when microbatch 4:

- consumes the same ordered blocks and target-token counts;
- produces finite losses and gradients;
- records zero exhausted overflow retries;
- remains within the T4 memory-headroom gate;
- stays within a predeclared loss/gradient agreement tolerance;
- provides a meaningful throughput improvement over microbatch 1.

If this gate fails, the launcher fails closed rather than silently reverting or changing another training variable.

## Schedule and persistence cadence

The exact update count and token horizons are derived from the completed, fully verified 100M manifest. The same fractional WSD policy is retained:

```text
warmup updates: max(16, ceil(5% of planned updates))
decay updates: ceil(20% of planned updates)
stable updates: all remaining updates
minimum LR ratio: 0.1
```

To preserve approximately the same number of operational observations and checkpoint trees across a run that is about 10x longer, cadence is scaled by 10x in optimizer steps:

```text
local checkpoint: every 250 successful updates
validation: every 500 successful updates
remote checkpoint publication: every 500 successful updates
```

A final validation, local checkpoint, and verified remote publication remain mandatory at the actual last update even when it is not a cadence boundary.

This is an operational scaling rule, not a change to model optimization. It keeps storage and recurring overhead comparable to the 10M qualification while retaining observations at similar fractions of training progress.

## Launch and evidence contract

The next implementation must expose one fail-closed Kaggle entry point that:

1. requires an NVIDIA T4 and the same secret surfaces used by the accepted run;
2. creates a clean detached worktree at one frozen launch commit;
3. finds the attached 100M dataset and rejects every nonmatching profile;
4. performs a literal full dataset scan;
5. verifies the Drive manifest against every local shard identity;
6. regenerates the exact one-pass block and WSD plan;
7. runs the microbatch-1 versus microbatch-4 gate;
8. starts the full microbatch-4 run from a fresh initialization only after the gate passes;
9. records W&B telemetry, local checkpoints, validation, and private remote publication;
10. writes durable logs, exit codes, hashes, the exact trainer command, and a final summary under `/kaggle/working`;
11. fails closed on identity, numerical, memory, overflow, block-order, checkpoint, validation, or publication errors.

The microbatch probes are diagnostics only. Their checkpoints are isolated and must never be reused as the full-run initialization.

## Interpretation boundary

This 100M-token experiment is a data-scaling experiment for the already-qualified 20M model. It is intended to measure how much the 10M checkpoint's weak factual retrieval, topic drift, repetition, and short-range coherence improve when the same model receives approximately 10x more distinct training data.

It is not the approximately-100M-parameter architecture comparison. Model enlargement remains a later, separately authorized stage.
