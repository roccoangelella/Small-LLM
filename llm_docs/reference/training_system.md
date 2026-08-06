# Training System

_Last updated: 2026-08-03_

## Decision

The schema-v2 consumer, first single-device trainer, and approximately-20M qualification path are implemented.

On 2026-08-03 the trusted launch policy was corrected to match the accepted T4 evidence:

- the obsolete GDN-2 parity-defect block was removed;
- trusted T4 FP16 GDN-2 runs resolve to chunk size 32;
- non-32 FP16 chunks require an explicit diagnostic override;
- hybrid whole-matrix Muon + AdamW is the default bounded-CLI optimizer;
- pure AdamW remains the explicit control.

This freezes the training-system boundary and selected optimizer architecture. It does not freeze the final training-cache block size, token budget, LR values, WSD horizons, checkpoint cadence, or acceptance thresholds. Those choices are listed in `20m_training_readiness.md`.

## Implemented boundary

The `trainer/` package provides:

- strict schema-v2 `context+1` block decoding;
- a bounded live-producer consumer compatible with `StreamCacheProducer`;
- deterministic immutable-shard reading for completed local caches;
- deterministic reading from a restored checkpoint's `drive_manifest.json` and prefetched `cache/` window;
- one prepared block as one atomic optimizer update;
- sequence microbatching inside that block;
- next-token cross-entropy over cropped semantic logits;
- pure AdamW and hybrid whole-matrix Muon + AdamW;
- fail-closed optimizer parameter routing;
- FP32, CUDA FP16 with `GradScaler`, and BF16 modes;
- global gradient clipping and bounded FP16 overflow retry;
- token-count constant and WSD-style schedules;
- per-optimizer-group LR multipliers;
- token-weighted validation loss and perplexity;
- uncached greedy generation for checkpoint qualification;
- model, optimizer, scheduler, scaler, counters, validation state, and RNG checkpoint state;
- integration with `dataset.src.joint_checkpoint.CheckpointCoordinator`;
- a bounded CLI at `python -m trainer`.

## Trusted GDN-2 launch policy

The corrected T4 harness cleared the recurrent-versus-chunkwise parity concern. The old CLI still described GDN-2 as blocked by a parity defect, so it was inconsistent with the accepted evidence.

The CLI now treats GDN-2 as trusted under the qualified execution geometry:

```text
architecture: gdn2_hybrid
precision: fp16
gdn_chunk_size: 32
initialization: normal
```

For `gdn2_hybrid + fp16`, omitting `--gdn-chunk-size` resolves to 32. A different FP16 chunk is rejected unless `--allow-unqualified-gdn2-chunk` is present.

For FP32 or BF16 GDN-2, the current model-family default remains 64 unless explicitly overridden. Transformer architectures reject a GDN chunk argument because it would otherwise be meaningless configuration noise.

The old `--allow-unqualified-gdn2` parity bypass has been removed. Diagnostic permission is now narrow and describes the actual unresolved condition: a non-qualified FP16 chunk size.

## Optimizer architecture

The default trainer optimizer is:

```text
--optimizer hybrid_muon_adamw
```

The explicit control is:

```text
--optimizer adamw
```

The hybrid path routes complete ordinary feature-transform matrices to Muon and keeps the tied embedding, norms, biases, GDN dynamics, and depthwise temporal filters on AdamW. Routing fails on a new unrecognized trainable parameter.

The first Muon recipe uses:

```text
Nesterov momentum: 0.95
whole-matrix orthogonalization
FP32 momentum and Newton-Schulz
8 aggressive + 2 stabilizing iterations
target update RMS: 0.18 qualification default
Muon weight decay: 0.1
shared token schedule
configurable Muon LR multiplier
```

The exact routing and recipe are stored in optimizer checkpoint state. Resume rejects drift.

See `optimizer_strategy.md` for the full parameter policy and `20m_training_readiness.md` for values still requiring discussion.

## Atomic block contract

A schema-v2 prepared block is the smallest durable trainer unit.

The trainer may divide its sequences into microbatches to fit the accelerator, but it accumulates gradients over the complete block and performs exactly one optimizer step. The consumer acknowledges the block only after that step succeeds. A failed or repeatedly overflowing step leaves the block unacknowledged and replayable.

For the hybrid optimizer, "one optimizer step" means both branches complete:

1. unscale gradients;
2. compute and apply one global clip;
3. apply Muon updates;
4. apply AdamW updates;
5. commit the token-count scheduler;
6. acknowledge the block.

A joint checkpoint is legal only when:

- no block is outstanding;
- gradient accumulation is at position zero;
- trainer and consumer agree on `last_consumed_block_id`;
- model, optimizer, scheduler, scaler, RNG, and consumed-block state describe the same completed boundary.

For a live producer, submitted queue contents must be drained before checkpointing. Queued blocks are already locally durable but are not yet represented in model state.

## Why microbatching does not reduce the effective token batch

This distinction matters enough to state directly.

Suppose a block contains 512 sequences at context 2,048 and microbatch size is 1. The trainer runs 512 forward/backward microbatches, but it divides each loss by the complete block's target-token count and accumulates all gradients before one optimizer step.

Therefore:

```text
microbatch size = accelerator memory unit
prepared-block size = optimizer/update/checkpoint unit
```

Changing `--microbatch-size` from 1 to 2 changes the number of forward/backward passes. It does not turn one 512-sequence block into smaller optimizer steps.

The accepted 10M operational cache used 512 sequences per block, which is about 1,048,576 target tokens per update. It remains valid dataset evidence but is unsuitable for the first optimizer-qualification cache. A smaller-block cache must be built after that geometry is selected.

The full calculation and the unresolved block-size choices are in `20m_training_readiness.md`.

## Immutable cache reader

`SchemaV2ShardReader` reconstructs prepared-block offsets from immutable shard metadata. It verifies:

- schema version and `context+1` geometry;
- split and contiguous block ranges;
- sequence and byte counts;
- safe relative paths;
- file size and SHA-256 before first use;
- exact cursor identity on resume;
- token IDs remain inside the semantic vocabulary.

Older schema-v2 manifests may not record `sequences_per_block`. In that case the value must be supplied explicitly. When the manifest records it, a conflicting CLI value is rejected.

This rejection is deliberate. The trainer cannot reinterpret one durable block as a different set of acknowledgeable optimizer units without changing dataset and checkpoint identity.

## Restored-cache reader

The empty-environment path uses the `drive_manifest.json` copied into a joint checkpoint.

Drive entries preserve block ranges, byte sizes, hashes, and original logical names. Dataset identity ignores publication-only fields and local filesystem paths, so a checkpoint created against the original cache can continue against a verified restored cache window.

When the next required shard was not prefetched, the reader fails clearly rather than skipping data or changing order.

## Trainer state

`TrainerEngine.state_dict()` contains:

```text
model parameters and buffers
complete optimizer state
hybrid optimizer recipe and routing identity when selected
learning-rate scheduler state
GradScaler state
global optimizer step
consumed non-padding target tokens
overflow counter
best validation loss
Python RNG state
PyTorch CPU RNG state
all CUDA RNG states
complete TrainerConfig
```

All tensors are moved to CPU before serialization. On restore:

- model loading is strict;
- complete trainer configuration must match;
- hybrid optimizer recipe and routing must match;
- optimizer tensors are moved back to the selected device;
- scheduler token count must equal the trainer token count;
- scaler and RNG states are restored.

## Optimizer checkpoint behavior

The hybrid optimizer is one `torch.optim.Optimizer` from the trainer's point of view. Its state includes:

- Muon FP32 momentum buffers;
- AdamW FP32 first and second moments;
- AdamW step counters;
- parameter groups and current effective LRs;
- group LR multipliers;
- exact routed parameter-name lists;
- Newton-Schulz coefficients and iteration counts;
- target update RMS;
- Muon momentum, LR multiplier, and weight decay;
- a versioned recipe identifier.

A checkpoint created with pure AdamW cannot resume as hybrid Muon + AdamW, or the reverse, because `TrainerConfig` is part of checkpoint identity.

## Token-count scheduler

The scheduler advances only after a successful atomic update, using committed non-padding target tokens.

For the hybrid optimizer:

```text
AdamW LR = base scheduled LR
Muon LR = base scheduled LR × muon_lr_multiplier
```

The scheduler now respects each parameter group's `lr_scale`. It no longer overwrites the Muon multiplier when preparing or committing a step.

Constant and WSD schedules remain available. The exact WSD horizons are open.

## FP16 overflow behavior

For CUDA FP16:

- forward and backward run under autocast;
- `GradScaler` scales losses;
- gradients are unscaled before clipping;
- a skipped optimizer step does not acknowledge the block;
- the scheduler does not commit skipped tokens;
- the same block may be retried up to the configured limit;
- repeated failure aborts with the block unacknowledged.

The GDN-2 chunkwise recurrence keeps its sensitive internal arithmetic in FP32. Muon momentum and Newton-Schulz also remain FP32.

## CLI

A trusted local-cache preflight now has this shape:

```bash
uv run --extra model python -m trainer \
  --dataset-dir /data/climbmix-training-pilot \
  --checkpoint-dir /data/small-llm-checkpoints \
  --steps <bounded-step-count> \
  --sequences-per-block <selected-training-block-size> \
  --model-size smoke \
  --architecture gdn2_hybrid \
  --gdn-chunk-size 32 \
  --initialization normal \
  --optimizer hybrid_muon_adamw \
  --device cuda \
  --precision fp16 \
  --microbatch-size 1 \
  --checkpoint-every-steps <selected-cadence> \
  --validation-blocks <selected-validation-blocks>
```

The explicit pure-AdamW control changes only:

```text
--optimizer adamw
```

Resume uses the identical semantic arguments plus:

```text
--resume step-XXXXXXXX
```

The values still shown as placeholders are intentionally unresolved. They are documented in `20m_training_readiness.md` rather than being silently promoted from current defaults.

## Local and remote checkpoints

The CLI currently saves verified local joint checkpoints through `CheckpointCoordinator`.

The repository also contains:

- two-phase remote checkpoint publication;
- checkpoint manifests and pointers;
- empty-environment restore;
- verified Drive-shard prefetch from the next unconsumed block.

The bounded trainer CLI does not yet construct the remote publisher or automatically publish each saved checkpoint. That is still an integration task before the complete approximately-20M qualification can pass. It is not a decision to change storage backend; Google Drive remains the selected durable data mirror and the existing remote checkpoint protocol remains the implementation target.

## Qualification completed in repository tests

CPU tests cover:

- strict prepared-block decoding and vocabulary bounds;
- live queue ordering and acknowledgement boundaries;
- immutable shard offsets, final partial blocks, hashes, and cursor resume;
- original-cache to restored-Drive-manifest identity equivalence;
- token-count schedule state;
- deterministic interrupted/resumed updates versus uninterrupted updates;
- checkpoint refusal with queued or outstanding work;
- validation and generation from trainer-owned model state;
- trusted CLI chunk selection and diagnostic rejection;
- hybrid optimizer routing completeness;
- Muon and AdamW FP32 state creation;
- scheduler preservation of the Muon LR multiplier;
- hybrid optimizer recipe/routing checkpoint round-trip.

These tests qualify the software contract, not T4 performance or long-run stability.

## Remaining operational gates

Before the approximately-20M qualification can be marked passed:

1. run the complete repository suite on the exact launch commit;
2. discuss and freeze the training-cache block geometry;
3. build and verify the smaller-block bounded training cache;
4. run the GDN-2 chunk-32 FP16 hybrid-Muon preflight;
5. record longer-run loss, gradients, scaler, clipping, memory, throughput, and data wait;
6. intentionally interrupt and resume at a local joint checkpoint;
7. wire remote publication into the trainer workflow;
8. restore into an empty environment and continue from the prefetched Drive cache window;
9. verify next block, counters, optimizer state, scaler, scheduler, RNG, and trajectory;
10. run validation and generation checks from trainer-produced checkpoints.

The complete 90B corpus and approximately-100M architecture comparison remain unauthorized until these gates pass.

## Open engineering choices

All remaining choices for the first run are centralized in `20m_training_readiness.md`:

- block size and bounded cache token target;
- base LR and Muon multiplier/update RMS;
- WSD horizons and LR floor;
- checkpoint/evaluation cadence;
- remote publication cadence and prefetch window;
- acceptance thresholds and instrumentation;
- number of updates and seeds.

Keeping these choices in one place prevents operational defaults from being mistaken for approved scientific values.
