# Training System

_Last updated: 2026-08-01_

## Decision

On 2026-08-01 the user authorized implementing the schema-v2 consumer, the first single-device trainer, and the approximately-20M end-to-end qualification path without further supervision unless a real scientific or operational decision was required.

This authorization freezes the **training-system boundary**, not the final training recipe. AdamW, FP16, the current learning rate, clipping value, schedule shape, microbatch size, and initialization exposed by the CLI remain configurable qualification defaults. They are not approved substantive-run hyperparameters until measured on the T4 and compared in controlled pilots.

## Implemented boundary

The `trainer/` package now provides:

- strict schema-v2 `context+1` block decoding;
- a bounded live producer consumer compatible with `StreamCacheProducer`;
- deterministic immutable-shard reading for completed local caches;
- deterministic reading from a restored checkpoint's `drive_manifest.json` and prefetched `cache/` window;
- one prepared block as one atomic optimizer update;
- sequence microbatching inside that block;
- next-token cross-entropy over the model's cropped semantic logits;
- AdamW with the model package's explicit no-weight-decay exclusions;
- FP32, CUDA FP16 with `torch.amp.GradScaler`, and BF16 execution modes;
- global gradient clipping and bounded FP16 overflow retry;
- token-count constant and warmup/stable/decay schedules;
- token-weighted validation loss and perplexity;
- uncached greedy generation for checkpoint qualification;
- model, optimizer, scheduler, scaler, counters, validation state, and RNG checkpoint state;
- integration with `dataset.src.joint_checkpoint.CheckpointCoordinator`;
- a bounded CLI at `python -m trainer`.

## Atomic block contract

A schema-v2 prepared block is the smallest durable trainer unit.

The trainer may divide its sequences into microbatches to fit the accelerator, but it accumulates gradients across the complete block and performs exactly one optimizer step. The consumer acknowledges the block only after that step succeeds. A failed or repeatedly overflowing step leaves the block unacknowledged and therefore replayable.

A joint checkpoint is legal only when:

- no block is outstanding;
- gradient accumulation is at position zero;
- the trainer and consumer agree on `last_consumed_block_id`;
- the model update, optimizer state, scheduler state, scaler state, and RNG state all describe the same completed boundary.

For a live producer, submitted queue contents must be drained before checkpointing. This is deliberate: queued blocks are already locally durable but have not yet been incorporated into model state. The producer's own pipeline state is merged into the trainer checkpoint after its consumed cursor is checked.

## Immutable cache reader

`SchemaV2ShardReader` reconstructs prepared-block offsets from immutable shard metadata. It verifies:

- schema version and `context+1` geometry;
- split and contiguous block ranges;
- sequence and byte counts;
- safe relative paths;
- file size and SHA-256 before first use;
- exact cursor identity on resume;
- token IDs remain inside the semantic vocabulary.

Older schema-v2 manifests do not record `sequences_per_block`. The reader therefore requires that value explicitly for existing caches. Future manifests may include it, in which case a conflicting CLI value is rejected.

The restore path uses the `drive_manifest.json` copied into the joint checkpoint. Drive entries preserve the original block geometry and checksums. The dataset identity intentionally ignores publication-only fields and local filesystem paths, so a checkpoint made against the original cache can resume against a verified restored cache window. When the next required shard was not prefetched, the reader fails clearly rather than skipping data or changing order.

## Trainer state

`TrainerEngine.state_dict()` contains:

```text
model parameters and buffers
optimizer state
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

All tensors are moved to CPU before serialization. On restore, optimizer tensors are moved back to the selected device, model loading is strict, the complete trainer configuration must match, and the scheduler's committed-token count must equal the trainer's token counter.

The dataset coordinator still owns the atomic directory write, hashes, fsync, rename, remote publication, and empty-VPS restore protocol. The trainer remains only the framework-specific state adapter.

## Training policy surface

The implementation includes an AdamW baseline because the first integrated run needs a conservative, inspectable optimizer. It does not decide that AdamW is the final optimizer.

The current optimizer values are qualification defaults. Because the first T4 report found a GDN-2 recurrent/chunkwise parity defect, the CLI refuses `gdn2_hybrid` unless `--allow-unqualified-gdn2` is supplied explicitly for diagnosis. Plan B may be used to qualify trainer plumbing without changing the primary architecture decision.

The current policy values are:

```text
optimizer: AdamW
peak learning rate: 3e-4
betas: 0.9, 0.95
weight decay: 0.1
maximum gradient norm: 1.0
precision: FP16
microbatch size: 1
schedule: constant unless a complete WSD token horizon is supplied
```

Normalization scales and the explicit GDN dynamic/offset parameters already named by `model.accounting.optimizer_no_weight_decay_parameter_names` are excluded from decay. Every policy value is stored in checkpoints and resume rejects drift.

No optimizer, schedule, batch, initialization, checkpoint cadence, or token budget is frozen for the approximately-100M comparison yet.

## CLI

A bounded local-cache smoke run is started with:

```bash
uv run --extra model python -m trainer \
  --dataset-dir /data/climbmix-pilot \
  --checkpoint-dir /data/small-llm-checkpoints \
  --steps 10 \
  --sequences-per-block <pilot-block-size> \
  --model-size smoke \
  --architecture swa_hybrid \
  --device cuda \
  --precision fp16 \
  --microbatch-size 1 \
  --checkpoint-every-steps 5 \
  --validation-blocks 2
```

Resume the same run with the identical semantic arguments and:

```bash
--resume step-00000005
```

After an empty-VPS restore, point `--dataset-dir` at the restored `cache/`, point `--dataset-manifest` at the checkpoint's `drive_manifest.json`, and keep the same context, block geometry, model, and trainer arguments. The coordinator identities are read from the restored checkpoint metadata before loading opaque state.

## Qualification completed in repository tests

CPU tests cover:

- strict prepared-block decoding and semantic-vocabulary bounds;
- live queue ordering and acknowledgement boundaries;
- immutable shard offsets, final partial blocks, hashes, and cursor resume;
- original-cache to restored-Drive-manifest identity equivalence;
- token-count schedule state;
- deterministic interrupted/resumed updates versus uninterrupted updates;
- checkpoint refusal with queued or outstanding work;
- validation and generation from trainer-owned model state.

These tests qualify the software contract, not target-hardware performance.

## Remaining operational gates

The following still require the real dataset artefacts and accelerator environment:

1. approve the exact mixture report and weight-file SHA-256;
2. pass the authenticated bounded dataset pilot and completed-resume idempotence check;
3. diagnose the T4 recurrent/chunkwise parity and FP16 non-finite failures, then rerun qualification;
4. qualify trainer/checkpoint plumbing with Plan B against a verified schema-v2 pilot while the GDN-2 defect is isolated;
5. intentionally interrupt and resume it, then compare the next batch, step counters, and loss trajectory;
6. restore the same joint checkpoint into an empty environment and continue from the prefetched Drive shard window;
7. record starvation, throughput, peak memory, scaler behavior, checkpoint latency, and recovery behavior;
8. repeat the integrated run with GDN-2 only after parity is fixed, then screen the approximately-100M training policy.

The complete 90B corpus and substantive architecture comparison remain unauthorized until these gates pass.
