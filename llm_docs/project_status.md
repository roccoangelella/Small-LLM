# Project Status

_Last updated: 2026-08-03_

## Current phase

The dataset software, exact-mixture calibration, model reference package, corrected T4 qualification harness, and first single-device training system are code-complete and CPU-tested.

The authenticated 10M dataset pilot passed. Corrected GDN-2 mathematical parity passed. The first trusted approximately-20M launch path has now been aligned with the accepted evidence:

- the obsolete parity-defect trainer gate is removed;
- trusted T4 FP16 GDN-2 defaults to chunk 32;
- non-32 FP16 chunks require an explicit diagnostic override;
- hybrid whole-matrix Muon + AdamW is implemented and is the default bounded-CLI optimizer;
- pure AdamW remains the explicit matched control;
- optimizer routing and recipe identity are checkpointed and fail closed.

The next phase is **integrated operational qualification**, not additional architecture selection.

The complete 90B dataset build and approximately-100M architecture comparison remain unauthorized.

## Authenticated 10M dataset pilot — passed

The bounded dataset acceptance pilot passed on 2026-08-02 at commit `e4776501d68e39746f8a75dcbb9c49515f215abd`, using the approved weight SHA-256 `76e82e22760adcac59c7294fe9bac11358f5a8b7a26035aae64c3f2e6fa1acb7` and real personal-Google-Drive durability.

Accepted evidence includes:

- run ID `climbmix-pilot-001`;
- target/minimum/maximum of 10,000,000 / 9,000,000 / 11,000,000 accepted source tokens;
- 10,000,662 completed accepted source tokens;
- 14,136 consumed source documents;
- seven immutable local shards and seven matching Drive objects;
- intentional interruption after the first durable checkpoint at 2,000,112 tokens and 2,814 documents;
- termination of the actual producer process group with status 143;
- deterministic resumed completion;
- schema-v2 full verification;
- completed-resume idempotence;
- fail-closed acceptance report status `PASS`.

Canonical evidence is under `/data/climbmix-ops`. The acceptance report SHA-256 is `b18decde4aa0e6e7376c3fecd3dda4406dee983f11224537cf73dd22a66bc00b`.

A previous attempt was rejected because it killed only a wrapper shell while the producer continued. It remains archived and excluded from accepted evidence.

## Dataset implementation

The production dataset system includes:

- deterministic pinned-source work plans and bounded range readers;
- structural validation and cluster-11 exclusion;
- exact source-token scheduling and rolling mixture accounting;
- context+1 packing and provenance;
- immutable shards with durability-before-consumer semantics;
- schema-v2 verification;
- durable interruption and deterministic resume;
- 80B minimum, 90B target, and 100B hard maximum;
- verified Google Drive mirroring before durable cursor advancement;
- configuration, schema, policy, weight, and source drift rejection;
- locking, disk preflight, retry policy, orphan cleanup, and empty-VPS restore primitives;
- fail-closed acceptance verification from concrete logs, snapshots, manifests, hashes, and exit codes.

The dataset subsystem remains frozen except for defects revealed by operational use and narrow trainer compatibility work.

## Exact-mixture calibration

The full-corpus scan completed successfully:

- 100 pinned source files;
- 7,457/7,457 work items;
- 1,987,970,304,099 source bytes;
- 553,315,056 records;
- 356,864,528,972 all-cluster source tokens;
- 351,792,454,745 accepted source tokens;
- 5,072,074,227 excluded cluster-11 tokens.

The approved exact production weight file remains:

```text
SHA-256 76e82e22760adcac59c7294fe9bac11358f5a8b7a26035aae64c3f2e6fa1acb7
```

The public calibration implementation and audit artifacts are in `roccoangelella/climbmix-token-mixture`.

## Model architecture

The initial family remains a dense decoder-only hybrid with:

- dominant GDN-2;
- periodic full gated MHA;
- repeating `[GDN-2, GDN-2, GDN-2, MHA]`;
- sequential pre-RMSNorm blocks and final RMSNorm;
- MHA-only RoPE;
- MHA QK-RMSNorm and output gating;
- dense SwiGLU;
- zero dropout;
- tied padded embeddings with semantic-logit cropping;
- 2,048-token context.

Implemented geometries:

```text
approximately-20M smoke hybrid: 20,637,592 parameters
approximately-100M hybrid: 101,252,280 parameters
matched Plan B / Plan C: 101,237,760 parameters
```

The 20M model is an engineering qualification model. The 100M scale is the first intended architecture comparison.

## Corrected T4 GDN-2 qualification

The schema-v2 T4 run passed all recurrent-versus-chunkwise parity cases:

- chunks 16, 32, and 64;
- FP32 and FP16-quantized recurrence inputs;
- zero-state training and bounded carried-state profiles;
- outputs, final state, and named gradients.

Full-model operational results at context 2,048 and microbatch 1 were:

- FP32 chunks 16, 32, and 64 passed;
- FP16 chunks 16 and 32 passed;
- FP16 chunk 64 produced non-finite values;
- FP16 chunk 32 reached approximately 1,291 tokens/s;
- Plan B reached approximately 17,260 tokens/s in the short ordinary-PyTorch benchmark.

The current trusted candidate is:

```text
gdn2_hybrid
ordinary PyTorch chunkwise backend
FP16
chunk 32
normal initialization
```

Chunk 64 remains the general model default for other modes, but it is not trusted for T4 FP16 training.

## Trainer launch correction

The old trainer CLI still blocked GDN-2 using wording from the invalid schema-v1 parity report. That inconsistency is now fixed.

The corrected behavior is:

- no broad parity-defect bypass;
- trusted `gdn2_hybrid + fp16` resolves to chunk 32;
- a different FP16 chunk requires `--allow-unqualified-gdn2-chunk`;
- transformer runs reject GDN chunk arguments;
- model configuration receives the resolved chunk explicitly.

This means the trusted CLI can no longer silently construct the known-failing FP16 chunk-64 path.

## Optimizer implementation

The selected first-run optimizer is now implemented:

```text
hybrid whole-matrix Muon + AdamW
```

Muon handles complete ordinary feature-transform matrices. AdamW handles the tied embedding, norms, biases, GDN dynamics, and structured depthwise filters.

The implementation includes:

- explicit parameter-role routing;
- rejection of overlap, omission, or unknown parameters;
- FP32 Nesterov momentum;
- FP32 ten-step hybrid Newton-Schulz;
- qualification target update RMS `0.18`;
- decoupled Muon weight decay;
- FP32 AdamW moments;
- shared token-count schedule with a Muon LR multiplier;
- one atomic optimizer step for both branches;
- recipe and exact routed-name identity in checkpoint state.

The bounded trainer CLI default is `hybrid_muon_adamw`. Pure AdamW remains available as the control.

The exact routing and mechanics are in `optimizer_strategy.md`.

## Training-block geometry finding

The accepted 10M operational cache used the dataset default of 512 sequences per block.

At context 2,048:

```text
512 × 2,048 = 1,048,576 target tokens per optimizer update
```

The trainer treats one complete prepared block as one atomic update. Microbatching only changes memory usage; it does not split the durable update unit.

The accepted cache therefore supplies only about ten optimizer updates. It is valid evidence for dataset operations, but it is not suitable as the first optimizer/stability qualification cache.

A second bounded cache with smaller blocks is required. The exact block size is intentionally not selected yet. The calculation, examples, and remaining choices are in `20m_training_readiness.md`.

## Training system

The trainer currently provides:

- immutable schema-v2 readers;
- restored Drive-manifest/cache-window readers;
- atomic prepared-block updates;
- internal microbatching;
- semantic next-token loss;
- hybrid Muon + AdamW and pure AdamW;
- FP32, FP16, and BF16 modes;
- clipping and bounded overflow retry;
- constant and WSD token schedules;
- validation and generation;
- complete trainer/RNG state;
- local joint checkpoint save/load;
- deterministic CPU interruption/resume tests;
- a bounded CLI.

The remote publication and empty-environment restore primitives exist in the repository. The trainer CLI still needs to wire remote checkpoint publication into the live workflow before the complete 20M qualification can pass.

## Immediate next steps

1. Run the complete offline suite on the exact implementation commit.
2. Discuss and freeze the training-cache `sequences_per_block` and bounded token target.
3. Build and fully verify that smaller-block cache.
4. Run a short T4 GDN-2 chunk-32 FP16 hybrid-Muon preflight.
5. Run a longer segment and record loss, gradients, scaler, clipping, memory, throughput, and data wait.
6. Interrupt the actual trainer process and resume from a local joint checkpoint.
7. Wire and qualify remote joint-checkpoint publication.
8. Restore into an empty environment and continue from the prefetched Drive cache window.
9. Validate generation and held-out loss from trainer-produced checkpoints.
10. Run the matched pure-AdamW control.
11. Only then authorize the approximately-100M comparison.

## Remaining engineering choices

The unresolved first-run choices are centralized in `20m_training_readiness.md`:

- training-cache block size and token target;
- base LR;
- Muon LR multiplier and whether target RMS stays `0.18`;
- WSD warmup/stable/decay horizons;
- checkpoint and validation cadence;
- remote publication cadence and restore prefetch window;
- acceptance thresholds;
- instrumentation completion;
- number of updates and seeds.

These are deliberately left open for discussion. They must not be inferred from CLI defaults.

## Dataset-scale constraints before 90B

The 10M pilot also exposed three separate production-scale constraints:

- resume currently replays prior documents to verify the cursor, so late-cursor recovery must be measured or redesigned;
- the current disk preflight requires about 222.3 GiB for 90B and 247.0 GiB for 100B, while the pilot VPS had about 95 GiB free;
- the 10M operational sample represented only seven of nineteen accepted clusters and used a permissive rolling-mixture-error bound.

A production orchestrator must also manage and verify the complete producer process group.

These constraints do not block the bounded 20M training cache. They do block the complete 90B launch.

## Decisions no longer open

Frozen choices include:

- pinned source revision and cluster policy;
- GPT-2 token IDs and EOD token;
- context+1 packing;
- exact empirical mixture and approved weight hash;
- Google Drive as durable mirror;
- PyTorch;
- the geometry-scalable model family;
- GDN-2-dominant 3:1 pattern;
- full MHA in attention layers;
- QK-RMSNorm and attention output gating;
- pre-RMSNorm/final RMSNorm;
- MHA-only RoPE;
- dense SwiGLU;
- zero dropout;
- tied padded embeddings;
- initial 2,048 context;
- smoke and approximately-100M geometries;
- Plan B and Plan C ordering;
- matched transformer FFN widths;
- schema-v2 atomic-block acknowledgement;
- joint checkpoint binding of exact consumed block and complete trainer/RNG state;
- trusted T4 FP16 GDN-2 chunk 32 for the bounded qualification path;
- hybrid whole-matrix Muon + AdamW as the default first-run optimizer;
- pure AdamW as the required control.

Scale and recipe values must still follow measurements rather than parameter labels.
