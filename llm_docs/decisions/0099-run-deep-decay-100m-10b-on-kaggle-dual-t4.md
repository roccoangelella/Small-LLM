---
status: accepted
date: 2026-08-18
supersedes: 0095
---

# 0099 — Run the 100M/10B deep-decay continuation on Kaggle dual T4

## Context

ADR 0095 authorized the scientific trajectory that forks the exact original uncooled `100m-10b-data-001/checkpoints/step-00015500` state and changes its learning-rate scheduler. Its first implementation targeted one Beam GPU. The user has now chosen Kaggle's two Tesla T4 GPUs as the execution lane for that continuation.

The repository already has an exact-batch Kaggle DDP implementation qualified under ADR 0056. That implementation keeps the raw model checkpoint topology-neutral, synchronizes non-finite/overflow decisions before either replica can step, compensates DDP gradient averaging, and confines validation/checkpoint/Hugging Face/W&B side effects to rank zero. The 100M/10B trajectory, however, is frozen to a 64-sequence global optimizer block rather than the historical 16-sequence DDP qualification geometry.

There is also direct 100M/T4 hardware evidence: the first 100M/2B SFT dual-T4 start OOMed during a no-step backward prewarm at local shape `4x2048`, reaching 13.79 GiB allocated on a 14.56-GiB Tesla T4. That evidence explicitly ruled out per-rank execution microbatch four and moved the same 100M model to microbatch two. Because execution microbatch only slices the same accumulated global optimizer block, it can be changed for the Kaggle topology without changing the optimizer batch or token schedule.

## Decision

Move the authorized main deep-decay continuation to **Kaggle 2x Tesla T4 exact-batch DDP**.

Retain the complete ADR-0095 LR schedule unchanged:

```text
source checkpoint:            step-00015500
source targets:               2,031,616,000
source LR:                    3.0e-4

cosine settle end step:       17,789
cosine settle end targets:    2,331,639,808
settle LR:                    1.0e-4

power-law exponent:           ~1.6270515945
cooldown start step:          73,242
cooldown start targets:       9,599,975,424
cooldown start LR:            1.0e-5

final step:                   76,294
final targets:                10,000,007,168
final LR:                     5.0e-6
```

Preserve model, optimizer and optimizer state, scaler, RNG, data cursor, exact corpus order, frozen 16-block validation prefix, FP16, GDN-2, hybrid Muon+AdamW, and the 64-sequence global optimizer block. The exact source checkpoint remains microbatch four, but the Kaggle fork rewrites only the execution microbatch field to two while recomputing the checkpoint configuration hash; the model/optimizer/scaler/RNG/data state itself is not translated or reinitialized.

For two-T4 execution:

- use two replicated ranks with NCCL;
- split each ordered 64-sequence global block 32/32 across ranks;
- use microbatch two, therefore sixteen local microbatches per rank;
- retain the existing DDP `no_sync`/final-sync, loss scaling, clipping, synchronized overflow, raw-model checkpoint, and rank-zero-side-effect semantics from ADR 0056;
- pin PyTorch 2.10.0 + CUDA 12.8 wheels, Triton 3.6.0, and `fla-core==0.5.2`, matching the qualified Kaggle T4 runtime;
- treat DDP and finer accumulation reduction-order differences as numerically equivalent execution differences, not bitwise identity with one-GPU microbatch-four accumulation;
- publish live continuation checkpoints to the existing Hugging Face model-repository namespace every 250 successful updates on Kaggle so notebook/session loss costs at most one durability interval;
- fail closed unless a resumable **Kaggle microbatch-two** deep-decay checkpoint is valid, or, for the first segment, unless the exact original uncooled `step-00015500` source is the Hugging Face source pointer;
- CPU-stage and verify the checkpoint-aligned rolling 10B dataset window before starting the trainer;
- keep the continuation run ID `100m-10b-deep-decay-from-step15500` and its W&B/HF namespace unchanged across the provider migration.

The canonical human entry point is:

```bash
python kaggle/launch.py deep-decay --model 100M --tokens 10B
```

Rerunning the same command after a Kaggle interruption automatically restores the newest verified Kaggle continuation checkpoint. `--max-steps-this-session` may be used for bounded qualification or notebook segments.

## Consequences

### Positive

- Both Kaggle T4 GPUs are used while the 64-sequence scientific optimizer batch remains unchanged.
- The topology-neutral checkpoint format permits migration from the original one-GPU state without translating model keys or optimizer state.
- The execution microbatch is selected from direct 100M/T4 memory evidence rather than assuming that the 4090 microbatch fits a 14.56-GiB T4.
- The main human launcher remains `kaggle/launch.py`; the 10B rolling-data semantics stay in a dedicated continuation module instead of being forced into the finite-data profile table.
- Hugging Face remains the cross-session durability boundary, so Kaggle notebook restarts do not require Beam state.

### Limits and gates

- ADR 0056's parity/throughput qualification was measured on the 16-sequence 20M/2B geometry. The synchronization algebra is reused, but the first 64-sequence 100M/10B Kaggle segment remains a live execution gate rather than prior measured proof of throughput.
- The 100M/T4 microbatch-two evidence comes from SFT prewarm, not this exact pretraining path. It strongly constrains memory fit but does not replace the first live pretraining gate.
- A first bounded segment should confirm two Tesla T4 devices, correct `64 -> 32/32 -> microbatch 2` geometry, exact step-15,500 restore/fork, finite loss and gradients, correct LR, rank-zero-only W&B, and a verified remote checkpoint before relying on long unattended Kaggle sessions.
- Kaggle session limits may interrupt the run; exact resume is therefore operationally expected rather than exceptional.

## Implementation

- `kaggle/launch.py` exposes the dedicated `deep-decay` action.
- `kaggle/deep_decay_10b_from_15500.py` owns exact source/continuation restore, the execution-microbatch rewrite, rolling dataset staging, the frozen ADR-0095 scheduler, and two-T4 command construction.
- `kaggle/dual_t4_train_block64.py` reuses the qualified shared DDP engine with global block 64 and local microbatch two.

## Evidence

- [`../evidence/scaling/100m_2b_sft_t4_microbatch4_oom_2026-08-13.md`](../evidence/scaling/100m_2b_sft_t4_microbatch4_oom_2026-08-13.md)

## Links

- [`0095-decay-1e-4-to-1e-5-then-5e-6.md`](0095-decay-1e-4-to-1e-5-then-5e-6.md)
- [`0056-adopt-exact-batch-dual-t4-ddp-for-kaggle-only.md`](0056-adopt-exact-batch-dual-t4-ddp-for-kaggle-only.md)
- [`../runbooks/100m_10b_deep_decay_kaggle.md`](../runbooks/100m_10b_deep_decay_kaggle.md)
- [`../../kaggle/deep_decay_10b_from_15500.py`](../../kaggle/deep_decay_10b_from_15500.py)
