# 2026-09-24: 2×T4 MoE global-block exact-resume probe

The user asked to use **both** T4s for the accepted run rather than treat the
second GPU as spare. The existing `kaggle/src/dual_t4_train.py` is a dense,
16-sequence FP16 DDP shim and cannot train the accepted 64E, Top-2 BF16 MoE:
its model output and router-state semantics differ. The new
`kaggle/moe_train_dual_t4.py` adapts the actual `MOE_model` CLI via torchrun,
without modifying the scientific model, optimizer, schedule, data identity,
checkpoint format, or HF/W&B run identity.

Each process restores the same complete HF step-272,500 checkpoint and
checkpoint-aligned v2 dataset. Rank 0 owns the rolling shard cache (download
and eviction), validation, local checkpoint, and external telemetry/publishing.
Rank 1 verifies files from that shared local cache and never evicts them.
The full 64-sequence optimizer block is read on both ranks and split 32/32;
DDP gradient averaging is compensated by a factor of two. Each of the eight
Quantile Balancing routers merges the per-expert top-M score frontier from
both ranks before setting its next bias and accounting counters. A
single-source CUDA RNG is duplicated across GPUs; the accepted model has no
dropout. This is a change in execution topology, not a new architecture.

**Live Kaggle measurements, isolated from the running W&B/HF writer:** the
notebook had two Tesla T4s, PyTorch 2.10.0+cu128, fla-core 0.5.2, and a local
verified copy of the real checkpoint at step 272,500. Both `torchrun` workers
completed step 272,501 with the real trainer CLI; rank 0 validated one held-out
block and saved a portable ~1.16 GB checkpoint. A *second torchrun* restored
that DDP-created checkpoint and completed step 272,502, again validating and
saving. Remote publishing and W&B were explicitly disabled in this probe;
the production HF/W&B run was untouched. Both ranks used the identical real
64-sequence block, BF16, Quantile Balancing, and frozen WSD schedule.

At microbatch **1 per rank**, step 272,501 reported 10,703 targets/s and
4,316,723,712 bytes peak allocated on rank 0. Validation CE on one block:
2.62083. At microbatch **2 per rank**, a separate update from the *same*
checkpoint reported 9,241 targets/s (5,249,380,352 bytes rank-zero peak),
so microbatch 1 remains the default. The earlier single-T4 isolated update
was ~7,051 targets/s; the measured warm training-step throughput gain is
~52%, not 2×. Launch and final-validation/checkpoint overhead are additional.
A separate two-rank replay from the DDP checkpoint at step 272,503 exercised
the nine-hour drain path with an artificially one-second wall budget: both
ranks exited together, validated and saved step 272,504, and emitted one
`drained` event instead of hanging at the next DDP barrier.

For a serial-vs-DDP comparison from the same step-272,500 checkpoint, the
single-T4 update had loss 2.72540307. The two-T4 update checkpoint contained
exactly the same absolute step (272,501) and consumed-token count
(35,717,251,072); all eight routers' next-step selection biases matched
**exactly**. The largest absolute weight difference across 190 state tensors
was 2.665e-5, on `blocks.0.ffn.gate_weight`, attributable to BF16/reduction
order rather than a different logical batch. The unit test also independently
checks a two-rank synthetic top-M merge against the exact global fourth-order
statistic.

**Limit:** this is offline training/validation/local-save/exact-resume, not a
live W&B `must` resume or HF upload. The original run still reported
`running`; sending this slower replica to the same latest pointer at the same
time would allow rollback even if its step count lags. The saved Kaggle
notebook has not been edited via its browser UI. Do not start two writers.
