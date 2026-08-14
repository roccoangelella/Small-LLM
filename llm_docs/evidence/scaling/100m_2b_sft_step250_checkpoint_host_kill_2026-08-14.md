# 100M/2B SFT step-250 checkpoint host-memory kill — 2026-08-14

The checkpoint-first live rerun used the pinned two-T4 SFT runtime, per-rank microbatch 2, FP16, and the 4% SFT bundle. Training remained finite through optimizer step 250.

The final completed update reported:

```text
step: 250
block: 249
consumed loss-bearing targets: 8,044,261
loss: 2.110750436782837
gradient norm: 0.8772922158241272
gradient clipped: false
loss scale: 32768
peak allocated bytes: 8,345,722,880
peak reserved bytes: 11,697,913,856
targets/second: 3,663.904170480196
```

The new cadence telemetry then localized the failure to checkpoint serialization:

```text
phase: checkpoint:start
checkpoint_id: step-00000250
host_rss_bytes: 16,144,891,904
host_peak_rss_bytes: 16,158,162,944
cuda_allocated_bytes: 1,336,704,512
cuda_reserved_bytes: 11,697,913,856
```

Approximately 2.7 seconds later rank 0 exited with signal 9 (`SIGKILL`). Rank 1, which was correctly waiting on the Gloo control barrier, then reported `Connection closed by peer` and was terminated by torchrun. No `checkpoint:done`, `local_checkpoint`, publication, validation, or behavior event appeared.

This rules out the previous NCCL cadence timeout and localizes the current failure before evaluation and before remote publication. The old checkpoint implementation recursively converted every model and optimizer tensor with `detach().cpu().clone()` before pickle serialization, materializing a complete extra host-side tensor tree while rank 0 was already near the host-memory limit.

The remediation is ADR 0072: stream DDP trainer state with PyTorch serialization without a full pre-save CPU clone, retain exact optimizer/scheduler/scaler/RNG state and legacy pickle readability, release collectable host heap before memory-sensitive checkpoint I/O, and mmap streamed state when loading on CPU. The 100M/2B profile is pinned to the implementation before the next live run.

The next live qualification gate is successful step-250 durability in this order:

```text
checkpoint:start
checkpoint:done
local_checkpoint
publication:start
publication:done
remote_publication
validation:start
```

If evaluation fails after that point, step 250 is still an exact remotely durable resume boundary.
