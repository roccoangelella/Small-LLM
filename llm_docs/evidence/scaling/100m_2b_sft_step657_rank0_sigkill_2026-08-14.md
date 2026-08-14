# 100M/2B SFT step-657 rank-0 SIGKILL — 2026-08-14

The continuous two-T4 Kaggle SFT rerun resumed successfully past the durable step-500 checkpoint and completed optimizer step 657 before rank 0 was externally killed.

The last completed update reported:

```text
step: 657
block_id: 656
consumed_tokens: 21,183,524
loss: 2.03832745552063
learning_rate: 3e-05
gradient_norm: 0.8965684175491333
gradient_clipped: false
grad_scaler_scale: 16384
overflow_events_total: 2
overflow_retries: 0
peak_memory_bytes: 11,105,720,320
peak_reserved_memory_bytes: 11,693,719,552
target_tokens: 31,999
tokens_per_second: 2938.8181581966232
```

The launcher command for this run contained the bounded inline qualification arguments:

```text
--validation-blocks 1
--behavior-cases 2
```

and did not contain `--max-steps-this-session`.

The process then failed with:

```text
local_rank: 0
exitcode: -9
traceback: Signal 9 (SIGKILL) received by PID 223
```

Rank 1 was subsequently terminated by torchrun with SIGTERM. There was no Python exception, CUDA OOM, non-finite loss, failed gradient check, or optimizer overflow at the final completed step.

Because training reached step 657, the step-500 cadence had already returned successfully; checkpointing, remote publication, and bounded inline evaluation at step 500 therefore completed before ordinary training resumed.

Relevant implementation facts at the time of failure:

- rank 0 alone owned W&B and long cadence side effects;
- the hybrid optimizer was `InstrumentedHybridMuonAdamW` on both ranks;
- that instrumentation cloned each parameter before its update to derive effective-update statistics;
- every SFT optimizer step serialized the full nested `StepMetrics.as_dict()` payload to stdout and W&B on rank 0;
- the optimizer diagnostics were explicitly non-checkpointed.

ADR 0076 responds by disabling the qualification-only optimizer instrumentation for the Kaggle SFT child process, bounding glibc allocator arenas, and preserving the continuous exact-resume scientific protocol.
