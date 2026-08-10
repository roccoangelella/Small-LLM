# 100M Kaggle Console Output Decision

_Date: 2026-08-05 Europe/Rome_

## Decision

The official `python kaggle/run_20m_100m.py` entry point must show concise, human-readable progress instead of raw per-step JSON.

Console output is limited to meaningful phase and event lines:

```text
[environment] GPU identity
[setup] dependency and frozen-worktree phases
[dataset] full scan and exact-plan phases
[resume] restored checkpoint and block cursor
[probe mb=N] progress, loss, throughput, gradient state, and VRAM
[probe] final microbatch gate verdict
[train] session/global step, block, loss, LR, throughput, gradient state, and VRAM
[wandb] project, run ID, and mode
[validation] held-out loss, perplexity when available, and elapsed time
[checkpoint] local save and verified remote publication
[segment] final checkpoint boundary
```

## Invariants

- Raw trainer JSON remains byte-for-byte in the evidence log files.
- Microbatch parsing, segment verification, W&B logging, checkpoint publication, and exact resume continue to consume the raw logs.
- The pinned scientific launch commit remains `43190cb72443a2de290dc8e6f2c54f29d8dff501`.
- This is presentation-only and does not alter model, data, optimizer, schedule, seed, precision, batching, checkpoint, or resume behavior.
- A process already running before this change retains the old console output. The formatter applies on the next invocation after `git pull --ff-only`.
