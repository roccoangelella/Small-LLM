---
status: accepted
date: 2026-09-13
supersedes: null
---

# 0183 — Continue the MoE run across provider accounts

- Status: accepted
- Date: 2026-09-13
- Scope: accepted MoE production; historical dense and pilot recipes stay separate.

## Context and problem statement

Edo requested the ability to consume the available provider credits, then continue the
same training on another account/GPU. No credit monitor, scheduler, or preventive
budget shutdown is needed. Checkpoints are the recovery boundary.

## Considered options

- Rely on the current provider's volume alone: not portable across accounts.
- Publish verified checkpoints to the existing HF Storage Bucket and resume explicitly.

## Decision outcome

Both MoE launchers now use the existing HF Storage Bucket transport. Every cadence
checkpoint and final checkpoint is uploaded synchronously, verified by read-back
SHA-256, then exposed through `latest.json`. Keep the latest snapshot for continuation
and the best validation-loss snapshot for later evaluation. Prune other remote
snapshots only after publication succeeds. This reuses the trainer's existing best
pointer toward the verified `/last` tree; it does not upload a second copy of the best.

The CPU gate recovers a newer remote snapshot into an empty provider volume before
dataset staging. A missing checkpoint is an error unless the operator explicitly
selects `--resume new`; a new run may not reuse a run ID with existing progress.
`--resume latest` is the normal launch and recovery command. Run only one writer at
a time. `best` is not an automatic continuation source.

The accepted MoE may change microbatch size and logging/checkpoint/evaluation cadence
on resume. Its logical optimizer batch remains one dataset block (64 sequences of
2,048 targets in production), and its weights, optimizer moments, RNG, router buffers,
token scheduler and data cursor are restored. Precision, LR, optimizer, schedule,
model identity and corpus identity remain strict. New checkpoints record the new
execution configuration; existing snapshots are not rewritten to adapt them.

W&B uses the same explicit entity/project/run ID across providers. Initial creation
uses `allow`; checkpoint continuation uses `must`. A durability receipt binds the
checkpoint to the corpus metadata hashes, source commit, HF bucket and W&B identity.

## Consequences

- There is no dependence on access to the previous provider after a verified upload.
- A hard interruption replays updates after the last durable checkpoint. It cannot
  preserve an unfinished update/upload. W&B is telemetry, not the resume authority;
  unflushed telemetry may be lost on a hard kill, and replayed updates can recur on
  the custom `trainer/global_step` axis.
- Upload/read-back is synchronous and consumes time. Its live overhead has not been
  measured for this MoE integration; do not invent a throughput benefit or cost.
- Different GPUs/microbatch partitions may introduce floating-point differences.
  Restored tensors are exact; an identical future numerical trajectory is not promised.
- This authorizes implementation, not a paid run, deployment or push.

## Verification

`tests/test_moe_provider_continuation.py` exercises a real tiny MoE and the real CLI,
with a byte-accurate bucket SDK fake and a W&B fake. It covers empty-account restore,
execution changes, preserved state/cursor/LR, subsequent updates, same-volume artifact
identity, per-checkpoint publication, stable W&B identity, rejected scientific drift,
failed upload, corrupt download, and latest/best retention. Live cross-GPU continuation
on provider hardware remains to be exercised under an authorized launch.

Procedure: [MoE provider continuation](../runbooks/moe-provider-continuation.md).

Evidence: [CPU continuation verification](../evidence/moe_provider_continuation_cpu_2026-09-13.md).
