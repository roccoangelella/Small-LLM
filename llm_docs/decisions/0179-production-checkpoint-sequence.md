---
status: accepted
date: 2026-09-12
supersedes: null
---

# 0179 — Production checkpoint sequence for the accepted MoE run

## Context and problem statement

The accepted 64E/Top-2 production run is roughly 762,939 updates at about 2.16 s
per update, close to 19 days of wall clock, with a checkpoint near 1.16 GB.

The launch path as delivered by ADR 0177 could not survive that horizon:

- `checkpoint_every_steps` defaulted to zero, so the trainer wrote only the
  final checkpoint of a segment. A container lost at day 12 lost 12 days.
- The Modal function has `timeout=24h` and `retries=3`. A retry re-runs the same
  payload with the same `--resume`, so a container killed at the timeout restarted
  the whole run from the original starting point instead of continuing.
- The run volume was committed once, after the child process exited. A checkpoint
  written to the volume but never committed is not durable.
- No wall-clock budget and no signal handler existed, so the 24 h timeout always
  arrived mid-update.
- Nothing pruned local checkpoints: 762 checkpoints of 1.16 GB is about 885 GB of
  volume storage.

The trainer CLI is shared with the dense line, whose `TrainerConfig` is hashed into
`configuration_hash` and gates exact resume, so no new knob may enter that config.

## Considered options

- Keep the final-checkpoint-only behaviour and rely on Modal retries.
- Checkpoint periodically and keep every checkpoint.
- Checkpoint periodically, keep only the last checkpoint plus the best one.
- Checkpoint periodically, keep a rolling window of the last N plus the best,
  optional milestones, drain before the provider timeout, commit the volume per
  checkpoint, and resolve `--resume` from the newest complete local checkpoint.

## Decision outcome

Chosen option: **the rolling window with drain, per-checkpoint commit and
auto-resume**, because it bounds the loss from any single container death to one
checkpoint interval, bounds storage, and makes a retry or a manual relaunch continue
the run rather than restart it.

Concretely:

- A local checkpoint every 1,000 updates (about 36 minutes) on the production path.
  The shared trainer default stays zero.
- Local retention keeps the newest 3 step checkpoints and never deletes the
  checkpoint last promoted by `publish_best_model_if_improved`, the resume source,
  or any checkpoint whose update number is a multiple of `--milestone-every-steps`
  (off by default). A deletion happens only after the newest checkpoint verifies as
  complete against its own `local_manifest.json` and `checkpoint.json`.
- `--max-wall-seconds` drains the process: when the next update would exceed the
  budget, the trainer saves through the existing `ensure_local_checkpoint`, prints a
  final `{"drained": {...}}` line and exits zero. Modal passes 23 h against its 24 h
  function timeout; Beam passes zero because its GPU function has no timeout.
- `moe_production.run_provider_payload` streams the child's stdout and commits the
  run volume after every `local_checkpoint` event and once at the end.
- An absent or `latest` resume resolves to the newest complete checkpoint under
  `run_root/<run_id>/checkpoints`, and `--steps` is launched as the remaining
  updates, since the trainer reads `--steps` as a per-segment count. The request
  therefore carries `total_steps`, the absolute target.
- The completeness contract lives once, in
  `dataset/src/checkpoint_sequence.py`; `moe_pilot` now imports it instead of
  holding a second copy.

Rejected: **keep only the last checkpoint plus the best**, with no rolling window.
A corrupted or half-written last checkpoint would leave only the best checkpoint,
which is selected on held-out loss and can be days old, so the worst case stays
unbounded exactly when durability matters most.

The new trainer flags are argparse-only and default off. They are deliberately not
`TrainerConfig` fields: that dataclass is hashed into the checkpoint
`configuration_hash`, so adding an operational knob there would change the identity
of every existing run and break resume.

## Consequences

### Positive

- Worst-case loss from a container death is one checkpoint interval, about 36 minutes.
- Volume storage is bounded at 3 checkpoints plus best plus milestones.
- Modal retries and manual relaunches continue the run; a finished run relaunches
  into a no-op instead of restarting training.
- A drained segment ends on a complete checkpoint rather than mid-update.
- The dense trainer path is unchanged when the flags are absent.

### Negative or limiting

- A drained run needs a relaunch: the drain exits zero, so Modal does not retry it.
- Retention verifies the newest checkpoint's manifest hashes on every prune, which
  reads the checkpoint once (about 1.16 GB) per interval.
- Auto-resume trusts the local volume; a lost volume still loses everything not
  published remotely.
- Per-checkpoint commits make the volume commit cost part of the training loop.

## Validation

`tests/test_production_checkpoint_sequence.py` covers the drain (checkpoint saved,
exit zero, no double save on a checkpoint boundary), retention (last N, milestones,
promoted best, resume source, and no deletion behind an incomplete newest
checkpoint), auto-resume (latest complete checkpoint plus remaining steps, explicit
ID, finished run) and one volume commit per checkpoint event.
`tests/test_moe_production_wiring.py` pins the production defaults and asserts the
trainer CLI accepts every flag the production command emits. The operational proof
is the first production segment: a drain at the 23 h budget followed by a relaunch
that resumes at the drained checkpoint.

## Links

- [Accepted MoE production launch contract](0177-accepted-moe-production-launch-contract.md)
- [100M durability precedent](0004-run-100m-in-one-session-with-250-step-durability.md)
- [Paired pilot contract](0168-moe-paired-pilot-controller-and-observation.md)
