# Runtime-only upgrades of the existing MoE run

The runtime-efficiency candidate is opt-in. Defaults remain validation microbatch1 and synchronous upload. The active provider checkout is independently pinned; editing this candidate never upgrades a running Modal container.

## Source compatibility and rollback

Pass the **actual clean executor HEAD** as `--source-commit`. For a reviewed execution-only upgrade, also pass `--resume-source-commit <original-run-commit>` and `--resume latest` to the Modal or Beam production launcher. The latter flag is the immutable **run origin**, retained on subsequent launches with the upgraded executor, not a replacement for checking the actual HEAD.

The original v1 transport receipt remains unchanged, including its source anchor. All receipt fields still compare exactly: dataset manifest, run contract, bucket, validation blocks and W&B identity. A missing or invalid checkpoint rejects the transition. Each newly saved `checkpoint.json` records a separately versioned `executor` object containing actual source and original run source. Existing loaders accept this optional metadata; tensor/optimizer/scheduler/RNG formats are unchanged. `checkpoint.source_hash` binds corpus/work-plan identity, not the Git commit.

This mechanism authorizes no arbitrary semantic changes: review the exact commit and prove scientific configuration/state parity first. Rollback uses the previous clean executor, original source anchor, same run and latest verified checkpoint. Never rewind the shared run or use `new`. A CPU subprocess test executes the previous checkout against a new checkpoint and compares the next update exactly.

## Opt-in controls

- `--validation-microbatch-size 4`: same held-out blocks and active targets, token-weighted cross entropy; minor floating-point grouping differences are expected. Default1 remains the fallback. No validation cadence/content changes.
- `--async-checkpoint-upload`: one background publisher reading a completed filesystem snapshot. No access to live model tensors or optimizer. Capture validation and step at enqueue; publish/verify latest before best promotion and cleanup. Training polls failures every update. Before another local save/prune, wait for the previous upload, so retention cannot delete its input. Drain/normal termination waits for all uploads; a training failure also closes the worker without hiding the primary error. Hard process death can still lose work after the last verified remote checkpoint.
- Keep checkpoint cadence unchanged. A wider interval increases potential recomputation after provider termination and is a separate decision.

## Adoption

Use an already-needed stop/transition. Verify zero previous GPU writers, full latest checkpoint hashes, correct dataset/W&B identity, source review and billing authority. Respect any corpus hold until producer completion. First resume with performance flags disabled; qualify source compatibility through a verified new checkpoint. Introduce validation4 and then async publication at subsequent controlled transitions, measuring each independently. Do not hotpatch, start an extra GPU experiment, or auto-rotate accounts for credits.

Compare whole checkpoint cycles at similar token/schedule position and hardware: wall target tokens/s, validation duration, save duration, upload duration, peak memory and remote step. Async upload time itself may remain unchanged; useful gain is GPU time hidden behind computation. CPU tests do not establish H100 throughput or BF16 numerical equivalence. On a new failure stop and inspect; do not blind-retry. Functionally disable a feature by returning validation to1/removing async, or roll back the executor at latest with the same preflight.

Optimizer telemetry is unchanged. A tiny-MoE CPU experiment found exact optimizer-state/update parity with instrumentation disabled, but timings are noisy and cannot establish H100 training savings. Sampling is deferred pending attributable GPU evidence from the ordinary run.
