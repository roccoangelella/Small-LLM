---
status: observed
date: 2026-08-14
---

# 100M / 10B step-250 Beam fsync incident and resume

The first production segment reached update 250 with finite metrics:

```text
step: 250
target tokens/s: 42,398.7423
overflow events: 0
validation blocks: 16
validation target tokens: 2,097,152
validation loss: 8.8270058781
validation perplexity: 6,815.8488
validation elapsed: 51.6018 seconds
```

After validation, the checkpoint writer serialized and hashed the joint state,
wrote `checkpoint.json` and `local_manifest.json`, and atomically renamed the
staging tree to `step-00000250`. The final directory and manifest became visible
in `small-llm-runs`, but the trainer produced no `local_checkpoint` event or
step 251 for roughly ten minutes. Code inspection localized the only operation
after final rename and before return to `os.fsync()` on the checkpoint parent
directory. The GPU function was cancelled while idle; no later update existed.

Beam Volumes already persist distributed writes without a client commit call.
Source commit `1f9dff920ecc45ce2fdb43fd875514a18391273d` therefore keeps the staging/manifest/atomic-rename and
hash-verification protocol but sets `SMALL_LLM_CHECKPOINT_FSYNC=0` for Beam, so
local-disk power-loss barriers are not issued against the distributed Volume.
Other providers retain fsync by default.

Because the checkpoint was produced by launch source `42b0376`, the resume path
contains a one-time exact parent gate for that full commit. It first compares
all other frozen runtime fields, records `42b0376` as
`resume_parent_source_commit`, and moves the active infrastructure source to
`1f9dff920ecc45ce2fdb43fd875514a18391273d`. This does not change model state, optimizer state, scaler, RNG,
dataset order, block size, schedule, seed, precision, or microbatch.

The next CPU launch independently reported:

```text
checkpoint_source: modal_volume
local_checkpoint_id: step-00000250
local_step / completed_steps: 250 / 250
next_block_id / start_block_id: 250 / 250
remote_step: 0
GPU dispatch allowed: true
```

The successful resume gates were import function
`6541e0a9-cee6-4385-b986-321e7f914309`, stage function
`4356f89d-867e-442e-bde9-e3a3fd1ab513`, and fresh visibility function
`89663f69-7874-498e-a399-600a051d6f8b`.

The GPU trainer launched with `--resume step-00000250`, 76,044 remaining
updates, microbatch 4, and W&B resume policy `must`. W&B printed `Resuming run
100M model on 10B tokens` at the unchanged run URL. After rebuilding the
container-local Triton cache, production advanced through at least update 267;
step 267 reported 41,454.4172 target tokens/s with no error.

Focused validation before resume: 19 Beam/checkpoint tests passed and compileall
passed. The full `tests.test_remote_checkpoint` module retained its pre-existing
mock state-equality failure because restored state includes `python_rng_state`;
the new distributed-Volume no-fsync checkpoint test passed.
