---
status: observed
date: 2026-08-14
---

# 100M / 10B Beam worker loss and step-1500 resume

The RTX5090 production container disappeared at roughly 09:22 UTC after the
trainer had emitted finite updates through at least step 1,656. Beam showed no
active container or worker, the launcher-side task remained stale as `RUNNING`,
and W&B later classified the session as crashed. No trainer traceback, nonfinite
metric, CUDA error, or orderly shutdown was observed. This establishes an
abrupt worker/container loss; it does not establish the provider-side root
cause.

The latest independently verified remote durability point was:

```text
repository: roccoangelella/small-llm-100m-qualification
checkpoint: step-00001500
completed steps: 1500
next block: 1500
```

The stale local launcher was cancelled, which changed old Beam task
`79ba5b26-49b9-4931-9814-827d24d8f6e9` to `CANCELLED` at
09:37:51 UTC. Relaunch used a detached worktree at the exact active source
commit `1f9dff920ecc45ce2fdb43fd875514a18391273d`; no current-branch source
changes entered the training payload.

The replacement CPU gates reported `checkpoint_source=hf_remote`,
`local_checkpoint_id=step-00001500`, `remote_step=1500`, and
`start_block_id=1500`. Import preflight, dataset staging, and fresh volume
visibility completed as tasks `70906719-82e5-4228-9f41-da01cf90ac6f`,
`8368424e-19c2-4803-a311-108a792f80ce`, and
`dadcf0a0-8bc8-4bc4-b8cd-3ef9ab546f47` respectively.

Replacement GPU task `7e92c8a8-d893-4d49-be00-500db8b5e7c9` started one
RTX5090 container at 09:39:37 UTC. The trainer command retained microbatch 4,
the uncapped remaining 74,794-update horizon, `--resume step-00001500`, W&B
mode `online`, the unchanged run ID, and W&B resume policy `must`. W&B printed
`Resuming run 100M model on 10B tokens` at the existing run URL.

The first post-resume update was step 1,501. Its 661.57 target tokens/s includes
the one-time compile/cold-start cost. Production then returned to the qualified
steady range and advanced through at least step 1,529; steps 1,527-1,529
reported 42,225.37, 42,021.79, and 40,718.77 target tokens/s. Beam showed
exactly one `RUNNING` container at that observation point.
