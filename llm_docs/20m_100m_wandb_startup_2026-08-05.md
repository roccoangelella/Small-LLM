# 20M/100M W&B startup handling — 2026-08-05

Repeated Kaggle launches reached the real-training boundary but failed before the first optimizer update because `wandb.init()` exhausted its initialization window.

## Verification result

The first timeout patch was incomplete:

- it used `os.environ.setdefault("WANDB_INIT_TIMEOUT", "300")`, so an existing Kaggle or notebook value such as `90` was preserved instead of overridden;
- its regression test failed under the project Python 3.13 runtime because it attempted `ast.literal_eval()` on every top-level assignment, including non-literal assignments;
- a timed-out first `wandb.init()` can create the fixed run on the W&B server before the client gives up, while the next fresh launcher attempt previously used the effective `resume="never"` policy and could reject that existing run.

## Operational implementation

- keep W&B in online mode for the 20M-model/100M-token run;
- force `WANDB_INIT_TIMEOUT=600` in the one-click entrypoint before launcher imports;
- copy the timeout explicitly into the trainer subprocess environment;
- use `wandb-resume=allow` when no training checkpoint exists, so a run created immediately before an initialization timeout can be recovered;
- continue using `wandb-resume=must` when resuming from a verified training checkpoint;
- preserve the pinned scientific training commit, optimizer configuration, microbatch-4 selection, 16-sequence optimizer block, dataset order, checkpoint cadence, and fixed W&B run identity;
- treat the PyTorch message about unavailable NumPy as a non-fatal warning separate from the W&B communication timeout;
- do not claim that a failed W&B initialization completed any optimizer update or produced a resumable training checkpoint.

The change is operational only and does not modify training mathematics.

## User correction and revised diagnosis

The user reported that Kaggle had already been allowed to wait approximately 300 seconds for `wandb.init()` on multiple attempts. Therefore increasing the initialization timeout is not accepted as the root-cause solution. A healthy W&B initialization should complete in seconds, not minutes.

Revised decision:

- retain a long timeout only as a final safety ceiling;
- treat any initialization taking more than roughly 30 seconds as abnormal;
- diagnose W&B startup before launching expensive training;
- separate four phases: notebook environment propagation, local `wandb-core` startup, API-key/entity/project authentication, and online run creation/resume;
- capture and surface `debug.log`, `debug-internal.log`, and `debug-core.log` when the online probe fails;
- require an explicit verified W&B entity rather than relying silently on default-entity resolution for the fixed run identity;
- do not launch model training until a minimal online W&B probe succeeds promptly under the exact pinned Python and SDK environment.

Notebook caveat: if the earlier timeout was set with `!export WANDB_INIT_TIMEOUT=300` in a separate Kaggle cell, that export did not persist because each IPython `!` command runs in a separate shell. `%env WANDB_INIT_TIMEOUT=300`, assignment through `os.environ`, or prefixing the same shell command would persist. This caveat does not explain a confirmed timeout that actually reported 300 seconds.

## Evidence

Implemented and pushed on `main`:

```text
commit: 877718d
message: Fix Kaggle W&B startup retries
```

Verification completed locally:

```text
10 Kaggle launcher/console tests: passed
4 W&B telemetry tests: passed
wandb SDK pin: 0.26.1
resolved init_timeout with WANDB_INIT_TIMEOUT=600: 600.0 seconds
```

The implementation above verifies timeout propagation and retry semantics only. It does not prove that Kaggle can establish a healthy online W&B run; the revised diagnostic gate remains required.
