# 20M/100M W&B startup handling — 2026-08-05

Repeated Kaggle launches reached the real-training boundary but failed before the first optimizer update because `wandb.init()` exhausted its initialization window.

## Verification result

The first timeout patch was incomplete:

- it used `os.environ.setdefault("WANDB_INIT_TIMEOUT", "300")`, so an existing Kaggle or notebook value such as `90` was preserved instead of overridden;
- its regression test failed under the project Python 3.13 runtime because it attempted `ast.literal_eval()` on every top-level assignment, including non-literal assignments;
- a timed-out first `wandb.init()` can create the fixed run on the W&B server before the client gives up, while the next fresh launcher attempt previously used the effective `resume="never"` policy and could reject that existing run.

## Operational decision

- keep W&B in online mode for the 20M-model/100M-token run;
- force `WANDB_INIT_TIMEOUT=600` in the one-click entrypoint before launcher imports;
- copy the timeout explicitly into the trainer subprocess environment;
- use `wandb-resume=allow` when no training checkpoint exists, so a run created immediately before an initialization timeout can be recovered;
- continue using `wandb-resume=must` when resuming from a verified training checkpoint;
- preserve the pinned scientific training commit, optimizer configuration, microbatch-4 selection, 16-sequence optimizer block, dataset order, checkpoint cadence, and fixed W&B run identity;
- treat the PyTorch message about unavailable NumPy as a non-fatal warning separate from the W&B communication timeout;
- do not claim that a failed W&B initialization completed any optimizer update or produced a resumable training checkpoint.

The change is operational only and does not modify training mathematics.

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
