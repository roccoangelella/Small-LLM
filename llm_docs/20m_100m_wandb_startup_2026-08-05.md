# 20M/100M W&B startup diagnosis and fix — 2026-08-05

## Root cause

The repeated Kaggle `wandb.init()` timeout was not caused by slow healthy initialization. The fixed production W&B run ID `20m-100m-data-001` had previously been created and deleted. With `wandb==0.26.1`, the W&B backend returned HTTP 409 from the GraphQL `upsertBucket` request:

```text
run 20m-100m-data-001 was previously created and deleted; try a new run id
```

The SDK classified that permanent conflict as retryable, retried with backoff, and eventually surfaced only the generic initialization-timeout exception. Increasing `WANDB_INIT_TIMEOUT` therefore delayed the same deterministic failure and was not a solution. This explains why earlier attempts also failed after being allowed to wait approximately 300 seconds.

## Isolated phases in the real Kaggle runtime

The diagnosis used the exact required runtime: Python 3.13.14 and `wandb==0.26.1`.

A minimal independent run established that the Kaggle environment itself was healthy:

| Phase | Result | Representative elapsed time |
|---|---:|---:|
| Kaggle `WANDB_API_KEY` propagation | passed | <0.001 s |
| DNS resolution for `api.wandb.ai` | passed | 0.003 s |
| TLS connection to `api.wandb.ai` | passed, TLS 1.3 | 0.011 s |
| API-key validation | passed | 0.22 s |
| local `wandb-core` startup via offline init | passed | 0.39 s |
| online project/run initialization with a valid ID | passed | 1.26 s |

`WANDB_ENTITY` was not configured as a Kaggle Secret. Authentication still resolved the key to entity `rocchissimo936-none`, so absence of an explicit entity was not the failure.

The exact production trainer command reproduced the failure with run ID `20m-100m-data-001` and a 30-second budget. Its `debug-internal.log` recorded repeated HTTP 409 responses containing the deleted-run message. `debug-core.log` showed that the local service started and accepted its Unix-socket connection normally before the client cancelled the request at timeout.

The same exact trainer integration, dataset, model configuration, Python version, W&B version, and optimizer settings initialized promptly with a clean run ID and reached optimizer update 1. This confirmed that project/run creation and resume identity were the only failing phase.

## Fix

The operational fix is deliberately narrow:

1. Replace the tombstoned W&B identity with the new stable run ID `20m-100m-data-003`.
2. Restore a 30-second initialization budget. Healthy initialization is expected to complete far below that limit.
3. Add `kaggle/wandb_preflight.py`, invoked before dataset scanning, microbatch qualification, or training. It verifies:
   - exact Python and W&B versions;
   - secret propagation;
   - DNS and TLS connectivity;
   - API-key authentication and resolved entity;
   - local `wandb-core` startup;
   - creation/resume of the exact production project and run ID.
4. Preserve the newest online `debug.log`, `debug-internal.log`, and `debug-core.log` under the launch evidence directory with SHA-256 hashes.
5. Detect the deleted-run 409 marker and fail with an actionable tombstoned-ID error instead of starting expensive work or recommending a longer timeout.
6. Reuse the preflight-created production run with `resume="allow"` for a fresh training launch and `resume="must"` only when a verified training checkpoint is restored.

## Scientific invariants

The fix does not modify:

- pinned scientific launch commit `43190cb72443a2de290dc8e6f2c54f29d8dff501`;
- model architecture or initialization;
- dataset contents or ordering;
- one-pass token schedule;
- optimizer type or hyperparameters;
- effective 16-sequence optimizer batch;
- microbatch-4 qualification thresholds;
- checkpoint schema, directory, cadence, or remote identity.

Only W&B startup validation and the unusable W&B run identity changed.

## Regression coverage

The regression tests cover:

- the 30-second launcher budget;
- exact Python 3.13 and `wandb==0.26.1` preflight command construction;
- use of the clean stable production run ID;
- all six required preflight phases;
- rejection of an online initialization exceeding the healthy budget;
- classification of the W&B deleted-run HTTP 409;
- deterministic preservation of all three online W&B debug logs;
- unchanged W&B resume semantics for fresh and checkpoint-resumed training;
- existing W&B telemetry behavior and API-key non-disclosure.

## Verification record

Before-fix reproduction in Kaggle:

```text
runtime: Python 3.13.14, wandb 0.26.1
run ID: 20m-100m-data-001
result: wandb.init timed out at 30 seconds
root evidence: repeated HTTP 409 "previously created and deleted"
```

Fixed preflight in Kaggle:

```text
run ID: 20m-100m-data-003
status: passed
all phases: passed
total elapsed: 2.396 seconds
online project/run phase: 1.261 seconds
resolved entity: rocchissimo936-none
debug.log: preserved from online run
debug-internal.log: preserved from online run
debug-core.log: preserved from online run
```

The final full-launch verification and commit SHA are recorded after the pushed launcher is executed in the real Kaggle notebook.
