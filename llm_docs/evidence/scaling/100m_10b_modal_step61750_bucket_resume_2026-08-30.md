---
status: evidence
date: 2026-08-30
---

# 100M/10B Modal step-61,750 Bucket migration and live resume — 2026-08-30

## Why the durable step changed

The first ADR-0132 migration saw only verified step 61,500. Later, the active
`generico-podcast1` workspace's `small-llm-runs` Volume exposed the completed
step-61,750 commit from the stopped H100 worker. This was distinct from the
incomplete step-61,750 upload in the legacy shared model repository: the Volume
directory contained `trainer_state.pkl` plus all four JSON files.

The Volume held nine retained checkpoint directories from step 59,750 through
step 61,750. A CPU-only Modal function loaded and independently verified the
newest tree before any new H100 allocation:

```text
checkpoint:                    step-00061750
global_step:                   61750
consumed targets:              8,093,696,000
last consumed block:           61749
microbatch:                    16
scheduler committed targets:   8,093,696,000
scheduler LR:                  1.3200863017015169e-05
validation loss:               2.846469341078773
trainer state bytes:           913,885,544
trainer state SHA-256:         0bb16f9e907f3953cc26aa0d6c5bc9699b4df43e610cfd61b7cee73ecde467cc
```

`verify_local_manifest` passed, and the independently streamed SHA-256 matched
the manifest entry. The checkpoint retained the historical best validation loss
`2.8437069645151496`, which W&B history binds to step 59,250. Because those exact
step-59,250 bytes are not retained, step 61,750 was not mislabeled as best and
the dedicated best-model repository remained absent.

## Bucket handoff

Source commit `61573f2248f397524c767c57beb57d9048bdd5ed` launched the canonical
detached command under Modal profile `generico-podcast1`. Its CPU prepare gate
selected local step 61,750 over Bucket/model-repository step 61,500, staged the
checkpoint-aligned data window, and published the verified tree to:

```text
roccoangelella/small-llm-100m-qualification-checkpoints
  run/100m-10b-deep-decay-from-step15500/
    checkpoints/step-00061750/last/...
    latest.json
```

The read-back pointer is 884 bytes with SHA-256
`2611706f3d6a3c004dea5491aa9509a23553118a34173285b8788e3c40565302`.
It names step 61,750 and records the same 913,885,544-byte trainer object and
SHA-256 above. Run-scoped mutable pruning then removed the superseded step-61,500
objects. The completed stable `100m-2b-data-001/step-00015267` Bucket namespace
was untouched.

## Live production confirmation

Detached app `ap-dJqLY4FYc9VvkR7VPA04hc` dispatched one exact H100 only after
the Bucket handoff succeeded. The trainer command resumed `step-00061750`, kept
global block 64 / microbatch 16, requested 14,544 remaining steps, resumed the
same W&B run with `resume=must`, and used these separate destinations:

```text
latest: roccoangelella/small-llm-100m-qualification-checkpoints
best:   roccoangelella/small-llm-100m-qualification-best-100m-10b-deep-decay-from-step15500
```

The first restored update was slower while kernels/state warmed. Subsequent
updates were finite; observed telemetry through step 61,762 included finite loss,
gradient norm, LR, scaler state, and optimizer statistics. Step 61,751 reported
loss `2.9124209880828857`, gradient norm `0.626305103302002`, LR
`1.320016738716934e-05`, and no overflow retry. After the local detached client
closed, Modal still reported the app as detached with one running task.

No H100 smoke or duplicate bounded run was launched. This is the production
continuation from the Bucket source.
