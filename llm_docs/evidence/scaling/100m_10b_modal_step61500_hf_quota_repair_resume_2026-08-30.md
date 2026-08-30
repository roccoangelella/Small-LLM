---
status: evidence
observed_at: 2026-08-30
run_id: 100m-10b-deep-decay-from-step15500
---

# 100M/10B Modal step-61,500 HF quota repair and resume

## Failure diagnosis

Modal app `ap-86sjxvNMobYbQ9pTYhfsZw` trained normally through a locally valid
`step-00061500`. Its fatal traceback came from the Hugging Face commit API
during remote checkpoint publication:

```text
huggingface_hub.errors.BadRequestError: Private repository storage limit
reached, please upgrade your plan to get more storage space.
```

The request was the model-repository commit used by the rolling checkpoint
publisher. W&B's final summary agreed with the filesystem state: the newest
local checkpoint was `step-00061500`, while the last completed remote
publication remained `step-00061250`. There was no CUDA exception, nonfinite
loss, gradient overflow, data-identity mismatch, or checkpoint-state mismatch.
The stop was therefore a remote durability quota failure after successful
training computation.

## Repository cleanup and checkpoint repair

Before cleanup, the Hugging Face API reported 74,936,390,024 bytes of used
storage for `roccoangelella/small-llm-100m-qualification`. The current tree had
71 files. All 11 `latest.json` run pointers resolved to one current checkpoint
directory each, and the stable 100M/2B artifact was present; the excess was in
superseded Git/LFS history rather than required current files.

The `main` branch was super-squashed with a maintenance commit. Verification
afterward showed one branch commit while retaining all 11 current pointers and
the stable artifact. This permanently removed superseded branch and LFS
history; it did not delete the current checkpoint trees or stable model.

A CPU-only Modal repair then independently verified the already-local
step-61,500 checkpoint and published its 913,885,544-byte payload:

```text
checkpoint:     step-00061500
payload SHA256: a3c8b018f49f3315a3443eb73810712dfc2adbb53bc3c49774ef693d32cf43ff
old pointer:    step-00061250
new pointer:    step-00061500
cleanup:        pruned_and_squashed
```

The upload succeeded and the pointer was verified before production resumed.
No H100 was allocated for this repair. Earlier repair-preflight attempts failed
before publication on CPU image/import/path checks and did not mutate the live
pointer.

## Exact production resume

The normal production launcher resumed the repaired remote checkpoint in the
same run identity:

```text
profile:       generico-podcast1
environment:   main
app:           ap-9ctKLcPFhmbBBGwE2GXeiw
source commit: 9e6eaf84cfa49c5c2c4fedcb31c9009b48feb125
checkpoint:    step-00061500
completed:     61,500 updates
expected LR:   1.3288285153726578e-05
remaining:     14,794 updates
final step:    76,294
```

The CPU gate classified the existing execution layout as
`already_modal_sliced`, staged block 61,500, and returned ready before one exact
H100 was dispatched. Model ancestry, optimizer, scaler, RNG, data cursor,
64-sequence global block, FP16, GDN-2, hybrid Muon+AdamW, deep-decay schedule,
W&B identity, and Hugging Face namespace were unchanged.

The first resumed update was finite:

| step | block | consumed targets | loss | grad norm | LR | overflow retries |
|---:|---:|---:|---:|---:|---:|---:|
| 61,501 | 61,500 | 8,061,059,072 | 2.815817 | 0.634031 | 1.328793360472205e-05 | 0 |

Finite telemetry was observed through at least step 61,505 at approximately
64k target tokens/s after first-step compilation, with zero overflow retries.
An authenticated Modal listing then showed app
`ap-9ctKLcPFhmbBBGwE2GXeiw` detached with one live task. W&B resumed with
`resume=must` at
`https://wandb.ai/rocchissimo936-none/Small-LLM/runs/100m-10b-deep-decay-from-step15500`.

## Verification scope

No bounded H100 test run or broad test suite was used. The production CPU gate
provided checkpoint, schedule, and dataset verification; the repaired remote
publication proved the quota path writable; and the first finite production
updates supplied the hardware integration check. Only focused documentation
structure/link checks were run after recording this evidence.
