---
status: evidence
date: 2026-08-30
---

# 100M/10B Modal latest/best Hugging Face transport migration — 2026-08-30

## Scope

This records the ADR-0132 migration of the active Modal deep-decay run from Git-backed rolling checkpoint writes to the split transport:

- `latest` exact-resume state in `roccoangelella/small-llm-100m-qualification-checkpoints`;
- strict validation-loss `best` in the dedicated model repository `roccoangelella/small-llm-100m-qualification-best-100m-10b-deep-decay-from-step15500` when and only when the corresponding verified best checkpoint bytes are available.

Implementation commit `5b942181163ce5ca3f74e1ae61da4f9bcbb4e92b` completed the Modal/trainer support after pushed commits `ed3a2f7` and `b28bc11` had left the best-model path incomplete. The focused Modal/best-model suite passed 21/21 tests before push.

## Latest checkpoint migration

The legacy shared model-repository pointer resolved `step-00061500`. Its checkpoint manifest records:

```text
checkpoint:       step-00061500
trainer_state:    913,885,544 bytes
trainer SHA-256:  a3c8b018f49f3315a3443eb73810712dfc2adbb53bc3c49774ef693d32cf43ff
validation loss:  2.8463459765771404
```

The 913,885,544-byte trainer state was downloaded from the legacy repository and independently SHA-256 verified against the checkpoint manifest. The same object was then copied server-side into the checkpoint Bucket. The Bucket reports Xet content hash:

```text
90965319dc0c388b082e9ae893f419944e96bef02f20f21ad430c8e60c893e5a
```

That exactly matches the source object's signed Xet/CAS hash observed before migration.

The checkpoint's provider-neutral durability metadata was rewritten to:

```text
transport:                    modal-hf-bucket-checkpoint-v1
bucket_id:                    roccoangelella/small-llm-100m-qualification-checkpoints
source_commit:                5b942181163ce5ca3f74e1ae61da4f9bcbb4e92b
resume_parent_source_commit:  115769ada2324025a190653a486d47b8b19ea9ee
retention:                    latest_only_mutable_bucket
```

`checkpoint.json`, `local_manifest.json`, the rewritten `drive_manifest.json`, and the regenerated `checkpoint_manifest.json` were downloaded back from the Bucket and matched their local SHA-256 values exactly. Only after those checks did `run/100m-10b-deep-decay-from-step15500/latest.json` move to `step-00061500`. The pointer was downloaded back byte-for-byte, and run-scoped Bucket pruning reported `deleted_files=0`.

The legacy shared model repository also contains an incomplete `step-00061750` staging attempt from 09:20 UTC: only small metadata files exist and `latest.json` still resolves `step-00061500`. It is not a valid continuation checkpoint and was not promoted.

## Best-model status

The verified step-61,500 trainer state contains:

```text
global_step:           61500
consumed_tokens:       8060928000
best_validation_loss:  2.8437069645151496
overflow_events:       24
```

W&B scan history binds that persisted best loss to `trainer/global_step=59250`.

Therefore step 61,500 is valid `latest` but is not the run-wide `best`:

```text
step 59250 best loss:  2.8437069645151496
step 61500 loss:       2.8463459765771404
```

The dedicated best-model repository does not currently exist. No retained `step-00059250` checkpoint was found in the local filesystem, the current HF cache/history, or either configured Modal profile's `small-llm-runs` Volume. Publishing step 61,500 as `best` would therefore be false and was deliberately refused. The new trainer seeds its best threshold from checkpointed `best_validation_loss`; it will create/replace the dedicated best repo only for a verified checkpoint that equals a missing persisted best on resume or strictly improves the historical best thereafter.

## Deletion safety

The shared `roccoangelella/small-llm-100m-qualification` model repository is not safe to delete. It contains the stable `100m-2b-data-001` artifact and multiple pretraining/SFT/R-SFT run namespaces. No wholesale model-repository deletion was performed.

## Operational blocker

A CPU-only Modal invocation of the migration gate was attempted from commit `5b94218`, but Modal rejected the invocation because workspace `ac-DiJNvUEOeQU331C3N5PmE0` had exceeded its spend limit. The successful migration above was therefore executed through authenticated Hugging Face host tooling without allocating a Modal GPU or mutating scientific checkpoint bytes.
