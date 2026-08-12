---
status: accepted
date: 2026-08-12
---

# 0054 — Retire Google Drive for new dataset durability

## Context

ADR 0053 makes a private Hugging Face Storage Bucket the canonical store for the approximately-10B derived corpus and uses a verified rolling local cache during Modal training. The repository still retained an older Google Drive upload backend, OAuth setup, Google client dependencies, and publication configuration from the earlier finite-dataset pipeline.

Keeping two remote durability providers no longer provides a useful training property. Reusing a dataset shard in a later pass does not require a second permanent mirror: the immutable HF object can be downloaded again. With a rolling cache, training on shard N can overlap the verified download of the next logical shard. At an epoch boundary, a future epoch-aware reader can wrap the logical successor from the final shard to shard zero and prefetch it during work on the final shard.

The current trainer remains one-pass. Repeated epochs are therefore a future trainer/cursor feature, not a storage-backend requirement.

## Decision

For all new dataset production:

- Hugging Face Storage Buckets are the only remote dataset durability backend.
- Remove the executable Google Drive shard backend, Google OAuth setup code, Google API/auth dependencies, Drive credential environment variables, and Drive-specific tests.
- Keep the private Kaggle publication workflow available, but make its dataset build/durability phase HF-backed.
- Keep local-only production only as an explicit bounded/test mode; trusted remote production uses HF.
- Continue to verify every remotely durable shard by byte size and SHA-256 read-back before advancing durable producer state or evicting the local finalized shard.
- The 10B rolling cache continues to keep current + next train shards locally and can redownload the same immutable HF shard in a future repeated pass.

## Legacy compatibility boundary

Do **not** invalidate already-built datasets or existing checkpoints merely to rename historical schema fields.

The following names remain readable compatibility contracts:

- `drive_manifest.json`;
- `drive_file_id`;
- `drive_checksums`;
- trainer/checkpoint arguments and identity fields that explicitly bind a historical `drive_manifest.json`.

For new HF-backed production these names are provider-neutral legacy schema fields only. No Google Drive API, authentication, upload, verification, or download implementation remains behind them.

Historical runbooks/evidence may continue to describe Google Drive because that is the storage provider actually used for those completed artifacts. They are records, not current operating instructions.

## Consequences

- New dataset setup needs `HF_TOKEN` and either `SMALL_LLM_HF_DATASET_BUCKET_ID` or `SMALL_LLM_HF_REPO_ID`; no Google credentials are required.
- The Modal image and dataset remote requirements no longer install Google API/auth packages.
- Dataset durability and cross-workspace portability use the same general storage provider family as Modal checkpoint transport, while remaining separate bucket namespaces.
- Future multi-epoch training can reuse the same HF shard inventory with rolling prefetch; implementing repeated passes requires an epoch/pass-aware logical cursor and checkpoint state, but no new corpus copy and no Drive mirror.
- Existing 20M/Kaggle datasets and historical checkpoints remain readable through the frozen legacy manifest schema.

## Relationship to earlier decisions

This decision narrows the storage-provider choice in ADR 0053 and retires the older Google Drive production implementation. It does not alter source data, token order, model geometry, optimizer geometry, training precision, checkpoint cadence, or the ADR-0050 behavioral gate for launching 100M/10B training.

## Links

- [`0053-stream-10b-through-one-gib-hf-shards-and-cpu-stage-before-h100.md`](0053-stream-10b-through-one-gib-hf-shards-and-cpu-stage-before-h100.md)
- [`0037-consolidate-dataset-profile-tools-and-retire-one-off-qualification-code.md`](0037-consolidate-dataset-profile-tools-and-retire-one-off-qualification-code.md)
- [`../reference/dataset_and_tokenization.md`](../reference/dataset_and_tokenization.md)
