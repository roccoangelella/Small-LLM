---
status: accepted
date: 2026-08-24
---

# Teacher-forced evaluation must support incremental HF validation datasets

The post-pretraining teacher-forced diagnostic must remain fail-closed on dataset identity while supporting both historical static schema-v2 datasets and the current incremental Hugging Face dataset architecture.

For historical checkpoints, preserve the existing exact `drive_manifest.json` SHA-256 match. For modern checkpoints whose provider-neutral checkpoint transport records `dataset_manifest_sha256`, use that manifest hash as the dataset identity proof instead.

In `auto` mode, first reuse any identity-matched local dataset under Kaggle inputs or the Small-LLM Kaggle working dataset cache. If no matching modern incremental dataset is present, reconstruct the same stable consumer manifest from the immutable run contract and shard frontier, require its SHA-256 to equal the checkpoint-recorded `dataset_manifest_sha256`, and download only the frozen validation shards from the Hugging Face dataset bucket. Do not stage unrelated training shards solely for teacher-forced evaluation.

Multiple local rolling-cache roots with the same modern consumer-manifest hash are equivalent for this diagnostic because the validation inventory is frozen by the incremental dataset contract; choose one deterministically rather than treating duplicate identical cache copies as a scientific ambiguity.

Teacher-forced reports should record `dataset_manifest_sha256` for all datasets and retain `drive_manifest_sha256` when a legacy dataset has one.
