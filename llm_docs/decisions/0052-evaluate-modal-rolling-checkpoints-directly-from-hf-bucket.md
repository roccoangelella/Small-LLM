---
status: accepted
date: 2026-08-12
---

# 0052 — Evaluate Modal rolling checkpoints directly from the HF Storage Bucket

## Context

The Modal production checkpoint transport authorized by ADR 0047 writes verified two-phase checkpoints to a private Hugging Face Storage Bucket. In rolling latest-only mode the bucket retains `run/<run_id>/latest.json` plus the checkpoint tree it names and deliberately removes `best.json` and superseded checkpoint trees.

The existing `trainer.post_pretraining_prompt_suite` reads the Git-backed Hugging Face model repository and therefore cannot resolve a Modal-only bucket checkpoint such as `run/100m-2b-data-001/checkpoints/step-00015267/last`.

The frozen canonical full post-pretraining comparison in ADR 0025 still defines `pointer=best`. A rolling bucket `latest` checkpoint must therefore not be silently relabeled as `best`.

## Decision

Add a dedicated bucket-backed evaluator entrypoint:

```text
python -m trainer.post_pretraining_prompt_suite_bucket
```

It reuses the existing prompt-suite prompts, decoding, model reconstruction, and output format and changes only checkpoint transport. It:

- resolves the checkpoint bucket from `SMALL_LLM_HF_CHECKPOINT_BUCKET_ID` when set, otherwise from `<SMALL_LLM_HF_REPO_ID>-checkpoints`, matching Modal runtime convention;
- requires an explicit run identity through `--run-id` or `SMALL_LLM_RUN_ID`;
- accepts only `--pointer latest`, because rolling Modal bucket retention intentionally does not preserve `best.json`;
- downloads `run/<run_id>/latest.json` and its referenced `.../last` tree directly from the private bucket;
- verifies `local_manifest.json` and the pointer/embedded checkpoint publication manifest before loading the native trainer checkpoint;
- records `checkpoint_source=hf_storage_bucket` and the bucket ID in the evaluation output metadata.

Keep `trainer.post_pretraining_prompt_suite` unchanged for Git-backed repository checkpoints. Do not add an automatic `best -> latest` fallback because that would silently weaken the frozen comparison contract.

Raise the `post-training` optional dependency to `huggingface-hub>=1.5,<2`, matching the Storage Bucket API requirement already used by Modal production.

## Comparison consequence

A Modal bucket `latest` evaluation is a valid deterministic diagnostic of that exact verified checkpoint. It is not, by itself, evidence that the ADR-0025 canonical `best` checkpoint comparison was performed. If a canonical best-checkpoint comparison is required for a run whose bucket has already pruned `best`, recover or publish the actual best checkpoint separately rather than changing the pointer meaning.

## Links

- [`0047-use-hf-storage-bucket-for-modal-cross-workspace-checkpoints.md`](0047-use-hf-storage-bucket-for-modal-cross-workspace-checkpoints.md)
- [`0025-freeze-canonical-full-post-pretraining-prompt-suite.md`](0025-freeze-canonical-full-post-pretraining-prompt-suite.md)
- [`../runbooks/post_pretraining_prompt_suite.md`](../runbooks/post_pretraining_prompt_suite.md)
