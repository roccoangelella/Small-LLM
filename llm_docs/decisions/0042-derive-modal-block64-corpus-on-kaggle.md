---
status: accepted
date: 2026-08-11
---

# Derive the Modal block-64 corpus directly on Kaggle

## Decision

Do not download the existing 2B finite dataset to the local workstation for Modal preparation.

Use the already-published Kaggle dataset `small-llm-20m-2b-dataset-001` as the source of the byte-preserving block-64 derivation authorized by ADR 0041. Run the derivation inside Kaggle, where the attached dataset is already available read-only under `/kaggle/input`.

The canonical Kaggle-side output directory is:

```text
/kaggle/working/modal-2b-b64-dataset-001
```

The canonical wrapper is:

```bash
python kaggle/reblock_for_modal.py
```

The wrapper must auto-discover exactly one attached schema-v2 corpus with production run ID `20m-2b-dataset-001`, context length 2,048, and 16 sequences per source block. It must not select a dataset by folder name alone.

The wrapper delegates the actual byte-preserving transformation to `dataset.reblock`, producing the `modal-2b-b64-dataset-001` corpus under the fixed Kaggle working directory. The derived directory is then uploaded directly from Kaggle to the Modal `small-llm-data` Volume at `/datasets/modal-2b-b64-dataset-001`; no local workstation copy is required.

## Rationale

The verified 2B corpus is already present in Kaggle and is several gigabytes. Downloading it to the workstation only to rewrite block boundaries and upload it again to Modal adds unnecessary transfer time, disk use, and another opportunity for operator error.

Kaggle already mounts the immutable source dataset. Reblocking there preserves the exact sequence bytes while keeping the transfer path simple:

```text
Kaggle attached dataset -> /kaggle/working block-64 derivative -> Modal Volume
```

The fixed output directory also removes an avoidable launch-time choice and gives the Modal upload command a stable source path.

## Consequences

- the local workstation does not need a 2B dataset copy;
- the original Kaggle dataset remains unchanged and read-only;
- `/kaggle/working/modal-2b-b64-dataset-001` may be rebuilt explicitly with `--replace-output`;
- Modal training still performs its own full dataset verification on first use inside the read-only `small-llm-data` Volume;
- checkpoint and W&B behavior from ADR 0041 is unchanged.
