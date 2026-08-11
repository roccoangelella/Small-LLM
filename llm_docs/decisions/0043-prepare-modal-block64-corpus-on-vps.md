---
status: accepted
date: 2026-08-11
supersedes: 0042
---

# Prepare the Modal block-64 corpus on the VPS

## Decision

All operator interaction for the 100M / 2B Modal trajectory is performed from the VPS. Kaggle remains only the remote source that already stores the verified 2B finite dataset; no Kaggle notebook is part of the Modal workflow.

The canonical VPS preparation command is:

```bash
python modal/prepare_dataset.py
```

The command must:

1. use the frozen Kaggle dataset slug `small-llm-20m-2b-dataset-001`;
2. discover its exact authenticated `owner/slug` handle with the Kaggle CLI using `datasets list --mine --search`, unless `SMALL_LLM_2B_KAGGLE_DATASET_HANDLE` is explicitly set;
3. download and unzip the dataset on the VPS only when a verified cached source is absent;
4. verify schema-v2 geometry and production run ID `20m-2b-dataset-001` before reblocking;
5. delegate the byte-preserving 16-to-64 sequence transformation to `dataset.reblock`;
6. verify the derived `modal-2b-b64-dataset-001` corpus;
7. upload the derived directory from the VPS to Modal Volume `small-llm-data` at `/datasets/modal-2b-b64-dataset-001` unless `--no-upload` is requested;
8. be idempotent across successful stages so rerunning does not normally redownload, reblock, or reupload completed artifacts.

The fixed VPS paths are:

```text
source download/cache: ~/small-llm-data/kaggle/small-llm-20m-2b-dataset-001
derived block-64 corpus: ~/small-llm-data/modal-2b-b64-dataset-001
Modal destination: /datasets/modal-2b-b64-dataset-001
```

The VPS `.venv` is the operator environment for both Kaggle and Modal CLIs. Kaggle authentication uses the official CLI mechanisms, preferably `KAGGLE_API_TOKEN` or the user's token file; Modal authentication remains the existing VPS `modal setup` profile.

## Rationale

The user wants one control plane for the new training platform. Moving Kaggle-notebook authentication and Modal credentials into a notebook adds another execution environment and secret surface without providing a training benefit.

The VPS already hosts the Small-LLM checkout and the authenticated Modal CLI. Downloading the several-gigabyte finite corpus there once is acceptable because the source is cached and the byte-preserving reblock is also cached. The resulting path is operationally simpler:

```text
Kaggle dataset -> VPS verified cache -> VPS block-64 derivative -> Modal Volume -> Modal GPU training
```

No workstation or Kaggle notebook participates.

## Consequences

- `kaggle/reblock_for_modal.py` is retired;
- the Kaggle-notebook-to-Modal runbook is retired;
- `modal/prepare_dataset.py` becomes the supported preparation surface;
- the original Kaggle dataset remains immutable and is never republished by this workflow;
- the Modal runtime still performs its own full schema-v2 verification on first use;
- the block-64 optimizer geometry, 16/32/48/64 microbatch qualification, 250-update checkpoint/validation cadence, durable `small-llm-runs` checkpoints, and W&B exact-resume contract from ADR 0041 remain unchanged.
