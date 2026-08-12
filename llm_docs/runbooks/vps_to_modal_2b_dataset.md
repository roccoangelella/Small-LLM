---
status: current
last_reviewed: 2026-08-12
---

# VPS preparation of the Modal 2B dataset

Use this path for the block-64 corpus required by the 100M / 2B Modal run. Kaggle is only the remote source of the already-published finite corpus. All commands and credentials for the new Modal lane stay on the VPS.

## One-time VPS prerequisites

Use the project `.venv` (Python 3.12) and install both CLIs there:

```bash
cd ~/Projects/Small-LLM
source .venv/bin/activate
uv pip install kaggle 'modal>=1.1,<2'
```

Modal authentication is the existing VPS profile created by `modal setup`.

For Kaggle, prefer the current API token environment variable:

```bash
export KAGGLE_API_TOKEN='...'
```

The official token file mechanisms are also accepted by the Kaggle CLI. No Kaggle notebook secrets are used.

## Prepare and upload in one command

From the VPS checkout:

```bash
git pull --ff-only
python modal/prepare_dataset.py
```

The helper is stage-idempotent. It performs:

1. authenticated Kaggle handle discovery for the frozen slug `small-llm-20m-2b-dataset-001` using `kaggle datasets list --mine --search ... --csv`;
2. Kaggle download/unzip only if a verified cached source is absent;
3. schema-v2 and production-run verification for `20m-2b-dataset-001`;
4. byte-preserving block-64 reblocking through `dataset.reblock` only if a verified derivative is absent;
5. resolve the actually authenticated Modal workspace/environment, create `small-llm-data` there if needed, verify `/datasets/modal-2b-b64-dataset-001` against the local manifest inventory, and upload only when that active remote destination is missing or incomplete; after any upload, verify the remote destination again before reporting readiness.

Fixed VPS paths:

```text
~/small-llm-data/kaggle/small-llm-20m-2b-dataset-001
~/small-llm-data/modal-2b-b64-dataset-001
```

Fixed Modal destination:

```text
/datasets/modal-2b-b64-dataset-001
```

If automatic Kaggle owner discovery cannot resolve the private dataset, set only the exact handle override and rerun:

```bash
export SMALL_LLM_2B_KAGGLE_DATASET_HANDLE='OWNER/small-llm-20m-2b-dataset-001'
python modal/prepare_dataset.py
```

## Recovery controls

Normal recovery, including a Modal account/workspace switch, is just rerunning the same command. The local upload marker is diagnostic only; the active Modal workspace/environment is checked on every upload-enabled run. Explicit controls exist only for repairing a known-bad stage:

```bash
python modal/prepare_dataset.py --force-download
python modal/prepare_dataset.py --force-reblock
python modal/prepare_dataset.py --force-upload
```

`--force-upload` now means re-upload even when the current active Modal workspace already verifies. A successful `modal volume put` is followed by remote inventory + manifest-SHA verification; if that verification fails, the helper exits non-zero instead of trusting the CLI success message.

Prepare locally on the VPS without touching Modal:

```bash
python modal/prepare_dataset.py --no-upload
```

The original Kaggle dataset is never modified or republished.

## Launch after preparation

The helper prints the canonical next command. Launch from the same VPS checkout:

```bash
modal run --detach modal/launch.py --model 100M --tokens 2B --gpu H100
```

The Modal runtime performs its own first-use full dataset verification inside the read-only `small-llm-data` Volume, then probes execution microbatches 16/32/48/64 and freezes the fastest safe candidate before optimizer step 1.

Checkpoints remain in `small-llm-runs`, with checkpoint + validation every 250 successful optimizer updates, final checkpointing, and W&B exact-resume semantics.
