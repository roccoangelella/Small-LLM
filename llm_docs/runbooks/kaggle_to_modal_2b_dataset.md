---
status: current
last_reviewed: 2026-08-11
---

# Kaggle to Modal 2B dataset handoff

Use this path for the block-64 corpus required by the 100M / 2B Modal run. The local workstation never needs to hold the 2B dataset.

## Required Kaggle attachment

Attach the existing Kaggle dataset:

```text
small-llm-20m-2b-dataset-001
```

The wrapper does not trust the mount directory name alone. It scans the attached dataset manifests and requires exactly one schema-v2 corpus with:

```text
production run ID: 20m-2b-dataset-001
context length: 2048
stored tokens per sequence: 2049
source sequences per block: 16
```

## Derive block 64 on Kaggle

From the Small-LLM checkout in the Kaggle notebook:

```bash
git pull --ff-only
python kaggle/reblock_for_modal.py
```

Canonical output:

```text
/kaggle/working/modal-2b-b64-dataset-001
```

If an earlier incomplete or intentionally replaceable derived directory already exists:

```bash
python kaggle/reblock_for_modal.py --replace-output
```

The wrapper delegates to `dataset.reblock`, which verifies the source, preserves the train/validation sequence byte streams exactly, creates the block-64 manifest/shards, verifies the derived corpus, and prints the next Modal upload command.

## Upload directly from Kaggle to Modal

Do not download the derived directory to the workstation. Add fresh `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` values as private Kaggle Secrets, then expose them only to the notebook process. Modal supports these environment variables as client credentials and they take precedence over local profile configuration.

Example notebook cell:

```python
import os
from kaggle_secrets import UserSecretsClient

secrets = UserSecretsClient()
os.environ["MODAL_TOKEN_ID"] = secrets.get_secret("MODAL_TOKEN_ID")
os.environ["MODAL_TOKEN_SECRET"] = secrets.get_secret("MODAL_TOKEN_SECRET")
```

Install the current Modal client in the Kaggle notebook if needed, then upload:

```bash
python -m pip install -q 'modal>=1.1,<2'
modal volume put small-llm-data \
  /kaggle/working/modal-2b-b64-dataset-001 \
  /datasets/modal-2b-b64-dataset-001
```

Verify the destination exists:

```bash
modal volume ls small-llm-data /datasets/modal-2b-b64-dataset-001
```

The Modal training runtime performs its own full schema-v2 verification on first use, so the upload is not treated as sufficient proof of correctness by itself.

## Launch after upload

From the normal local Small-LLM checkout with the authenticated Modal client:

```bash
modal run --detach modal/launch.py --model 100M --tokens 2B --gpu H100
```

The run then probes microbatch 16/32/48/64 and freezes the fastest safe candidate before optimizer step 1. Checkpoints remain durable in `small-llm-runs`, with validation/checkpoint cadence every 250 successful updates and W&B exact-resume telemetry.
