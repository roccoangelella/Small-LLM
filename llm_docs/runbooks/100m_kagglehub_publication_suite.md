# 100M KaggleHub Publication Suite

_Last updated: 2026-08-05 15:34 Europe/Rome_

## Decision

The fixed 20M-model / 100M-token experiment will use one VPS command to build, verify, stage, and privately publish the training dataset to Kaggle.

The publication client is the official `kagglehub` Python library. Authentication is supplied through `KAGGLE_API_TOKEN` in the repository `.env` file. Dataset ownership is supplied through either `KAGGLE_USERNAME` or the exact non-secret `SMALL_LLM_KAGGLE_DATASET_HANDLE=owner/dataset` value.

## Official command

```bash
cd /path/to/Small-LLM
git switch main
git pull --ff-only
bash kaggle/build_and_push_100m.sh
```

The wrapper loads `.env`, pins Python 3.13 and `kagglehub==1.0.2`, and invokes `kaggle/build_and_push_100m.py`.

## Required `.env` values

```env
KAGGLE_API_TOKEN=<token from Kaggle settings>
KAGGLE_USERNAME=<Kaggle owner slug>
SMALL_LLM_GOOGLE_OAUTH_TOKEN=.secrets/google-drive-authorized-user.json
SMALL_LLM_DRIVE_FOLDER_ID=<existing qualified Drive folder ID>
```

Instead of `KAGGLE_USERNAME`, an exact handle may be supplied:

```env
SMALL_LLM_KAGGLE_DATASET_HANDLE=owner/small-llm-20m-100m-dataset-001
```

The default local paths are:

```text
weights: /data/climbmix-mixture-calibration/climbmix_code_free_weights.json
dataset: /data/small-llm/20m-100m-dataset-001
operations/evidence: /data/small-llm/20m-100m-ops
```

They may be overridden with the corresponding `SMALL_LLM_100M_*` environment variables documented in `kaggle/100m-publish.env.example`.

## Suite behavior

The one command:

1. requires a clean Small-LLM checkout and the fixed credentials;
2. starts the canonical `dataset.qualification_100m` producer when the output directory is absent;
3. automatically uses `--resume` when an interrupted output directory exists;
4. skips production when the fixed completed manifest is already present;
5. runs a literal full local shard scan;
6. derives `qualification_plan.json` and binds it to the local and Drive manifests;
7. stages only the training-facing shape:
   - `manifest.json`
   - `drive_manifest.json`
   - `qualification_plan.json`
   - `train/`
   - `validation/`
8. computes one deterministic tree identity for the staged dataset;
9. refuses a Kaggle handle that is anonymously readable before publication;
10. uploads with `kagglehub.dataset_upload`;
11. downloads the complete private Kaggle dataset back to the VPS;
12. requires byte-identical tree identity and another literal full scan;
13. checks that anonymous access is denied after publication;
14. writes an idempotent publication receipt and skips duplicate versions on later identical invocations.

`kagglehub` creates new datasets with its create request explicitly marked private. The suite also performs the anonymous-access check so an existing public handle cannot be used silently.

## Evidence

```text
/data/small-llm/20m-100m-ops/logs/
/data/small-llm/20m-100m-ops/kaggle-publish-state.json
/data/small-llm/20m-100m-ops/build-and-push-summary.json
/data/small-llm/20m-100m-ops/kaggle-dataset/
/data/small-llm/20m-100m-ops/kaggle-roundtrip/
```

Success requires the summary status to be `completed` or `already_published` and the publication state to be `verified`.

## Scientific boundary

This suite changes only dataset production and delivery ergonomics. It does not modify the fixed 100M source envelope, tokenizer, source revision, mixture, shard geometry, model, optimizer, seed, schedule, precision, effective optimizer batch, or one-pass training policy.
