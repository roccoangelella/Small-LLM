# 100M KaggleHub Publication Suite

_Last updated: 2026-08-10 Europe/Rome_

## Decision

The fixed 20M-model / 100M-token dataset is built, verified, staged, and privately published through the same canonical CLI used by the later scaling profiles.

The publication client remains the official `kagglehub` Python library. Authentication is supplied through `KAGGLE_API_TOKEN` in the repository `.env` file. Dataset ownership is supplied through either `KAGGLE_USERNAME` or the exact non-secret `SMALL_LLM_KAGGLE_DATASET_HANDLE=owner/dataset` value.

## Official command

```bash
cd /path/to/Small-LLM
git switch main
git pull --ff-only
python kaggle/launch.py publish --model 20M --tokens 100M
```

There is no longer a profile-specific shell wrapper. The Python launcher preserves the old bootstrap contract itself: it requires `uv` and `.env`, then re-executes under Python 3.13 with `kaggle/requirements-100m-publish.txt` installed.

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

The default local paths remain:

```text
weights: /data/climbmix-mixture-calibration/climbmix_code_free_weights.json
dataset: /data/small-llm/20m-100m-dataset-001
operations/evidence: /data/small-llm/20m-100m-ops
```

They may be overridden through `SMALL_LLM_100M_*` environment variables or the stable `launch.py publish` arguments.

## Suite behavior

The unified command:

1. requires a clean Small-LLM checkout and the fixed credentials;
2. starts `dataset.qualification build --profile 20m-100m` when producer output is absent;
3. automatically resumes an interrupted output directory;
4. skips production when the fixed completed manifest already exists;
5. runs a literal full local shard scan;
6. derives `qualification_plan.json` and binds it to the local and Drive manifests;
7. stages only `manifest.json`, `drive_manifest.json`, `qualification_plan.json`, `train/`, and `validation/`;
8. computes a deterministic tree identity while ignoring only KaggleHub's root-level numeric `.archive` transport artifact;
9. refuses a Kaggle handle that is anonymously readable before publication;
10. uploads with `kagglehub.dataset_upload`;
11. downloads the complete private Kaggle dataset back to the VPS;
12. requires byte-identical tree identity and another literal full scan;
13. checks that anonymous access is denied after publication;
14. writes an idempotent publication receipt and skips duplicate versions on later identical invocations.

Rerun the identical command after interruption. Do not pass `--resume`; publication resume is automatic and fail-closed.

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

This suite changes only production/delivery command organization. It does not modify the fixed 100M source envelope, tokenizer, source revision, mixture, shard geometry, model, optimizer, seed, schedule, precision, effective optimizer batch, or one-pass training policy.
