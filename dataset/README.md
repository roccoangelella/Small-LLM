# Nemotron-ClimbMix dataset pipeline

This package owns deterministic source selection, structural validation, cluster filtering, context-plus-one packing, immutable shards, remote durability, resume, rolling-cache transport, and evaluation-set construction.

## Frozen source contract

```text
repository: nvidia/Nemotron-ClimbMix
revision: 5eaa64b9c0c85b7f56af01d7dffdb0795816b12b
tokenizer: existing GPT-2 token IDs
accepted clusters: 1-10 and 12-20
excluded cluster: 11
validation split: deterministic document hash, probability 0.001
approved mixture-weight SHA-256: 76e82e22760adcac59c7294fe9bac11358f5a8b7a26035aae64c3f2e6fa1acb7
```

The explicit programming cluster is excluded; the resulting corpus is not guaranteed code-free.

## Active modules

- `dataset.qualification`: experiment-facing finite-dataset CLI and single registry for frozen profiles, including the block-64 Modal profiles.
- `dataset.production`: reusable schema-v2 producer with exact mixture accounting, verified Hugging Face Storage Bucket durability, bounded-disk shard eviction, locking, and resume.
- `dataset.rolling_cache`: checkpoint-aligned CPU staging plus current/next verified local shard caching for remotely stored datasets.
- `dataset.qualification_report`: shared manifest validation and exact one-pass WSD plan engine. Its optional `drive_manifest` input is legacy compatibility for already-built historical datasets.
- `dataset.eval_core`, `dataset.eval_core_accelerated`, `dataset.eval_core_cli`: reusable `eval_core_v1` construction and verification. Eval-core is intentionally separate from the training-cache producer.
- `dataset.main`: shared schema-v2 verification plus the low-level stream-cache development surface still covered by streaming tests.
- `dataset.src`: shared producer, streaming, storage, work-plan, retry, verification, HF bucket, checkpoint, and compatibility primitives.

Google Drive is not an active storage backend. New remote dataset production uses a private Hugging Face Storage Bucket. The historical filename `drive_manifest.json` and legacy `drive_file_id` fields remain readable only because existing datasets and checkpoints already bind those names into their identity.

## Finite dataset profiles

List the exact frozen identities:

```bash
uv run python -m dataset.qualification profiles
```

Build or resume a finite dataset without restating its scientific/storage geometry:

```bash
uv run --env-file .env python -m dataset.qualification build \
  --profile 20m-2b \
  --weights-file /path/to/approved-weights.json \
  --output-dir /data/small-llm/20m-2b-dataset-001

uv run --env-file .env python -m dataset.qualification build \
  --profile 20m-2b \
  --weights-file /path/to/approved-weights.json \
  --output-dir /data/small-llm/20m-2b-dataset-001 \
  --resume
```

Remote builds require `HF_TOKEN` and either `SMALL_LLM_HF_DATASET_BUCKET_ID` or `SMALL_LLM_HF_REPO_ID`; the latter derives `<repo>-datasets`. The `modal-10b-b64` profile additionally freezes one-GiB shards and verified local eviction so the producer never needs enough disk for the complete derived corpus.

The profile fixes run ID, source-token envelope, checkpoint cadence, context length, sequences per block, shard size, and remote-durability requirement. Reader/queue tuning remains available through the underlying production CLI.

Derive the exact one-pass trainer plan from a completed current-format cache:

```bash
uv run python -m dataset.qualification report \
  --profile 20m-2b \
  --dataset-dir /data/small-llm/20m-2b-dataset-001 \
  --output /data/small-llm/20m-2b-dataset-001/qualification_plan.json
```

For an already-built historical dataset whose identity included `drive_manifest.json`, pass `--drive-manifest` explicitly when reproducing its old qualification plan. This does not activate or require Google Drive.

The historical `20m-10m` profile remains reportable for reproducibility but is intentionally not buildable again.

## Rolling reuse

Remote shard storage is independent of training epochs. A future multi-pass experiment can traverse the same immutable HF shard sequence again: while the current shard trains, the rolling cache can prefetch the next logical shard, including wrapping from the final shard back to shard zero at an epoch boundary. The current trainer remains one-pass; adding repeated epochs requires an epoch-aware logical cursor/checkpoint contract, not another storage backend or another corpus copy.

## Evaluation corpus

```bash
small-llm-eval-data build --output-dir /data/eval_core_v1
small-llm-eval-data verify --eval-dir /data/eval_core_v1
```

## Retired one-time tooling

The full ~2 TB cluster-mixture calibration and the original 10M operational-acceptance suite were qualification jobs, not recurring production dependencies. Their active executables and tests were removed after acceptance. The approved mixture hash, measurements, and acceptance evidence remain under `llm_docs/`; the standalone reproducible calibration implementation is published in `roccoangelella/climbmix-token-mixture`. Historical commands remain readable in archived evidence and Git history.

The earlier per-budget modules (`qualification_100m.py`, `qualification_500m.py`, `qualification_2b.py` and matching report wrappers) were also removed. Add future finite budgets as one profile row in `dataset.qualification`, not as new wrapper files.

## Environment and tests

```bash
uv sync --locked
uv pip install -r dataset/requirements-remote.txt
uv run --extra model --with-requirements dataset/requirements-remote.txt \
  python -m unittest discover -v
```

Dataset production remains fail-closed: source revision, selection seed, split identity, profile identity, mixture weights, sequence geometry, shard hashes, and remote durability must match the recorded contract.
