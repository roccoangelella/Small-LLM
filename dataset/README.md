# Nemotron-ClimbMix dataset pipeline

This package owns deterministic source selection, structural validation, cluster filtering, context-plus-one packing, immutable shards, remote durability, resume, and evaluation-set construction.

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

- `dataset.qualification`: the experiment-facing finite-dataset CLI and single registry for frozen 10M/100M/500M/2B profile details. Active profiles can be built; all profiles can derive a trainer plan.
- `dataset.production`: reusable schema-v2 producer implementation with exact mixture accounting, verified Google Drive durability, locking, and resume. Normally invoked through `dataset.qualification build` for a fixed scaling run.
- `dataset.qualification_report`: shared manifest/Drive validation and exact one-pass WSD plan engine used by the profile CLI.
- `dataset.eval_core`, `dataset.eval_core_accelerated`, `dataset.eval_core_cli`: reusable `eval_core_v1` construction and verification. Eval-core is intentionally separate from the training-cache producer.
- `dataset.main`: shared schema-v2 verification plus the low-level stream-cache development surface still covered by streaming tests.
- `dataset.drive_auth`: Google Drive credential loading used by production, plus the infrequent interactive OAuth setup command.
- `dataset.src`: shared producer, streaming, storage, work-plan, retry, verification, and remote-backend primitives.

## Finite dataset profiles

List the exact frozen identities:

```bash
uv run python -m dataset.qualification profiles
```

Build or resume the current 2B profile without restating its scientific/storage geometry:

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

The profile fixes run ID, source-token envelope, checkpoint cadence, context length, sequences per block, shard size, and remote-durability requirement. Reader/queue tuning remains available through the underlying production CLI.

Derive the exact one-pass trainer plan from a completed cache:

```bash
uv run python -m dataset.qualification report \
  --profile 20m-2b \
  --dataset-dir /data/small-llm/20m-2b-dataset-001 \
  --drive-manifest /data/small-llm/20m-2b-dataset-001/drive_manifest.json \
  --output /data/small-llm/20m-2b-dataset-001/qualification_plan.json
```

The historical `20m-10m` profile remains reportable for reproducibility but is intentionally not buildable again.

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
