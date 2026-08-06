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

## Supported paths

- `dataset.production`: production schema-v2 cache with exact mixture accounting, verified Google Drive durability, locking, resume, and migration.
- `dataset.qualification_100m`: fixed producer for the current 20M-model/100M-token experiment.
- `dataset.eval_core`: deterministic `eval_core_v1` fast/full held-out corpus builder and verifier.
- `dataset.main`: lower-level build, status, stream-cache validation, and shared full-scan verification utilities.

The obsolete decoded-text/LLM-review pipeline has been deleted. Do not recreate it from the archive; Git history is sufficient if its implementation ever needs to be studied.

## Environment

```bash
uv sync --locked
uv pip install -r dataset/requirements-remote.txt
```

## Production build

Use the current experiment runbook rather than assembling flags manually:

- [`../llm_docs/runbooks/20m_100m_runbook.md`](../llm_docs/runbooks/20m_100m_runbook.md)
- [`../llm_docs/reference/dataset_and_tokenization.md`](../llm_docs/reference/dataset_and_tokenization.md)

The general production command remains:

```bash
uv run --env-file .env python -m dataset.production \
  --weights-file /path/to/approved-weights.json \
  --output-dir /data/climbmix-cache \
  --run-id climbmix-production-v1
```

Add `--resume` to continue the same immutable production identity.

## Evaluation corpus

```bash
small-llm-eval-data build --output-dir /data/eval_core_v1
small-llm-eval-data verify --eval-dir /data/eval_core_v1
```

## Tests

```bash
uv run --extra model --with-requirements dataset/requirements-remote.txt \
  python -m unittest discover -v
```

Dataset production remains fail-closed: source revision, selection seed, split identity, mixture weights, sequence geometry, shard hashes, and remote durability must match the recorded contract.
