# VPS SFT qualification

Use the persistent VPS for the canonical 100M / 2B parent-versus-SFT qualification. Kaggle is not required.

## One-time dataset placement

Copy the frozen datasets into:

```text
tests/test_datasets/eval_core_v1/
tests/test_datasets/100m-2b-sft-s0-001/
```

The eval-core directory must contain its complete verified `manifest.json`, binary suite files, and records files. The SFT directory must contain the complete verified bundle headed by `bundle-manifest.json`; do not keep only the validation/test shards.

See `tests/test_datasets/README.md` for the exact convention. Payload files are gitignored.

## Environment

Activate the project environment:

```bash
source .venv/bin/activate
```

The launcher reads the repository `.env` itself. At minimum, private Hugging Face resolution normally needs:

```text
HF_TOKEN=...
SMALL_LLM_SFT_HF_REPO_ID=roccoangelella/small-llm-100m-qualification
```

For this profile the completed parent and SFT checkpoints share the same qualification repository by default. Explicit `--repo-id`, `--parent-repo-id`, `--sft-repo-id`, `--parent-checkpoint-dir`, and `--sft-checkpoint-dir` overrides are available.

## Full qualification

From the repository root:

```bash
python -m tests.qualification.sft_100m_2b_vps --suite full
```

Default execution is CUDA + FP16. The launcher first verifies both local datasets, then scores the immutable pretrained parent followed by the SFT checkpoint using the existing `post_training.sft.eval_suite` implementation.

Default report:

```text
artifacts/100m-2b-sft-full-qualification.json
```

## Fast qualification

```bash
python -m tests.qualification.sft_100m_2b_vps --suite fast
```

## Useful overrides

```bash
python -m tests.qualification.sft_100m_2b_vps \
  --suite full \
  --dataset-dir /absolute/path/to/100m-2b-sft-s0-001 \
  --eval-dir /absolute/path/to/eval_core_v1 \
  --repo-id roccoangelella/small-llm-100m-qualification \
  --output artifacts/custom-post-sft-full.json
```

The default full suite keeps the evaluator's current 32 SFT validation blocks, 32 SFT test blocks, batch size 1, and 200 bootstrap samples unless explicitly overridden.
