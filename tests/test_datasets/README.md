# Local test datasets

This directory is the canonical **VPS-local** home for large qualification datasets. The payloads are ignored by git; only this README and `.gitignore` are committed.

For the 100M / 2B SFT full qualification, place the datasets exactly as follows:

```text
tests/test_datasets/
├── eval_core_v1/
│   ├── manifest.json
│   ├── fast.bin
│   ├── fast.records.jsonl
│   ├── full.bin
│   └── full.records.jsonl
└── 100m-2b-sft-s0-001/
    ├── bundle-manifest.json
    ├── validation/
    ├── test/
    └── ...the remaining files from the verified SFT bundle
```

The `eval_core_v1` directory should be the same frozen corpus previously attached to Kaggle. The SFT directory should be the complete verified bundle published as `roccoangelella/small-llm-100m-2b-sft-s0-001`; do not copy only its validation/test shards because bundle verification checks the complete manifest contract.

The launcher verifies both local datasets before loading either model checkpoint. Missing or modified data therefore fails before expensive GPU evaluation begins.

Run the complete qualification from the repository root with:

```bash
python -m tests.qualification.sft_100m_2b_vps --suite full
```

The launcher loads `.env` automatically when present. `SMALL_LLM_SFT_HF_REPO_ID` is used as the default repository for both the completed 100M parent and the SFT checkpoint; `HF_TOKEN` is used for private Hugging Face access. CLI repository or local-checkpoint overrides are also available through `--help`.

Default output:

```text
artifacts/100m-2b-sft-full-qualification.json
```
