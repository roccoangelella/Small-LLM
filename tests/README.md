# Tests and qualification

The `tests/` tree has three distinct roles. Keep them separate:

```text
tests/
├── test_*.py                 ordinary CPU/unit/integration regressions
├── qualification/            explicit GPU/model qualification launchers
├── test_datasets/            large VPS-local qualification corpora (gitignored)
├── production_helpers.py     shared production-test helpers
├── synthetic.py              shared synthetic dataset helpers
├── trainer_fixtures.py       shared trainer test fixtures
└── t4_qualification.py       historical hardware-acceptance harness
```

Files named `test_*.py` remain at the package root because the repository's current CI and local command use `unittest discover`. They should not be deleted or moved merely because they are old; prune them only when the production path they protect is removed or the coverage is demonstrably duplicated.

## Ordinary repository tests

From the repository root:

```bash
python -m unittest discover -v
```

For a single module:

```bash
python -m unittest -v tests.test_vps_sft_qualification
```

These tests should remain CPU-friendly unless the module explicitly documents a hardware requirement.

## VPS full SFT qualification

Kaggle is no longer required for the 100M / 2B parent-versus-SFT qualification. Put the frozen evaluation data under `tests/test_datasets/` as documented in [`test_datasets/README.md`](test_datasets/README.md), then run:

```bash
python -m tests.qualification.sft_100m_2b_vps --suite full
```

The launcher:

1. loads `.env` when present without overwriting already-exported variables;
2. verifies the complete local SFT bundle;
3. verifies the frozen local `eval_core_v1` corpus;
4. resolves the immutable 100M / 2B parent and completed SFT checkpoint from Hugging Face, or accepts explicit local checkpoint directories;
5. runs the existing provider-neutral `post_training.sft.eval_suite` on CUDA/FP16 by default;
6. writes `artifacts/100m-2b-sft-full-qualification.json` unless `--output` is supplied.

Use `--help` for repository, checkpoint, device, precision, block-count, and output overrides.

## Local qualification datasets

Large corpora are never committed. The canonical layout is:

```text
tests/test_datasets/eval_core_v1/
tests/test_datasets/100m-2b-sft-s0-001/
```

This replaces the old requirement to upload/attach those datasets to Kaggle simply to run qualification.

## Historical T4 hardware harness

`tests/t4_qualification.py` is a separate hardware-acceptance harness from the early dual-T4 qualification work. It is retained because current regression tests and archived runbooks still reference its parity/recommendation logic. It is **not** part of the normal full SFT qualification command above.

Run its CPU control tests with:

```bash
python -m unittest -v tests.test_t4_qualification
```

Only invoke the hardware harness itself when intentionally reproducing that historical CUDA qualification path.
