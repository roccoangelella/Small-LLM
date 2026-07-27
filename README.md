# Small-LLM

This repository is working toward a small English language model. The part that
is ready now is the pretraining-corpus builder in `dataset/`.

The builder takes NVIDIA Nemotron-ClimbMix at the immutable revision
`5eaa64b9c0c85b7f56af01d7dffdb0795816b12b`, samples deterministic byte regions,
keeps clusters 1–10 and 12–20, and writes the existing GPT-2 token IDs directly
to `train.bin` and `validation.bin`. Cluster 11 is the software/programming
cluster, so it is excluded. There is no decoded-text curation pass, code filter,
quality classifier, quota balancing, or LLM review in production.

The production commands are deliberately boring:

```bash
uv run python -m dataset.main build
uv run python -m dataset.main build --resume
uv run python -m dataset.main verify
```

Do not run the first command casually: the default target is 90B accepted source
tokens and the preflight expects roughly 239 GB of free space. See
[dataset/README.md](dataset/README.md) for the format, checkpoint contract,
bounded-test commands, and license details. The successful live source test is
recorded in
[dataset/PRODUCTION_SMOKE_TEST.md](dataset/PRODUCTION_SMOKE_TEST.md).
