# R-SFT Stage-2 scaling runbook

_Last updated: 2026-08-21 Europe/Rome_

This runbook implements ADR 0112/0113 and the dataset-production refactor in ADR 0114. The frozen 16,716-row Stage-1-expanded reasoning corpus is the immutable base; Superior Reasoning Stage 2 `instruction_following` is additive scaling data for the nested 1% / 2% / 4% sweep.

## Active architecture

New R-SFT corpus creation uses:

```text
post_training/R-SFT/dataset/build.py
post_training/R-SFT/dataset/common.py
post_training/R-SFT/dataset/over_context.py
post_training/R-SFT/dataset/sources/superior_reasoning.py
```

The Superior source adapter handles both Stage 1 and Stage 2. Historical `scale_superior_reasoning.py`, `adapt_superior_reasoning.py`, and keeper-resume scripts remain reproduction/compatibility paths for the already-completed Stage-1 artifacts; do not use them for new scaling corpora.

## Contract

Superior Stage 2 passes the same R0 processing contract used for Stage 1:

- strict source `<think>...</think>` parsing and non-empty final answer;
- normalized-prompt deduplication against the full frozen base and within the new source stream;
- `instruction_following` only;
- primary math/computation/proof and primary programming/code exclusion;
- reserved `<think>`, `</think>`, `<answer>` collision rejection;
- exact atomic 2,048-token chat serialization, never truncation;
- unchanged context-fit examples enter directly;
- over-context examples become generic adaptation candidates only;
- over-context rows require explicit curation before teacher traffic;
- only curated keepers may use Variant-D compression;
- every rewrite must pass the same exact 2,048-token validator and final deduplication;
- GemRouter teacher traffic remains Gemini-only with fallback disabled.

The unified Superior adapter streams the canonical Stage-1/Stage-2 JSONL files directly from Hugging Face, so no `datasets`/`fsspec` dependency is required for this lane.

## Build the 1% corpus

The parent identity is 2,001,000,448 training targets. A nominal 1% R-SFT run requests 20,010,004 total loss-bearing targets; the 90/10 contract therefore requests approximately 18,009,004 reasoning train targets and derives retention from the realized reasoning prefix.

For the current experiment, Stage 1 is already represented by the frozen 16,716-row base, so the builder defaults to Stage 2 only:

```bash
python post_training/R-SFT/dataset/build.py build \
  --work-dir artifacts/rsft-1pct-work \
  --output-jsonl artifacts/rsft-superior-1pct/reasoning.jsonl \
  --percent 1
```

The command performs source preparation and then immediately tries to assemble the 1% reasoning corpus from already-context-fit Stage-2 rows. It measures loss-bearing **train reasoning targets**, not bytes, raw text tokens, serialized input tokens, or row count, and mirrors `build_atomic.py`'s stable 1% validation + 1% test split per reasoning group.

The work directory contains source-specific state plus generic aggregate streams:

```text
artifacts/rsft-1pct-work/
  build.manifest.json
  fit.jsonl
  sources/superior_reasoning/
    fit.jsonl
    candidates.jsonl
    manifest.json
  over_context/
    candidates.jsonl
```

For an already-downloaded Stage-2 source file:

```bash
python post_training/R-SFT/dataset/build.py build \
  --work-dir artifacts/rsft-1pct-work \
  --output-jsonl artifacts/rsft-superior-1pct/reasoning.jsonl \
  --percent 1 \
  --superior-stage2-jsonl /path/to/Superior-Reasoning-SFT-gpt-oss-120b-stage2-train-data.jsonl
```

A future clean source rebuild can use both stages through the same adapter with `--superior-stages stage1,stage2`; this is intentionally not the current 1% default.

## If context-fit rows are insufficient

Do not relax filters or repeat epochs. The failed `build` command will report the achieved reasoning-target count and leave the generic over-context pool at:

```text
artifacts/rsft-1pct-work/over_context/candidates.jsonl
```

Curate every candidate with schema `small-llm-rsft-manual-curation-v1` and one of `keep`, `exclude_math`, `exclude_code`, or `exclude_safety`, then prepare only the keepers:

```bash
python post_training/R-SFT/dataset/build.py adapt-prepare \
  --work-dir artifacts/rsft-1pct-work \
  --curation-jsonl /path/to/manual-curation.jsonl
```

Run resumable GemRouter waves:

```bash
python post_training/R-SFT/dataset/build.py adapt-wave \
  --work-dir artifacts/rsft-1pct-work \
  --first-batch 1 \
  --batch-count 100 \
  --workers 4
```

Check progress:

```bash
python post_training/R-SFT/dataset/build.py adapt-status \
  --work-dir artifacts/rsft-1pct-work
```

After all curated keepers are complete, freeze their validated rewrites:

```bash
python post_training/R-SFT/dataset/build.py adapt-finalize \
  --work-dir artifacts/rsft-1pct-work
```

This writes:

```text
artifacts/rsft-1pct-work/over_context/adapted.jsonl
```

Then retry assembly:

```bash
python post_training/R-SFT/dataset/build.py assemble \
  --work-dir artifacts/rsft-1pct-work \
  --output-jsonl artifacts/rsft-superior-1pct/reasoning.jsonl \
  --percent 1 \
  --include-adapted
```

The generic `over_context.py` prompt is byte-identical to the completed Stage-1 Variant-D prompt (SHA-256 `4c971237585acac842ed9b5417eb4d231338a0fa813ef08a377e002a99e080b9`). The repair module also owns the Gemini-only health gate, maximum-size-four requests, strict JSON parsing, retry correction, recursive split recovery, resumable batch state, and exact atomic 2,048-token post-rewrite validation.

## 2% and 4%

The same main builder and source registry are used for the larger points. The required invariant remains:

```text
16,716-row frozen base ⊂ 1% ⊂ 2% ⊂ 4%
```

When freezing 2% and 4%, preserve the earlier frozen corpus as required input rather than replacing its examples with newly discovered data. Repeated epochs are not part of this scaling experiment.

## Current execution status

The modular producer and focused regression tests are committed on `main`. The first real 1% source-scan attempt remains recorded in `../evidence/rsft_stage2_1pct_build_attempt_2026-08-21.md`; no completed 1% artifact is claimed until the canonical Stage-2 source has actually been scanned and the resulting manifest frozen.
