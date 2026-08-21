# R-SFT dataset production

The active dataset-production path is modular and source-oriented.

```text
build.py                 main entry point / source registry / final token-budget assembly
common.py                shared canonical record, token, hash, and JSONL contracts
over_context.py          source-agnostic curation + GemRouter 2,048-token repair
sources/
  superior_reasoning.py  one adapter for Superior Reasoning Stage 1 and Stage 2
```

The older top-level `superior_reasoning.py`, `scale_superior_reasoning.py`,
`adapt_superior_reasoning.py`, `resume_superior_keep_adaptation.py`, and
`review_superior_candidates.py` remain historical compatibility/reproduction
paths for already-frozen Stage-1 experiments. New corpus creation should use
`build.py`.

## Current 1% build

The frozen 16,716-row Stage-1-expanded corpus remains the base. The main builder
therefore defaults to adding Superior Reasoning Stage 2 only.

```bash
python post_training/R-SFT/dataset/build.py build \
  --work-dir artifacts/rsft-1pct-work \
  --output-jsonl artifacts/rsft-superior-1pct/reasoning.jsonl \
  --percent 1
```

The command streams the canonical Stage-2 JSONL directly from Hugging Face,
selects `instruction_following`, applies the same math/code/output/dedup policy as
Stage 1, validates exact atomic 2,048-token serialization, and writes generic
intermediate state:

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

If the context-fit Stage-2 rows are sufficient, `build` directly freezes the 1%
reasoning JSONL. No GemRouter quota is spent.

For local/offline source files, use:

```bash
python post_training/R-SFT/dataset/build.py build \
  --work-dir artifacts/rsft-1pct-work \
  --output-jsonl artifacts/rsft-superior-1pct/reasoning.jsonl \
  --percent 1 \
  --superior-stage2-jsonl /path/to/stage2.jsonl
```

A clean rebuild from both Superior stages is supported by the same source file:

```text
--superior-stages stage1,stage2
```

For the current scaling experiment this is intentionally not the default because
Stage 1 is already represented by the frozen 16,716-row base.

## Generic over-context repair

If the fit pool cannot reach the requested token budget, `build` fails closed and
points to `over_context/candidates.jsonl`. These records are not training data yet.

Curate them using the generic schema:

```json
{"schema":"small-llm-rsft-manual-curation-v1","id":"...","decision":"keep","reason":"..."}
```

Allowed decisions remain `keep`, `exclude_math`, `exclude_code`, and
`exclude_safety`.

Prepare only the curated keepers:

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

Finalize all accepted rewrites into the generic adapted stream:

```bash
python post_training/R-SFT/dataset/build.py adapt-finalize \
  --work-dir artifacts/rsft-1pct-work
```

Then assemble again:

```bash
python post_training/R-SFT/dataset/build.py assemble \
  --work-dir artifacts/rsft-1pct-work \
  --output-jsonl artifacts/rsft-superior-1pct/reasoning.jsonl \
  --percent 1 \
  --include-adapted
```

`over_context.py` uses the exact same Variant-D fidelity-first prompt as the
completed Superior Stage-1 adaptation (SHA-256
`4c971237585acac842ed9b5417eb4d231338a0fa813ef08a377e002a99e080b9`). The
module is intentionally source-agnostic so future reasoning datasets reuse the
same curation, Gemini-only health gate, batching, retry/split recovery, and exact
2,048-token validator.

## Adding another reasoning source

Create one module under `sources/` that emits the common fit/candidate contract,
then register it in `build.py`. Source-specific parsing and semantic filtering
belong in the adapter. Do not implement another GemRouter repair path or another
final percentage assembler.
