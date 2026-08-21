# R-SFT Stage-2 scaling runbook

_Last updated: 2026-08-21 Europe/Rome_

This runbook implements ADR 0111/0112. The frozen 16,716-row Stage-1-expanded reasoning corpus is the immutable base; Superior Reasoning Stage 2 `instruction_following` is additive scaling data for the nested 1% / 2% / 4% sweep.

## Contract

Stage 2 must pass the same R0 processing contract used for Stage 1:

- strict source `<think>...</think>` parsing and non-empty final answer;
- normalized-prompt deduplication against the full frozen base and within Stage 2;
- primary math/computation/proof and primary programming/code exclusion;
- reserved `<think>`, `</think>`, `<answer>` collision rejection;
- exact atomic 2,048-token chat serialization, never truncation;
- unchanged context-fit examples enter directly;
- over-context examples become adaptation candidates only;
- over-context rows require explicit curation before teacher traffic;
- only curated keepers may use ADR-0103 Variant-D compression;
- every rewrite must pass the same exact 2,048-token validator and final deduplication;
- GemRouter teacher traffic remains Gemini-only, fallback disabled, NVIDIA disabled.

The upstream text dataset is read as config `stage2`, split `train`; the producer then selects rows whose `domain == instruction_following`. Do not use the domain-specific splits from the separate logprob companion dataset as a substitute for the canonical text source.

## Prepare Stage 2

Install the dataset-streaming dependency in the producer environment, then run:

```bash
python post_training/R-SFT/dataset/scale_superior_reasoning.py prepare-stage2 \
  --base-jsonl artifacts/rsft-superior-instruction-r0-expanded/reasoning.jsonl \
  --work-dir /path/to/rsft-stage2
```

This writes:

```text
/path/to/rsft-stage2/fit.jsonl
/path/to/rsft-stage2/candidates.jsonl
/path/to/rsft-stage2/candidates.manifest.json
```

`fit.jsonl` contains only unchanged Stage-2 instruction rows that already satisfy the complete Stage-1 policy and exact context limit. `candidates.jsonl` contains only otherwise-clean over-context rows; it is not itself training data.

For an already downloaded raw Stage-2 JSONL, pass:

```bash
--source-jsonl /path/to/Superior-Reasoning-SFT-gpt-oss-120b-stage2-train-data.jsonl
```

The same `domain == instruction_following` filter is still applied.

## Build the 1% corpus

The parent identity is 2,001,000,448 training targets. A nominal 1% R-SFT run requests 20,010,004 total loss-bearing targets; the 90/10 contract therefore requests approximately 18,009,004 reasoning train targets and derives the retention amount from the realized reasoning prefix.

First try the context-fit Stage-2 pool without spending Gemini quota:

```bash
python post_training/R-SFT/dataset/scale_superior_reasoning.py build-percent \
  --base-jsonl artifacts/rsft-superior-instruction-r0-expanded/reasoning.jsonl \
  --work-dir /path/to/rsft-stage2 \
  --output-jsonl /path/to/rsft-1pct/reasoning.jsonl \
  --percent 1
```

The assembler measures loss-bearing **train reasoning targets**, not bytes, serialized input tokens, or row count. It mirrors `build_atomic.py`'s stable per-group 1% validation + 1% test partition before measuring the train target count.

Stage-2 additions are ordered by their deterministic `instruction_following` stream index. This is the nesting contract: a later, larger Stage-2 scan can only append source rows, so the selected Stage-2 prefix for 1% remains literally contained in the 2% and 4% prefixes.

## If context-fit rows are insufficient

Do not relax filters or repeat epochs. Instead:

1. Curate every Stage-2 over-context candidate under the same `keep / exclude_math / exclude_code / exclude_safety` policy used by the completed Stage-1 expansion.
2. Freeze the curation identity.
3. Send only curated keepers through the existing keeper-only Variant-D GemRouter path, with Gemini-only health/fallback checks from ADR 0106.
4. Assemble validated rewrites as `adapted.jsonl` in the Stage-2 work directory.
5. Retry:

```bash
python post_training/R-SFT/dataset/scale_superior_reasoning.py build-percent \
  --base-jsonl artifacts/rsft-superior-instruction-r0-expanded/reasoning.jsonl \
  --work-dir /path/to/rsft-stage2 \
  --output-jsonl /path/to/rsft-1pct/reasoning.jsonl \
  --percent 1 \
  --include-adapted
```

The assembler fails closed if `--include-adapted` is requested but `adapted.jsonl` is absent.

## 2% and 4%

After the 1% corpus is frozen, use the same prepared/adapted pool and only change `--percent` and output path:

```text
--percent 2
--percent 4
```

Do not regenerate or reorder the earlier source prefix. The expected invariant is:

```text
16,716-row frozen base ⊂ 1% ⊂ 2% ⊂ 4%
```

Repeated epochs are not part of this scaling experiment.
