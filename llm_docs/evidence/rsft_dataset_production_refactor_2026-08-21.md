# R-SFT dataset-production refactor — 2026-08-21

## Decision

ADR 0115 refactors new R-SFT corpus creation into source adapters, generic over-context repair, and one main builder. The completed Stage-1 artifacts and their historical scripts remain immutable/reproducible; the new active path does not rewrite them.

## Active implementation

The new production interface is:

```text
post_training/R-SFT/dataset/build.py
post_training/R-SFT/dataset/common.py
post_training/R-SFT/dataset/over_context.py
post_training/R-SFT/dataset/sources/superior_reasoning.py
post_training/R-SFT/dataset/sources/__init__.py
```

`build.py` owns the source registry, aggregate deduplication, token-budget selection, and the `build / prepare / assemble / adapt-*` CLI. The registry currently contains only Superior Reasoning.

`common.py` owns the source-neutral canonical five-field R-SFT record, normalized-prompt identity, exact atomic-context accounting, assistant loss-target accounting, stable training identity, and JSONL/hash utilities.

`sources/superior_reasoning.py` is the single active Superior adapter for both Stage 1 and Stage 2. Both stages share the same instruction-only/no-primary-math/no-primary-code/strict-think-output policy and exact 2,048-token fit check. The current 1% command defaults to Stage 2 because the frozen 16,716-row base already contains the accepted Stage-1-expanded data.

`over_context.py` is source-agnostic. It owns explicit curation, the Gemini-only GemRouter health gate, maximum-size-four requests, the existing Variant-D fidelity-first compression prompt, strict JSON validation, harder retry prompt, recursive batch split recovery, exact 2,048-token post-rewrite validation, resumable accepted batches, status, and final adapted JSONL output.

## Prompt identity

The generic over-context module deliberately keeps the prior Superior adaptation prompt byte-for-byte unchanged.

```text
SHA-256: 4c971237585acac842ed9b5417eb4d231338a0fa813ef08a377e002a99e080b9
```

That hash matches the already-frozen Stage-1 adaptation candidate manifest, so the refactor changes module ownership rather than teacher policy.

## Upstream source verification

The live `Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b` repository was checked on 2026-08-21. Its `main` tree exposes both canonical files used by the unified adapter:

```text
Superior-Reasoning-SFT-gpt-oss-120b-stage1-train-data.jsonl  4.6 GB
Superior-Reasoning-SFT-gpt-oss-120b-stage2-train-data.jsonl 20.2 GB
```

The active adapter streams these JSONL files directly through HTTP. The Stage-2 build no longer requires adding the Hugging Face `datasets`/`fsspec` stack solely for source streaming.

## Validation performed

The new modules were syntax-compiled and exercised in an isolated synthetic repo layout with a stub transport and deterministic token counter.

Validated behavior included:

- one Superior adapter processing Stage 1 and Stage 2 in the same invocation;
- `instruction_following` selection while ignoring non-instruction domains;
- unchanged primary-code exclusion;
- context-fit rows emitted to `fit.jsonl` and >2,048 rows emitted to generic candidates;
- a frozen reasoning base plus Stage-2 fit pool assembled by projected loss-bearing train targets;
- generic curation using a deliberately non-Superior source name, proving the keep/exclude layer is source-agnostic;
- prompt hash equality with the frozen Variant-D prompt;
- work-directory identity pinning to the base hash, selected Superior stages, and any local source JSONL hashes.

Focused repository regression coverage is in:

```text
tests/test_reasoning_sft_dataset_refactor.py
```

No successful GitHub Actions status was visible from the available connector during this refactor, so this evidence does not claim a remote CI run.

## Current 1% command

```bash
python post_training/R-SFT/dataset/build.py build \
  --work-dir artifacts/rsft-1pct-work \
  --output-jsonl artifacts/rsft-superior-1pct/reasoning.jsonl \
  --percent 1
```

The builder first tries context-fit Stage-2 rows only. If they reach the requested reasoning target, the 1% corpus is frozen with no GemRouter calls. If they are insufficient, the command fails closed and leaves the source-neutral over-context pool under `artifacts/rsft-1pct-work/over_context/candidates.jsonl`; explicit curation and the generic `adapt-*` commands are then required before `assemble --include-adapted`.

## Legacy status

The old top-level `superior_reasoning.py`, `scale_superior_reasoning.py`, `adapt_superior_reasoning.py`, `resume_superior_keep_adaptation.py`, and `review_superior_candidates.py` remain in place for historical artifact reproduction and test compatibility. `post_training/R-SFT/dataset/README.md` marks them as legacy paths. They should not be copied or extended when future reasoning sources are added.
