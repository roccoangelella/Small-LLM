---
status: accepted
date: 2026-08-20
supersedes: null
---

# ADR 0106 — Resume expanded R-SFT with curation v2 and keeper-only Gemini adaptation

## Context and problem statement

ADR 0105 froze and trained the 12,306-row checkpoint while 4,476 manually-kept over-context Superior rows still awaited Variant-D compression. When Gemini quota reset on 2026-08-20, the expansion lane was resumed from the preserved 1,122 accepted historical batches.

Two issues appeared during the live resume. First, GemRouter's configuration automatically appended NVIDIA to the effective backend order whenever the NVIDIA provider was enabled, even when `GEMROUTER_BACKEND_ORDER=gemini-api`. This violates the dataset-generation requirement that the teacher remain Gemini-only. Second, a repeatedly empty Gemini response isolated a previously-kept prompt requesting a naked imprisoned woman. A targeted semantic re-audit found 24 clear misses under the safety policy already used by the original manual curation: explicit nudity/pornography, coercive hypnosis/fetish obedience, sexualized teen scenarios, sexual violence, or sexualized chatbot/persona generation.

The historical curation used by the completed 12,306-row model must remain immutable because its manifest pins that artifact.

## Considered options

- Continue all 9,624 candidates and waste teacher requests on rows already excluded by manual curation.
- Mutate the historical curation file in place and reuse its path.
- Freeze an expansion-only curation v2, adapt only missing curated keepers, preserve all historical accepted batches, and hard-disable NVIDIA at the router runtime.

## Decision outcome

Chosen option: **freeze curation v2 and use a keeper-only Gemini resume stream**.

The expansion curation is:

```text
artifacts/rsft-superior-instruction-r0-adaptation/manual-curation.expanded-v2.jsonl
```

Its SHA-256 is:

```text
fb4da2929b47ececbde839da199437144677e4c7e1ea52ef2e8f6d4525ae1cde
```

It covers the same 9,624 frozen candidates and records:

- 8,473 `keep`;
- 829 `exclude_code`;
- 212 `exclude_math`;
- 110 `exclude_safety`.

Exactly 24 v1 `keep` decisions become `exclude_safety`. The original `manual-curation.final.jsonl` remains unchanged and continues to describe the already-trained 12,306-row checkpoint.

The keeper-only resume implementation is `post_training/R-SFT/dataset/resume_superior_keep_adaptation.py`. It reuses ADR 0103's Variant-D prompt and exact 2,048-token validator, harvests 4,009 still-valid keep rewrites from the historical 1,122 accepted batch files, and freezes only the 4,464 missing v2 keepers into `keep-resume/candidates.jsonl`. Provider attempts and accepted keeper-resume batches remain local generated state and are resumable.

GemRouter must run with both:

```text
GEMROUTER_BACKEND_ORDER=gemini-api
GEMROUTER_NVIDIA_ENABLED=false
```

Before any adaptation traffic, `/health` must report `backendOrder=["gemini-api"]` and `fallbackEnabled=false`. If either condition fails, the pipeline must not send teacher requests.

Four-row Variant-D batches remain preferred. If a batch repeatedly fails, the same Variant-D request contract may be retried as deterministic halves and then singletons; accepted pieces are reassembled under the original keeper-resume batch identity. No alternative provider or prompt policy is introduced.

## Consequences

### Positive

- No new quota is spent adapting manually excluded rows.
- The completed 12,306-row model remains exactly reproducible from its historical curation.
- NVIDIA cannot silently enter the teacher path.
- Provider/schema failures are persisted and can be retried without losing accepted work.
- A problematic row can be isolated without discarding other rows in its original four-row batch.

### Negative or limiting

- The expansion corpus has a new curation identity and therefore must use a new future R-SFT run identity.
- Completion remains quota-limited and may require multiple daily quota windows.
- Final normalized-prompt deduplication can reduce the eventual corpus below the theoretical `7,683 + 8,473 + 630 = 16,786` rows.

## Validation

- Keeper-resume and historical adaptation unit tests must pass.
- Curation v2 must contain exactly 9,624 unique candidate IDs and the counts/hash above.
- Every accepted rewrite must pass the exact atomic 2,048-token serialization.
- GemRouter health must remain Gemini-only with fallback disabled during teacher traffic.
- Finalization must require an accepted rewrite for every v2 keeper and report any normalized-prompt collision exclusions explicitly.

## Links

- [`0103-simplify-overlength-superior-reasoning-with-batched-gemini.md`](0103-simplify-overlength-superior-reasoning-with-batched-gemini.md)
- [`0105-train-current-validated-superior-adaptation-checkpoint.md`](0105-train-current-validated-superior-adaptation-checkpoint.md)
- [`../runbooks/rsft_r0_atomic_production.md`](../runbooks/rsft_r0_atomic_production.md)
- [`../evidence/rsft_expansion_resume_2026-08-20.md`](../evidence/rsft_expansion_resume_2026-08-20.md)
