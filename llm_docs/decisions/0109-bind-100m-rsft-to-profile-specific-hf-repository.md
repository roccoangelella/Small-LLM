---
status: accepted
date: 2026-08-21
supersedes: null
---

# 0109 — Bind 100M R-SFT to a profile-specific Hugging Face repository

## Context and problem statement

A committed Kaggle run for the expanded 16,716-row R-SFT corpus successfully built the bundle and entered 2xT4 DDP, then failed before training because the launcher inherited `SMALL_LLM_HF_REPO_ID=roccoangelella/small-llm-20m-qualification`. It therefore tried to resolve parent run `100m-2b-sft-s0-001` from the 20M repository and raised `Hugging Face model repository contains no artifact for run '100m-2b-sft-s0-001'`.

The R-SFT launcher is fixed to the 100M/2B profile, so a cross-profile generic repository fallback is unsafe.

## Decision outcome

For 100M/2B R-SFT model artifacts, resolve the S0 parent repository as:

1. explicit `--parent-repo-id`;
2. `SMALL_LLM_100M_HF_REPO_ID`;
3. `roccoangelella/small-llm-100m-qualification`.

Resolve the R-SFT checkpoint repository as:

1. explicit `--checkpoint-repo-id`;
2. `SMALL_LLM_RSFT_HF_REPO_ID`;
3. `SMALL_LLM_100M_HF_REPO_ID`;
4. `roccoangelella/small-llm-100m-qualification`.

Do not consult generic `SMALL_LLM_HF_REPO_ID` or legacy `SMALL_LLM_SFT_HF_REPO_ID` in this fixed-profile R-SFT path. Train and eval share the same resolver.

## Consequences

- Stale 20M Kaggle secrets can no longer redirect 100M R-SFT parent or checkpoint lookup.
- The canonical R-SFT command remains self-contained without requiring repository flags.
- Alternate owners/repositories remain supported through explicit flags or the profile-specific environment variables.
- Generic repository variables remain valid for other launchers; this restriction is local to the fixed 100M/2B R-SFT profile.

## Validation

A regression test injects both generic and legacy SFT variables pointing at `small-llm-20m-qualification` and requires the production dry run to emit only `small-llm-100m-qualification`. The target repository was independently queried and contains `run/100m-2b-sft-s0-001/latest.json`.

## Links

- [`../evidence/rsft_kaggle_wrong_parent_repo_fix_2026-08-21.md`](../evidence/rsft_kaggle_wrong_parent_repo_fix_2026-08-21.md)
- [`../runbooks/rsft_r0_atomic_production.md`](../runbooks/rsft_r0_atomic_production.md)
