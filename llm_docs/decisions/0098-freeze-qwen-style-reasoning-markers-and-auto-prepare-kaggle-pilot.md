---
status: accepted
date: 2026-08-18
---

# ADR 0098 — Freeze Qwen-style reasoning markers and auto-prepare the Kaggle pilot

## Context

The 630-example R0 reasoning corpus is now frozen and committed at `artifacts/rsft-r0-pilot-630/generation/reasoning.jsonl`. The three promoted reasoning-control IDs were already fixed at 50257, 50258, and 50259, but their text spellings had intentionally remained open until the delimiter ablation. The first Kaggle R-SFT launcher also still required a prebuilt tokenized bundle, explicit token spec, and explicit run ID.

The project owner delegated the exact token spelling to the implementation, preferring a convention compatible with contemporary reasoning LLMs, and requested that the Kaggle R-SFT path be ready to train directly from the now-committed pilot corpus.

## Decision

Freeze the atomic token strings as:

```text
50257  <think>      reasoning start
50258  </think>     reasoning end
50259  <answer>     final-answer start
```

`<think>` / `</think>` follow the visible reasoning-boundary convention used by Qwen3-family reasoning chat templates, including the sub-1B Qwen3 line. The third `<answer>` marker is Small-LLM-specific because the previously accepted architecture already reserves an explicit answer-start control ID rather than relying only on the reasoning closing tag.

For the first 630-example delimiter pilot, make `kaggle/launch_r_sft.py train` self-preparing:

- `--delimiter-format atomic|textual` remains mandatory so the experimental arm is always explicit.
- `--dataset-dir`, `--token-spec`, `--run-id`, and `--s0-bundle` become optional overrides.
- With no prebuilt dataset, the pinned Kaggle worktree reads the committed `reasoning.jsonl` and canonical `post_training/R-SFT/reasoning-tokens.json`.
- The completed S0 dataset bundle for the 10% instruction-retention lane is resolved from an already attached Kaggle input when possible; otherwise the launcher downloads the private canonical S0 dataset `roccoangelella/small-llm-100m-2b-sft-s0-001`. An explicit path or `SMALL_LLM_S0_KAGGLE_DATASET_HANDLE` remains an override.
- The same committed reasoning partition and exact S0 retention records are used to build both atomic and textual native bundles before selecting the requested arm.
- Both arms remain one-pass experiments and continue to use the qualified 2xTesla-T4 DDP SFT engine, S0 parent checkpoint, model-row promotion, checkpointing, W&B and remote publication contracts already accepted for R-SFT.
- Canonical run IDs are `100m-2b-rsft-r0-atomic-pilot-001` and `100m-2b-rsft-r0-textual-pilot-001`.

For this deliberately small delimiter pilot, Kaggle automatic bundle preparation uses **2,048 loss-bearing target tokens per optimizer block** instead of the generic R-SFT builder default of 32,768. This is an experiment-local geometry change: with only 588 reasoning training examples, 32,768-target blocks would collapse the run into too few optimizer updates for a useful delimiter-learning comparison. The atomic and textual arms use the same 2,048-target geometry, so the ablation remains matched. The generic/larger R-SFT bundle builder keeps its 32,768 default.

## Consequences

The canonical Kaggle commands are now intended to be:

```bash
python kaggle/launch_r_sft.py train --model 100M --tokens 2B --delimiter-format atomic
python kaggle/launch_r_sft.py train --model 100M --tokens 2B --delimiter-format textual
```

The GPU trainer still never consumes raw JSONL directly. The launcher materializes and verifies native immutable SFT bundles first, then trains from the selected bundle. Repeated launches reuse a previously verified matched bundle root rather than rebuilding it.

This decision freezes only the pilot token spellings and Kaggle execution contract. The final production R-SFT corpus size/token budget remains open pending pilot statistics and behavior.
