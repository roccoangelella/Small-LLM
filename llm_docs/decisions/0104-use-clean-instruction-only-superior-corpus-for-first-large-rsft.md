---
status: accepted
date: 2026-08-19
supersedes: 0102
---

# ADR 0104 — Use a clean instruction-only Superior corpus for the first large R-SFT

## Context and problem statement

ADR 0102 selected a 30% science / 70% instruction-following Superior Reasoning Stage-1 mixture for the first larger R-SFT corpus. A later context audit and sample review exposed two problems with that plan.

First, the approximately-100M model has a hard 2,048-token context and roughly half of the originally selected Superior examples do not fit the actual atomic R-SFT chat serialization. Silent truncation is not acceptable because it can remove the answer or break the reasoning/answer protocol.

Second, the project already decided in ADR 0082 that R0 should teach transferable logic and reasoning structure rather than exact arithmetic, algebra, calculus, or long numerical calculation. Pretraining was not math-free: the accepted CLIMB corpus contains explicit Mathematics clusters, and S0 also retained 15% CLIMB replay. That incidental/general exposure is different from making worked mathematical problem solving a primary supervised behavior during R-SFT. The Superior `science` slice contains many such computational examples, while the `instruction_following` slice is much closer to the desired capability but still contains some math- and code-primary tasks.

## Considered options

1. Keep the ADR 0102 30/70 science/instruction mixture and simplify all over-context examples with Gemini.
2. Use all instruction-following rows regardless of whether the primary task is computation or programming.
3. Use only instruction-following rows, conservatively remove math/computation/code-primary tasks, require exact 2,048-token fit, and accept a smaller but cleaner corpus.

## Decision outcome

Chosen option: **3**.

For the first large production R-SFT corpus:

- use only `Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b`, config `stage1`, split `train`, domain `instruction_following`;
- exclude all Stage-1 `science`, `math`, and `code` domain rows;
- within `instruction_following`, deterministically exclude examples whose primary task is explicit mathematical computation/proof or programming/code generation/debugging;
- do **not** reject an example merely because it contains numbers, percentages, counts, budgets, or formatting constraints; incidental numerical language remains allowed;
- parse the source `<think>...</think>` output strictly, reject malformed outputs, and deduplicate normalized prompts as before;
- require every kept Superior example to fit the exact atomic R-SFT serialization at the model's 2,048-token context with no truncation;
- keep **all** valid, unique, clean, context-fit instruction examples rather than forcing an arbitrary 25,000-example target;
- merge the resulting Superior records with the frozen 630-example Gemini logic corpus, preserving the existing seven logic/magnitude skill labels on the Gemini records;
- retain ADR 0103's batched Gemini simplification prompt as a future recovery tool for selected over-context examples, but do not mass-rewrite thousands of examples for this first production run; this keeps API usage bounded and the first large experiment easier to interpret;
- train with the already-accepted top-level mixture measured in loss-bearing target tokens: **90% reasoning corpus / 10% completed S0 instruction retention**;
- sample the 10% retention lane from the exact completed S0 tokenized instruction records, preserving S0's internal instruction-source proportions and excluding ClimbMix replay from that retention lane;
- partition heterogeneous reasoning data deterministically within each `skill × difficulty` group, using 1% validation and 1% test per group with a minimum of one record per held-out split;
- retain the 32,768-target optimizer-block production geometry and the atomic `<think>`, `</think>`, `<answer>` token contract.

The committed production reasoning artifact is:

```text
artifacts/rsft-superior-instruction-r0/reasoning.jsonl
```

Its manifest records the exact source counts, filter exclusions, context-fit count, Gemini merge count, SHA-256, and final row count.

## Kaggle execution contract

`python kaggle/launch_r_sft.py train --model 100M --tokens 2B` is the canonical first-large-R-SFT entry point.

When `--dataset-dir` is omitted, the launcher:

1. resolves the exact completed 100M/2B S0 bundle (explicit path, attached Kaggle dataset, or the frozen private Kaggle handle);
2. reads the committed Superior-instruction reasoning JSONL from the pinned worktree;
3. builds and verifies a native `atomic-production-v1` R-SFT bundle;
4. verifies that the bundle declares `reasoning_corpus_contract=heterogeneous-groups-v1` and the accepted Superior-instruction dataset identity;
5. launches the existing qualified 2xT4 DDP R-SFT trainer for exactly one pass.

A prebuilt production bundle may still be supplied with `--dataset-dir`, but it must satisfy the same corpus and tokenizer contracts.

## Rationale

The main experimental question for this run is whether a much larger and more diverse reasoning/instruction corpus improves general reasoning behavior over the overfit 630-example pilot. Introducing a substantial supervised mathematical curriculum at the same time would confound that question and consume scarce capacity on a capability the project explicitly intends to handle later with tool use.

Using only examples that already fit the real model context also avoids turning the experiment into a large synthetic-rewrite study. The 630 Gemini examples preserve direct supervision for deduction, induction, abduction, relational reasoning, constraints, inference, and magnitude awareness while the Superior instruction examples provide scale and broader instruction-following behavior.

## Consequences

### Positive

- The first large R-SFT remains aligned with ADR 0082's logic-first boundary.
- No training example is silently truncated or loses its final answer.
- The reasoning/S0 90/10 contract remains directly comparable with the previous R-SFT run.
- Kaggle can reproduce the complete production bundle from committed reasoning data plus the frozen S0 artifact without a separate manual dataset-preparation step.
- The corpus is materially larger than the 630-example pilot even after conservative filtering.

### Negative or limiting

- Some useful instruction examples are conservatively removed because the deterministic filter favors precision and experimental cleanliness over maximum corpus size.
- Over-context examples that could be salvaged by ADR 0103 are deferred rather than used in this run.
- This run does not test mathematical reasoning gains; math/tool-use remains a later stage.

## Validation

Before promotion or launch, require:

- strict JSON/schema validation of every committed reasoning row;
- exact atomic R-SFT tokenization of every reasoning row at context 2,048;
- no `SR_SCIENCE` rows and no source `math`/`code` domain rows in the production corpus;
- deterministic filter/selection unit tests;
- heterogeneous partition tests;
- exact 90/10 retention-target tests;
- full native bundle verification against the completed S0 bundle;
- Kaggle production dry-run showing automatic preparation from the committed Superior-instruction corpus.

Exact realized corpus and bundle counts are recorded in the generated dataset/bundle manifests rather than hard-coded into this decision.
