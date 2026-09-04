---
status: accepted
date: 2026-09-04
owners: [Small-LLM]
supersedes_for_base_prompt_scoring:
  - 0140-wire-evaluation-v2-and-retire-fixed-length-qualitative-protocol
  - 0149-replace-recycled-base-prompt-v2-templates-with-120-unique-prompts
---

# ADR 0150: score Base Prompt v2 with a GemRouter semantic judge

## Decision

Base Prompt v2 objective generations must no longer be scored inside the GPU
evaluator by substring or regular-expression checks.

The checkpoint evaluator now emits durable raw evidence for every Base Prompt
case:

- case ID and family;
- prompt;
- benchmark `reference_answer` for the 100 objective cases;
- model continuation and generated token IDs;
- decoding seed/budget metadata;
- `judge_status=pending` for objective cases;
- no local `passed`/`checks` verdict.

The 20 qualitative continuations remain readable, unscored evidence.

A separate post-processing command, `python -m trainer.base_prompt_judge`, reads
the completed evaluation JSON and scores the objective cases through the same
private GemRouter endpoint used by the R-SFT data pipeline. The live judge uses:

- `GEMR_API_KEY` and `LLM_ENDPOINT`, matching the R-SFT transport contract;
- requested model `gemini-3.7-flash` by default;
- `temperature=0`;
- a fail-closed GemRouter health gate requiring
  `backendOrder=['gemini-api']` and `fallbackEnabled=false`;
- strict JSON judgments;
- batched requests (default 20 cases, hard cap 25);
- retry on provider or malformed-JSON failure, then fail closed.

Both greedy and project-standard sampled Base Prompt outputs are judged. The
judge accepts semantically equivalent wording and harmless explanation, but it
must evaluate the answer attributable to the original prompt rather than
literal text overlap. Contradictions override incidental matches. In
particular, a reference answer such as `0` appearing as characters inside an
incompatible answer such as `-40 C` must not receive credit.

The judge returns binary `correct` / `incorrect` verdicts, local numeric scores
of 1 / 0, concise reasons, aggregate accuracy, and per-family accuracy. It does
not judge the 20 qualitative continuation cases.

## Artifact boundary

Raw generation and AI judgment are intentionally separate artifacts.

This keeps the expensive GPU qualification reproducible and durable even when
the external judge is unavailable, and permits the same raw outputs to be
re-judged later without rerunning model inference or recompiling CUDA/Triton
kernels.

The judgment artifact records:

- source JSON SHA-256;
- judge model requested;
- judge prompt ID and SHA-256;
- judge temperature;
- batch/retry settings;
- sanitized GemRouter health-gate state;
- provider model returned per accepted batch.

## Comparability

Base Prompt semantic accuracy is comparable across model runs only when the
same corrected unique-120 prompt set, judge prompt ID/hash, and judge model are
used. If the judge contract changes, the raw generations may be re-judged under
the new contract, but the resulting score belongs to a new judgment series.

Historical Base Prompt v2 scores produced by substring/regex checks are not
canonical semantic-judge scores. `eval_core_v1` and L20 conditional-likelihood
metrics are unaffected by this decision.

## SFT qualification

The same Base Prompt raw generator is used in SFT qualification, so its Base
Prompt sections also stop producing local string-match accuracy. The
`trainer.base_prompt_judge` postprocessor accepts both pretrained evaluation-v2
bundles and parent-versus-SFT qualification-v2 bundles and emits separate parent
and SFT semantic judgments.

SFT Behavior v2 remains governed by its own task-specific behavior scoring; this
ADR changes only Base Prompt v2 scoring.
