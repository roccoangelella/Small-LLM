---
status: accepted
date: 2026-08-20
supersedes: null
---

# 0107 — Add prompt-wrapper robustness to R-SFT qualification

## Context and problem statement

The production R-SFT R0 model was trained on assistant turns serialized through the Small-LLM chat template, with atomic `<think>`, `</think>`, and `<answer>` control tokens. The canonical qualification already measures reasoning acquisition under that trained assistant-turn context, but it does not tell us how strongly the learned reasoning behavior depends on the exact chat wrapper.

During the first full R-SFT qualification, historical raw continuation prompts naturally showed no reasoning markers. That is expected for continuation tasks, but it exposed a useful missing diagnostic: for actual reasoning questions, does the model initiate its reasoning protocol only under the trained chat template, or does the behavior transfer to common raw question formats as well?

## Decision outcome

Add a separate deterministic **prompt-wrapper robustness** diagnostic to the canonical R-SFT qualification.

For the same representative held-out reasoning problems, evaluate the accepted R-SFT checkpoint under three wrappers:

1. `chat`: the trained `small-llm-s0-v1` user/assistant generation template;
2. `question_answer`: raw `Question: {problem}\nAnswer:` text;
3. `plain`: raw `{problem}\n` text.

Use temperature `0`, top-p `1`, top-k `0`, one response per problem, and the canonical reasoning generation budget. Reuse identical cases and seeds across wrappers so the comparison isolates prompt serialization rather than sampling noise.

Keep this diagnostic deliberately compact: the full suite uses two frozen novel reasoning cases per R0 skill, for 14 cases total and 42 generations across the three wrappers. The fast suite uses one case per skill, for 7 cases and 21 generations.

Report these axes separately for every wrapper:

- final-answer accuracy allowing any output format;
- strict answer accuracy requiring a well-formed reasoning protocol;
- reasoning-start rate;
- full atomic protocol health, including well-formedness, non-empty reasoning/answer, EOS termination, and reasoning/answer lengths;
- deltas for the raw wrappers relative to the trained chat baseline.

Do not merge prompt-wrapper robustness into the main novel-reasoning score or into a single master score. A raw-wrapper drop measures template dependence; it does not erase reasoning competence demonstrated under the trained assistant context.

## Consequences

### Positive

- Distinguishes learned reasoning behavior from exact-template imitation.
- Makes prompt-format dependence directly measurable instead of inferred from unrelated continuation prompts.
- Uses the same mechanically scored reasoning tasks and seeds, so wrapper deltas are easy to interpret.
- Adds only 42 deterministic generations to the full qualification instead of another full 35-case sampled matrix.

### Negative or limiting

- The probe tests only three wrappers and is not a complete prompt-robustness benchmark.
- Raw-wrapper correctness can still be higher than strict protocol correctness when the model answers directly without entering the reasoning protocol.
- The additional generations increase full-suite runtime modestly.

## Implementation

- `post_training/rsft_prompt_wrapper_eval.py`
- `kaggle/rsft_eval_runtime.py`
- `tests/test_rsft_prompt_wrapper_eval.py`
- `llm_docs/runbooks/rsft_r0_qualification.md`

The wrapper probe runs after the existing canonical qualification and appends `rsft.scorecard.prompt_wrapper_robustness` to the same versioned report, then recomputes the report hash.
