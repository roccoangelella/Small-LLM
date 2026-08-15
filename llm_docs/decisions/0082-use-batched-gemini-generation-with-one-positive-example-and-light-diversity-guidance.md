---
status: accepted
last_reviewed: 2026-08-14
---

# ADR 0082 — Use batched Gemini generation with one positive example and light diversity guidance

## Decision

The Gemini R-SFT teacher prompt should include at least one positive example showing the intended problem / reasoning / final-answer shape. The example exists to demonstrate the style and level of abstraction we want rather than to define a rigid template that Gemini must copy.

Teacher prompts should also contain a short diversity instruction making clear that a generated batch must contain meaningfully different problems, not superficial rewrites that merely swap names, nouns, verbs, or other surface details while preserving the same underlying instance.

Keep this guidance lightweight. Gemini is expected to handle semantic variation without a long list of prohibitions or manually enumerated variation axes.

Generation should be batched rather than one API request per example. The default initial target is approximately **10 independent examples per Gemini call**. The exact batch size remains an implementation parameter and may be adjusted later based on response quality, context size, latency, malformed-output rate, or provider limits.

## Prompt boundary

Project-side metadata such as skill codes and difficulty labels remain hidden from Gemini. The project translates them into plain-language structural requirements before building the teacher prompt, as decided in ADR 0081.

Gemini receives:

- the reasoning/problem-generation capability described in ordinary language;
- the structural conditions that generated problems should satisfy;
- a request for self-contained problems whose answers follow from the supplied information;
- a request for concise but complete reasoning with no hard step count or trace-length ceiling;
- one representative positive example of the desired output style;
- a brief instruction that batch members must be genuinely different rather than trivial surface substitutions;
- a structured output request containing a list of approximately 10 `{problem, reasoning, answer}` records.

Gemini does not receive internal `L1` / `L2` / `L3` labels or a requested number of reasoning steps, and the pipeline does not need to count the number of steps in the returned reasoning.

## Example teacher prompt shape

For a short deductive-generation cell, the teacher prompt can look conceptually like:

```text
Generate 10 self-contained deductive reasoning problems.

Each problem should require combining a short chain of dependent facts or rules rather than answering from one isolated fact. Keep the language natural and make every answer derivable entirely from the information in the problem.

For each problem, provide a concise but complete reasoning path and the final answer. Use whatever reasoning depth is naturally required. Do not add filler and do not omit necessary inferences.

Here is one example of the style we mean:
Problem: If a package reaches the sorting room, it is scanned. Every scanned package receives a tracking update. Mira's package reached the sorting room. Did it receive a tracking update?
Reasoning: Mira's package reached the sorting room, so it was scanned. Every scanned package receives a tracking update, so her package received one.
Answer: Yes.

Make the 10 generated problems genuinely different. Do not simply reuse the same logical instance while swapping names, subjects, or verbs.

Return a JSON array. Every item must contain exactly: problem, reasoning, answer.
```

The exact wording will be implemented later with the per-skill and per-difficulty structural contracts; this ADR freezes the prompting architecture, not every final prompt sentence.

## Rationale

A single positive example communicates the desired abstraction, concision, and output structure more efficiently than increasingly prescriptive prose. Batching reduces API overhead and gives Gemini enough room to produce useful semantic variation in one generation, while the lightweight diversity instruction guards against a batch collapsing into ten cosmetic variants of one template.
