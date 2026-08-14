---
status: accepted
last_reviewed: 2026-08-14
---

# ADR 0081 — Keep R-SFT level labels out of teacher prompts and do not count reasoning steps

## Decision

The R-SFT generation pipeline will keep internal level labels such as `L1`, `L2`, `L3`, and any later deeper bands out of the Gemini teacher prompt.

Those labels are project-side dataset metadata. Gemini does not share the project's level taxonomy, so teacher prompts must instead describe the actual structural conditions of the requested problem in plain language: for example one local inference, several dependent inferences, interacting constraints, branching/elimination, or deeper composition as appropriate.

The teacher will also **not** receive an exact requested number of reasoning steps. Once Gemini is asked to generate a problem satisfying the desired structural conditions, it may choose whatever complete reasoning path it considers appropriate.

The pipeline will not count or infer the number of reasoning steps Gemini used as a required dataset field. A valid trace is acceptable regardless of its step count, provided it is complete, relevant, and later passes whatever correctness/quality checks the project adopts.

## Gemini prompting contract

The project-side generator owns the internal selection of reasoning skill and difficulty. Before calling Gemini, it translates that internal selection into a plain-language problem-generation contract.

Gemini should receive only information that helps it generate the requested training example:

- the reasoning capability to exercise, described in ordinary language rather than an internal code such as `DED` or `CSP`;
- the structural conditions that make the problem appropriately easy or difficult, described directly rather than as `L1`/`L2`/`L3`;
- an instruction to create a self-contained problem that can be solved from the supplied information rather than outside knowledge;
- an instruction to provide a concise but complete reasoning path containing the inferences it considers necessary;
- an instruction to provide the final answer separately;
- a machine-readable output shape containing the generated problem, reasoning, and answer.

Gemini should **not** receive:

- internal difficulty labels;
- an exact or target number of reasoning steps;
- a hard reasoning-length ceiling;
- instructions to pad a solution to a requested length or to truncate a naturally complete solution.

The preferred conceptual output shape is:

```json
{
  "problem": "...",
  "reasoning": "...",
  "answer": "..."
}
```

The project keeps its own metadata next to the accepted example, for example `skill=DED` and `difficulty=L2`. Those metadata fields are not part of the Gemini-visible task and are later kept out of the student-visible R-SFT token stream as already decided.

## Example teacher prompt — short deduction

Internal project metadata might be `skill=DED`, `difficulty=L2`, but Gemini sees only something like:

```text
Generate one self-contained deductive reasoning problem.

The problem should require combining a short chain of dependent facts or rules rather than answering from one isolated fact. Keep the language natural and make the answer derivable entirely from the information in the problem.

Then provide a concise but complete reasoning path and the final answer. Use whatever reasoning depth is naturally required; do not add filler and do not omit necessary inferences.

Return JSON with exactly these fields: problem, reasoning, answer.
```

A suitable response could be:

```json
{
  "problem": "If a package reaches the sorting room, it is scanned. Every scanned package receives a tracking update. Mira's package reached the sorting room. Did it receive a tracking update?",
  "reasoning": "Mira's package reached the sorting room, so it was scanned. Every scanned package receives a tracking update, so her package received one.",
  "answer": "Yes."
}
```

## Example teacher prompt — constraint reasoning

Internal project metadata might be `skill=CSP`, `difficulty=L3`, but Gemini sees only something like:

```text
Generate one self-contained constraint-reasoning problem.

The problem should require keeping several restrictions active at once and eliminating incompatible possibilities before reaching a unique answer. Keep the search space small enough that the solution remains clear and compact. The answer must follow entirely from the stated constraints.

Then provide a concise but complete reasoning path and the final answer. Use whatever reasoning depth is naturally required; do not add filler and do not omit necessary inferences.

Return JSON with exactly these fields: problem, reasoning, answer.
```

A suitable response could be:

```json
{
  "problem": "Nora, Luca, and Sara each take one of seats 1, 2, and 3. Luca sits in seat 2. Nora cannot sit in seat 1. Which seat does Nora take?",
  "reasoning": "Luca already occupies seat 2. Nora cannot use seat 1, so the only remaining seat available to Nora is seat 3.",
  "answer": "Seat 3."
}
```

## Relationship to previous decisions

ADR 0077 still requires the project-side difficulty label for telemetry. ADR 0078 still defines the initial logic-first skill taxonomy and difficulty concept. ADR 0080's direction toward staged depth expansion remains valid, but any language suggesting that Gemini should receive a target step count is superseded by this ADR.

## Rationale

The level taxonomy is useful for controlling and measuring the dataset, not for teaching the teacher model our internal vocabulary. Prompting the actual structural requirements is more meaningful than passing an opaque label such as `L3`.

Likewise, prescribing or measuring an exact number of reasoning steps would impose an artificial decomposition on a teacher model that can decide its own natural solution depth. The project should control the challenge presented to Gemini, not micromanage the number of sentences or intermediate steps in Gemini's solution.
