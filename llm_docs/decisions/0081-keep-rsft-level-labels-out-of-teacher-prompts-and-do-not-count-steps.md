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

## Relationship to previous decisions

ADR 0077 still requires the project-side difficulty label for telemetry. ADR 0078 still defines the initial logic-first skill taxonomy and difficulty concept. ADR 0080's direction toward staged depth expansion remains valid, but any language suggesting that Gemini should receive a target step count is superseded by this ADR.

## Rationale

The level taxonomy is useful for controlling and measuring the dataset, not for teaching the teacher model our internal vocabulary. Prompting the actual structural requirements is more meaningful than passing an opaque label such as `L3`.

Likewise, prescribing or measuring an exact number of reasoning steps would impose an artificial decomposition on a teacher model that can decide its own natural solution depth. The project should control the challenge presented to Gemini, not micromanage the number of sentences or intermediate steps in Gemini's solution.