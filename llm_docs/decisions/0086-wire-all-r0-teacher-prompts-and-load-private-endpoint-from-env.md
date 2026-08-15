---
status: accepted
last_reviewed: 2026-08-15
---

# ADR 0086 — Wire all R0 teacher prompts and load the private LLM endpoint from environment configuration

## Decision

The reasoning-SFT module will own the teacher-generation prompt contracts for every accepted R0 reasoning family in `post_training/R-SFT/prompts.py`:

- `INF` — immediate inference;
- `DED` — deduction;
- `REL` — relational reasoning;
- `CSP` — constraint reasoning;
- `IND` — controlled induction;
- `ABD` — controlled abduction;
- `MAG` — numerical magnitude reasoning.

The prompt suite will preserve the generation structure established during the DED prompt experiments: batched generation (default 10 examples), one positive example per family, fully self-contained premises, explicit consistency and precision requirements, open-ended questions when natural, concise but complete reasoning, natural final answers rather than a forced yes/no label, semantic and structural diversity, and JSON-only `{problem, reasoning, answer}` output.

Each family prompt must define the intended reasoning structure tightly enough to keep examples inside that family and state nearby failure modes to avoid, while still allowing Gemini to choose the natural wording and reasoning depth. Internal skill codes and difficulty labels remain project-side metadata and are not emitted to Gemini as labels. Callers may inject plain-language structural requirements derived from internal difficulty metadata.

The private GemRouter/OpenAI-compatible endpoint will no longer be hardcoded in repository source. The transport resolves it from `LLM_ENDPOINT`, using the same configuration precedence as the bearer key: explicit constructor argument first, then process environment, then the repository-root `.env` file. A missing endpoint is a configuration error and fails before network activity. `.env.example` documents an empty `LLM_ENDPOINT=` field but does not contain the private URL.

`GEMR_API_KEY` remains the bearer-key variable and the default teacher model remains `gemini-3.7-flash` unless a later decision changes it.

## Rationale

The accepted R0 taxonomy needs stable, reviewable teacher contracts rather than ad-hoc prompt strings embedded in generation code. Sharing one generation structure across families makes dataset behavior easier to inspect while family-specific definitions reduce category drift such as DED batches becoming CSP or arithmetic tasks.

Keeping the endpoint outside source control avoids publishing private routing infrastructure while retaining a simple local/runtime configuration path.
