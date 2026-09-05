---
status: accepted
date: 2026-09-05
supersedes: 0151
---

# 0152 — Use fenced math blocks in journals

## Context and problem statement

ADR 0151 standardized display equations in `journals/` on `$$` delimiters. Although GitHub documents `$$...$$` as supported MathJax syntax, the user observed literal `$$` markup appearing as gibberish in the journal viewing surface.

GitHub also documents fenced `math` code blocks as a supported display-math syntax. This form avoids literal dollar delimiters around block equations and is less ambiguous for Markdown renderers.

## Considered options

- Keep `$$` display delimiters and accept renderer-dependent leakage.
- Convert display equations to plain text or Unicode approximations.
- Use fenced `math` blocks for display equations while retaining inline `$...$` math.

## Decision outcome

Chosen option: **use fenced `math` blocks for every display equation in repository journals**.

- Block equations use triple-backtick `math` fences.
- Inline expressions continue to use `$...$` because GitHub explicitly supports that form and it has not exhibited the reported block-rendering issue.
- No `$$` display delimiters should remain in `journals/journal*.md`.
- Mathematical content and historical prose remain unchanged apart from markup required for rendering.

## Consequences

### Positive

- Literal `$$` markers no longer appear around journal block equations.
- Display equations use GitHub's explicit math-fence syntax.
- The source is easier to distinguish from ordinary Markdown text.

### Negative or limiting

- Renderers that do not understand GitHub's `math` fence may show the equation as a code block rather than typeset math, but the LaTeX remains readable and the `$$` leakage is eliminated.

## Validation

- `journals/journal3.md`, `journals/journal4.md`, and `journals/journal5.md` contain fenced `math` blocks for all display equations.
- No journal that previously contained display equations retains `$$` delimiters.
- Inline LaTeX remains balanced and unchanged unless required for the display-block conversion.

## Links

- Supersedes ADR 0151.
- GitHub Docs: Writing mathematical expressions.
- `journals/`
