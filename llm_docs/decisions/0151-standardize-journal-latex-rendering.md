---
status: accepted
date: 2026-09-05
supersedes: null
---

# 0151 — Standardize journal LaTeX rendering

## Context and problem statement

The repository-level `journals/` are informal study notes, but their mathematical notation should still render reliably in GitHub Markdown. Existing notes mixed valid inline LaTeX with one-line display delimiters and a few raw equation-like expressions such as multiplication, equality, and magnitude comparisons.

This is a presentation/maintenance decision only. It does not change the technical claims or authority of the journals.

## Considered options

- Leave the historical Markdown unchanged even when mathematical notation is inconsistent.
- Convert all mathematical notation to plain text or code formatting.
- Keep LaTeX and normalize it to GitHub-supported inline and display math conventions.

## Decision outcome

Chosen option: **keep LaTeX and normalize it to GitHub-supported Markdown math conventions**.

- Inline expressions use `$...$`.
- Display equations use multiline `$$` delimiters with the opening and closing delimiters on their own lines.
- Mathematical operators written as prose when they are functioning as equations or relations should be moved into LaTeX, for example `\times`, equality, and `<` / `>` comparisons.
- Ordinary configuration strings, commands, literal special tokens, and non-mathematical prose should not be converted to LaTeX merely because they contain symbols.
- Journal edits for this purpose must preserve the historical technical meaning and informal tone.

## Consequences

### Positive

- Mathematical passages render consistently in GitHub.
- Display equations are easier to distinguish from surrounding prose and less sensitive to Markdown parsing context.
- Raw mathematical comparisons no longer risk being confused with Markdown/HTML syntax.

### Negative or limiting

- Some historical journal lines receive formatting-only diffs.
- This standard does not attempt to correct historical mathematical or scientific claims; it only normalizes their markup.

## Validation

The journal math audit must show:

- balanced dollar delimiters in every `journals/journal*.md` file;
- no one-line `$$...$$` display equations;
- no raw Unicode multiplication sign in normalized mathematical passages;
- `git diff --check` passes.

The normalization commit `b82821f` passed these checks before push to `main`.

## Links

- `journals/`
- `AGENTS.md`
