# Research: Markdown project memory for humans and coding agents

_Last reviewed: 2026-08-06_

## Question

How should a code repository organize Markdown so current state, design rationale, operational knowledge, and historical evidence remain useful to both humans and coding agents?

## Findings

### Give agents a map, not a manual

OpenAI's 2026 harness-engineering report describes a failed “one large AGENTS.md” approach: it consumed scarce context, made every rule look equally important, rotted quickly, and was hard to verify. Their replacement was a short agent-facing table of contents pointing into a structured documentation system of record.

Applied here: `AGENTS.md` stays short and points first to current status, roadmap, decisions, then relevant reference or runbook material.

### Separate documentation by user need

Diátaxis distinguishes tutorials, how-to guides, reference, and explanation because each serves a different need and should be written differently.

Applied here: operational procedures live in `runbooks/`, durable definitions in `reference/`, and investigations in `research/`. Project-memory-specific categories add `current/`, `decisions/`, `evidence/`, and `archive/`.

### Record one significant decision per file

MADR recommends numbered Markdown Architectural Decision Records kept in a dedicated decisions directory. A record captures context, considered options, the chosen outcome, and consequences. Separate files make status, supersession, review, and tooling easier than one continuously growing ledger.

A 2026 empirical comparison found that concise Nygard-style records and structured MADR records were the strongest candidates among the evaluated templates; the useful trade-off is brevity versus more explicit structure. This project uses a small MADR-inspired template.

### Keep the README navigational

GitHub recommends a repository README that explains what the project does, why it is useful, how to get started, and where to find help. GitHub also supports relative links and automatically generated heading outlines.

Applied here: the root README is an onboarding page, while `llm_docs/README.md` is the documentation map. Neither is the detailed project encyclopedia.

### Treat freshness as a first-class property

Current facts change faster than architecture rationale or completed evidence. Mixing them forces every reader to determine freshness manually.

Applied here:

```text
current/   reviewed frequently and kept short
reference/ changed with implementation contracts
decisions/ append/supersede instead of silently rewriting
evidence/  immutable completed observations
archive/   explicitly non-current
```

### Make the structure mechanically testable

A directory convention only works when drift is visible. The repository therefore tests required indexes, ADR sections, top-level cleanliness, relative index links, and absence of removed legacy paths.

## Sources

- OpenAI, “Harness engineering: leveraging Codex in an agent-first world” (2026): https://openai.com/index/harness-engineering/
- OpenAI, “Introducing Codex” — repository guidance through `AGENTS.md` (2025): https://openai.com/index/introducing-codex/
- Diátaxis documentation framework: https://diataxis.fr/
- Markdown Architectural Decision Records: https://adr.github.io/madr/
- GitHub, “About the repository README file”: https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes
- Nogueira, Silva, and Conte, “One Size Fits All? An Empirical Comparison of ADR Templates...” (2026): https://arxiv.org/abs/2604.27333

## Project conclusion

Use Markdown as a small linked knowledge graph rather than a pile of chronological files: a short entry map, a concise current layer, append-only decisions and evidence, purpose-specific reference/runbooks/research, and an explicit archive.
