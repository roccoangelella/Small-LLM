# Repository agent map

Use this file as a map, not as the project encyclopedia.

## Context discipline & lazy retrieval

- **Do NOT eagerly read documentation files at session startup.** Pull information strictly on demand when relevant to the user's specific request.
- Consult [`llm_docs/current/status.md`](llm_docs/current/status.md) ONLY when the task requires active checkpoint, model geometry, or running training state.
- Consult [`llm_docs/current/roadmap.md`](llm_docs/current/roadmap.md) ONLY when prioritizing next milestones, verifying gates, or resolving open decisions.
- Consult [`llm_docs/decisions/README.md`](llm_docs/decisions/README.md) ONLY when creating, evaluating, or superseding durable architectural choices.
- Open specific reference, runbook, research, or evidence records from [`llm_docs/README.md`](llm_docs/README.md) only on demand.
- Prefer targeted `grep_search` and `find_by_name` over bulk reading of large documentation files.
- For heavy multi-file exploration or log analysis, delegate to a subagent to keep the main conversation context lean.

## Sources of truth & memory hierarchy

```text
current status -> accepted ADR -> current reference/runbook -> evidence -> research -> archive/journals
```

- `llm_docs/current/`: only `status.md` and `roadmap.md`; high-density executive working memory.
- `llm_docs/decisions/`: durable choices (one accepted, proposed, superseded, or rejected ADR per file).
- `llm_docs/reference/`: current authoritative technical contracts and system descriptions.
- `llm_docs/runbooks/`: executable/reproducible commands and operational procedures.
- `llm_docs/research/`: investigations and external comparisons; not authorization.
- `llm_docs/plans/`: resumable execution plans for complex work (`plans/active/`, `plans/completed/`).
- `llm_docs/evidence/`: immutable measured/observed results and incident records.
- `llm_docs/archive/`: superseded plans, completed handoffs, and historical design records.
- `journals/`: informal study notes and personal reasoning, not authoritative project state.

## Working rules

- Keep the GDN-2 hybrid as the main architecture during the current 20M-model data-scaling stage.
- Do not authorize a larger model, new mixer, or architecture baseline without an explicit user decision.
- Keep dataset, tokenizer, model, checkpoint, and evaluation identities deterministic and fail closed on drift.
- Do not duplicate the qualitative prompt list; `trainer.post_pretraining_prompt_suite.PROMPT_CASES` is the source of truth.
- Prefer deleting dead executable code and redundant documentation snapshots over retaining competing copies. Preserve measurements as evidence.

## Documentation maintenance

- Keep `llm_docs/` root to `README.md` only.
- Keep `llm_docs/current/` to `status.md` and `roadmap.md` only.
- Update `current/status.md` when an operational fact changes.
- Update `current/roadmap.md` when priorities or gates change.
- Add or supersede an ADR when making a durable decision.
- Update relevant reference or runbook in the same change as behavior when practical.
- Record measured incidents/results in evidence; do not rewrite history.

## Verification commands

```bash
uv run --extra model python -m unittest discover -v
small-llm-eval-data verify --eval-dir /data/eval_core_v1
```

## Writing style

For project-facing prose, match `journals/`: informal, direct, slightly conversational, and concrete about trade-offs. Keep technical claims accurate and avoid corporate filler.
