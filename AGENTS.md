# Repository agent map

Use this file as a map, not as the project encyclopedia.

## Start every task here

1. Read [`llm_docs/current/status.md`](llm_docs/current/status.md).
2. Read [`llm_docs/current/roadmap.md`](llm_docs/current/roadmap.md).
3. Check [`llm_docs/decisions/README.md`](llm_docs/decisions/README.md) for accepted decisions.
4. Open only the relevant reference, runbook, research note, evidence record, or active plan from [`llm_docs/README.md`](llm_docs/README.md).

## Sources of truth

- `llm_docs/current/`: only `status.md` and `roadmap.md`; high-freshness working memory.
- `llm_docs/decisions/`: one accepted, proposed, superseded, or rejected durable choice per ADR.
- `llm_docs/reference/`: current technical contracts and system descriptions.
- `llm_docs/runbooks/`: executable/reproducible commands and operational procedures.
- `llm_docs/research/`: investigations and external comparisons; not authorization.
- `llm_docs/plans/`: resumable execution plans for complex work; completed plans are history.
- `llm_docs/evidence/`: immutable measured/observed results.
- `llm_docs/archive/`: superseded plans, completed handoffs, and historical design records; never current guidance.
- `journals/`: informal study notes and personal reasoning, not authoritative project state.

When documents conflict, use this precedence order:

```text
current status -> accepted ADR -> current reference/runbook -> evidence -> research -> archive/journals
```

## Working rules

- Keep the GDN-2 hybrid as the main architecture during the current 20M-model data-scaling stage.
- Do not authorize a larger model, new mixer, or architecture baseline without an explicit user decision.
- Keep dataset, tokenizer, model, checkpoint, and evaluation identities deterministic and fail closed on drift.
- Do not duplicate the qualitative prompt list; `trainer.post_pretraining_prompt_suite.PROMPT_CASES` is the source of truth.
- Prefer deleting dead executable code and redundant documentation snapshots over retaining several competing copies. Preserve measurements as evidence and historical decisions/handoffs in archive when they remain useful.

## Documentation maintenance

- Keep `llm_docs/` root to `README.md` only.
- Keep `llm_docs/current/` to `status.md` and `roadmap.md` only.
- Update `current/status.md` when an operational fact changes.
- Update `current/roadmap.md` when priorities or gates change.
- Add or supersede an ADR when the user makes a durable decision.
- Update the relevant reference or runbook in the same change as behavior when practical.
- Put unresolved analysis in research, not in current or an accepted ADR.
- Put measured incidents/results in evidence; do not rewrite them to fit later conclusions.
- Use an active execution plan for genuinely complex/resumable work, then move it to completed.
- Move superseded plans and completed investigation handoffs to archive.
- After a major experiment or investigation changes lifecycle, garden the documentation and run the memory-structure tests.

## Verification commands

```bash
uv run --extra model python -m unittest discover -v
small-llm-eval-data verify --eval-dir /data/eval_core_v1
```

## Writing style

For project-facing prose, match `journals/`: informal, direct, slightly conversational, and concrete about trade-offs. Keep technical claims accurate and avoid corporate filler.
