# Repository agent map

Use this file as a map, not as the project encyclopedia.

## Start every task here

1. Read [`llm_docs/current/status.md`](llm_docs/current/status.md).
2. Read [`llm_docs/current/roadmap.md`](llm_docs/current/roadmap.md).
3. Check [`llm_docs/decisions/README.md`](llm_docs/decisions/README.md) for accepted decisions.
4. Open only the relevant reference document or runbook from [`llm_docs/README.md`](llm_docs/README.md).

## Sources of truth

- `llm_docs/current/`: what is true now and what happens next.
- `llm_docs/decisions/`: one accepted, proposed, or superseded decision per ADR.
- `llm_docs/reference/`: durable technical contracts and system descriptions.
- `llm_docs/runbooks/`: commands and operational procedures.
- `llm_docs/research/`: investigations and external comparisons.
- `llm_docs/evidence/`: immutable completed-run evidence.
- `llm_docs/archive/`: superseded plans and historical scaffolding; never treat it as current guidance.
- `journals/`: informal study notes and personal reasoning, not authoritative project state.

When documents conflict, use this precedence order:

```text
current status -> accepted ADR -> current reference/runbook -> evidence -> archive/journals
```

## Working rules

- Keep the GDN-2 hybrid as the main architecture during the current 20M-model data-scaling stage.
- Do not authorize a larger model, new mixer, or architecture baseline without an explicit user decision.
- Keep dataset, tokenizer, model, checkpoint, and evaluation identities deterministic and fail closed on drift.
- Do not duplicate the qualitative prompt list; `trainer.post_pretraining_prompt_suite.PROMPT_CASES` is the source of truth.
- Prefer deleting dead code over retaining executable historical copies. Preserve completed results as evidence and rely on Git history for removed implementations.

## Documentation maintenance

- Update `current/status.md` when an operational fact changes.
- Update `current/roadmap.md` when priorities or gates change.
- Add or supersede an ADR when the user makes a durable decision.
- Update the relevant reference or runbook in the same commit as behavior changes.
- Do not rewrite evidence except to correct a factual transcription error; record the correction explicitly.
- Move superseded plans to `archive/` instead of leaving them mixed with active guidance.

## Verification commands

```bash
uv run --extra model python -m unittest discover -v
small-llm-eval-data verify --eval-dir /data/eval_core_v1
```

## Writing style

For project-facing prose, match `journals/`: informal, direct, slightly conversational, and concrete about trade-offs. Keep technical claims accurate and avoid corporate filler.
