---
status: accepted
date: 2026-08-22
---

# 0118 — Evaluate the current three-epoch R-SFT model by default

## Context

ADR 0116 provisionally promoted `100m-2b-rsft-r0-16716-e3-001` as the current/default R-SFT R0 model, but the canonical Kaggle R-SFT evaluation launcher still resolved the older historical `100m-2b-rsft-r0-12306-001` checkpoint.

That mismatch made the standard qualification command evaluate a superseded model even though chat and current project status already selected the three-epoch run.

## Decision

The canonical command

```bash
python kaggle/launch_r_sft.py eval --model 100M --tokens 2B --suite full
```

must evaluate `100m-2b-rsft-r0-16716-e3-001` against the completed S0 parent `100m-2b-sft-s0-001`.

The historical `100m-2b-rsft-r0-12306-001` run remains preserved as an explicit comparison artifact and is not renamed or reclassified as the current default.

The launcher records the current evaluation target independently from the historical accepted-run constant, and a regression test must fail if the canonical eval target drifts away from the current three-epoch model.

## Consequences

- `kaggle/launch_r_sft.py eval --model 100M --tokens 2B --suite full` now qualifies the model actually selected by the current R-SFT profile.
- The qualification suite itself is unchanged; only its default R-SFT checkpoint target changes.
- Historical one-epoch R-SFT evidence remains available for explicit comparison.

## Links

- [`../runbooks/rsft_r0_qualification.md`](../runbooks/rsft_r0_qualification.md)
- [`../current/status.md`](../current/status.md)
- [`../../kaggle/rsft_cli.py`](../../kaggle/rsft_cli.py)
