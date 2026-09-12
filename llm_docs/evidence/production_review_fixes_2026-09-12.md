# Production review fixes — 2026-09-12

Local verification on `edo/production-run`. All six requested findings were
confirmed against code and fixed. No GPU run or remote action was performed.

| Item | Change | Requested suite passed | Subtests passed |
|---|---|---:|---:|
| general 1 | Quarantine every invalid step directory; preserve complete checkpoints and allow resaving the ID | 45 | 5 |
| general 2 | Verify explicit resume before computing remaining steps, including at target | 46 | 9 |
| general 3 | Atomically persist the promoted best ID/metric and restore retention protection after restart; retention-off ignores the record | 47 | 9 |
| general 4 | Document asynchronous prune/volume-commit limitation in ADR 0179; reject production keep-last 1 | 48 | 14 |
| optim 1 | Set compiled backward autocast off when supported, matching eager backward outside autocast; document in ADR 0180 | 49 | 14 |
| optim 2/3 | Narrow CPU bias identity claim, record H100 rounding difference, verify backward at two capacities with executed compiled-frame counter | 50 | 14 |

Command, run after each fix using the repository virtualenv:

```bash
.venv/bin/python -m pytest -q tests/test_production_checkpoint_sequence.py tests/test_moe_production_wiring.py tests/test_moe_production_launch.py tests/test_moe_compile_lane.py
```

Baseline: 44 tests and 5 subtests passed. Final run: 50 tests and 14 subtests
passed in 64.68 seconds, with 14 existing torch deprecation warnings. The default
system Python has no pytest. Local torch is 2.13.0+cu130; these tests used CPU
`aot_eager`, not the production GPU/PyTorch 2.10 stack.

The quarantine and best-record regressions failed before their fixes. The new
backward test also passed independently. The H100 first-update bias absmax values
were checked against the existing qualification JSON artifacts: 0.03326237201690674
eager and 0.03327357769012451 compiled. No new GPU equivalence claim is made.

## Pending delivery

No commit was created. Git could not create its worktree index lock because the
actual metadata is outside the writable sandbox, under
`/home/edo/Documents/2_Code/study/Small-LLM-rocco/.git/worktrees/Small-LLM-moe-ready/`.
Approval escalation was unavailable. The project hub outside this checkout also
could not be updated. Changes remain in the working tree.

Next: from a session with writable Git metadata, stage only each fix's hunks and
create six separate commits in the order above, with title-only imperative messages:

1. Quarantine invalid checkpoints before auto-resume
2. Validate explicit production resume checkpoints
3. Preserve best checkpoint retention across restarts
4. Guard production retention and document commit lag
5. Match compiled backward autocast to eager training
6. Qualify compile bias claims and test dynamic backward

Do not stage the concurrent agent's `kaggle/launch.py` or ADRs numbered 0011–0178.
Those files were not edited by this work. Do not push. The remaining commit task
is a filesystem permission blocker, not a request for new product decisions.
