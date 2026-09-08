# Repository repair verification — 2026-09-08

The provider compaction parent (`4d31a5c`) and the compaction working tree both reported 555 unittest tests, 119 failures, 37 errors, and one skip. Matching baseline failures were not evidence of a healthy repository.

The repair fixes Kaggle source-root and executable-wrapper import behavior, validates saved R-SFT generation configuration before batch reuse, installs the locked test dependency group, and updates stale test paths and fixtures. Historical ADR bodies remain unchanged; explicit filename/SHA-256 exemptions preserve their existing formats while all other ADRs require the standard shape.

Final local offline verification on the repair working tree:

- `uv run --extra model python -m unittest discover -q`: 591 tests, OK, one skip.
- `uv run --extra model pytest -q -rs`: 702 passed, one skipped.
- `uv lock --check`: passed.
- `git diff --check`: passed.

The skip is `tests/test_live_remote_smoke.py`: real remote-storage checks require explicit `SMALL_LLM_LIVE_REMOTE_SMOKE=1`. No live cloud/GPU qualification was performed. `/data/eval_core_v1` is absent on this host, so full external corpus verification remains outstanding. These results establish offline test health, not production launch authorization.
