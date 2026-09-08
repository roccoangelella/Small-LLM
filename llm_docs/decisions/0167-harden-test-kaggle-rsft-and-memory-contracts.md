---
status: accepted
date: 2026-09-08
supersedes: null
owners: [Small-LLM]
---

# ADR 0167: harden test, Kaggle, R-SFT, and project-memory contracts

## Context and problem statement

The current repository repair spans four related contract failures.

- The default uv environment installed the runtime but not `pytest`, while
  `unittest` discovery does not collect the repository's plain-function tests.
  The committed lock and the README therefore did not describe one complete
  local test path.
- After the Kaggle workspace moved implementation files below `kaggle/src/`,
  some source modules still computed the old repository root or targeted files
  at the old depth. The root compatibility wrappers were executable, but
  importing them ran the delegated script instead of exposing an import-safe
  `main`. Flat module names such as `runtime` could also leak between Kaggle
  and Modal test imports.
- Resumable R-SFT generation wrote a hashed generation manifest, but a missing
  final JSONL could leave saved batch state available for a changed generation
  plan. Reusing those batches with different `examples_per_cell`, batch size,
  seed, or call count would mix identities.
- Project-memory tests used a historical numeric cutoff for ADR shape. Earlier
  ADRs intentionally retain several legacy formats, so converting them merely
  to satisfy a stricter test would rewrite history and erase the fact that the
  formats changed over time.

## Considered options

- Keep `pytest` outside the default environment and rely on an ambient install
  or on `unittest` alone.
- Rewrite the Kaggle tree into a new package and change pinned worktrees and
  remote import surfaces, or make the existing root wrappers and `src` paths
  correct without changing the scientific launch contracts.
- Reuse saved R-SFT batches whenever a final output is absent, or verify the
  saved plan's self-hash and identity fields before allowing a resume.
- Rewrite all legacy ADRs into the newest format, keep a numeric exemption, or
  record the exact historical nonconforming files and bytes in a hashed
  baseline while requiring the standard shape for new ADRs.

## Decision outcome

Chosen option: **make the local test environment and repository boundaries
fail closed, while preserving existing scientific and historical identities.**

The repair has four parts:

1. The `test` dependency group contains `pytest>=8,<9`, and uv's default groups
   are `runtime` and `test`. The committed `uv.lock` is updated with that
   contract. The concise README test path is:

   ```bash
   uv sync --locked
   uv run --extra model python -m unittest discover -v
   uv run --extra model pytest -q
   ```

   `uv sync --locked` both installs the locked default runtime/test groups and
   rejects project/lock drift. The explicit `model` extra supplies the model
   dependencies for these invocations. `pytest` is the complementary collector
   for plain function tests omitted by `unittest` discovery.

2. The stable Kaggle root wrappers remain the operator surface. They delegate
   with a non-`__main__` `runpy` name, expose the implementation symbols for
   imports, and call `main()` only when executed as scripts. Kaggle source
   modules now resolve the actual repository root, `kaggle/src`, and
   `kaggle/env`, and launcher commands target sibling source files at their
   real locations. Test imports use `kaggle/src` and isolate flat provider
   module names where necessary. These are compatibility and path fixes: they
   preserve pinned worktree selection, checkpoint/repository bindings, profile
   identities, and training/evaluation science rather than introducing a new
   Kaggle implementation.

3. Before reusing incomplete R-SFT generation state,
   `generate_resumable()` validates `generation-manifest.json` as a JSON object,
   checks its `manifest_sha256` self-hash, and compares the schema,
   `examples_per_cell`, `batch_size`, `total_calls`, and `seed` with the
   current plan. Invalid or changed state raises instead of silently mixing
   batches from different plans. Completed output continues through its
   existing completed-generation validation.

4. `tests/fixtures/project_memory_adr_legacy_baseline.json` is the explicit
   SHA-256 baseline for every historical nonconforming ADR. The project-memory
   test requires the set of nonconforming files to equal that baseline and
   requires every baseline file's bytes to retain its recorded digest. Every
   ADR outside that baseline, including this one, must have YAML metadata and
   the standard full headings. Historical ADRs are not rewritten.

## Consequences

### Positive

- A clean locked environment includes both the runtime and the test collector
  needed by the repository's two offline test commands.
- Root Kaggle commands remain usable from pinned worktrees while imports and
  nested path calculations agree with the `kaggle/src` and `kaggle/env` layout.
- R-SFT resumability is fail-closed against saved-plan drift rather than merely
  fail-closed after a final artifact has already been assembled.
- Project-memory governance becomes explicit: old formats remain immutable and
  new decisions cannot silently add another legacy exception.

### Negative or limiting

- `pytest` is now part of every default uv sync, even for users who only need
  runtime tooling.
- Kaggle's flat compatibility modules and root wrappers remain in place; this
  repair does not authorize a broader provider-runtime extraction.
- The historical baseline is a maintained fixture: intentionally changing a
  legacy ADR requires an explicit memory-governance decision rather than a
  casual formatting edit.
- This is a local repository repair only. It does not qualify a cloud run or
  verify the 100B corpus.

## Validation

The recorded local measurements before the added project-memory test were:

- `uv run --extra model pytest -q`: **701 passed, 1 skipped**.
- `uv run --extra model python -m unittest discover -v`: **590 tests, 1 skip**.

These are pre-addition measurements, not final totals. The offline contracts
also cover the locked default groups, root/`src` Kaggle imports and paths,
R-SFT saved-plan drift rejection, and the hashed historical ADR baseline. No
cloud qualification or corpus verification was done as part of this repair.

## Links

- [`../README.md`](../README.md)
- [`0139-kaggle-directory-reorganization.md`](0139-kaggle-directory-reorganization.md)
- [`0145-synchronize-readme-lifecycle-with-current-state.md`](0145-synchronize-readme-lifecycle-with-current-state.md)
- [`0146-normalize-kaggle-probe-runtime-paths-after-src-move.md`](0146-normalize-kaggle-probe-runtime-paths-after-src-move.md)
- [`0068-make-plain-uv-sync-install-complete-runtime.md`](0068-make-plain-uv-sync-install-complete-runtime.md)
- [`../../kaggle/README.md`](../../kaggle/README.md)
- [`../../post_training/R-SFT/production.py`](../../post_training/R-SFT/production.py)
- [`../../tests/test_project_memory.py`](../../tests/test_project_memory.py)
- [`../../tests/fixtures/project_memory_adr_legacy_baseline.json`](../../tests/fixtures/project_memory_adr_legacy_baseline.json)
