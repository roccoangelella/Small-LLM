# MR draft — tokenizer v2 corpus

Branch: `edo/tokenizer-v2-corpus` (from `edo/modal-durable`). Not pushed.

---

## Title

Select the corpus tokenizer by contract id and add the v2 8k artifact

---

## Description

The 100B v2 corpus was tokenized with a SuperBPE 8k artifact that is **not** the one
frozen in this repo, even though both files are called `superbpe_8000.json`. This MR
adds the v2 artifact under its own name and makes the producer choose between the two
by contract id. Run `moe-100b-superbpe-b64-dataset-001` is untouched.

### The two artifacts

| artifact | SHA-256 | Git blob SHA-1 | size |
| --- | --- | --- | --- |
| `tokenizer/superbpe_8000.json` (run 001, ADR 0175) | `4220ad83137407434877a128cb7f6b314161e8ed241514bfe9ba15045e899b06` | `a4daaa638a4d9db10270a65e8a6b55a9b94a9fd4` | 264,299 B |
| `tokenizer/superbpe_8000_v2.json` (new; corpus v2 workers) | `068e20ef8a16b1bff2a2a1d161e5a6f1cfd0935befa7c0ae734c9ec376d373bb` | `961597b0a1196a5dbfe1c0a18f2659f908b425ae` | 599,182 B |

The new file is a byte-identical copy of
`small-lm-english-next/seed-v2/tokenizer/superbpe_8000.json`, which is the
`tokenizer_sha256` pinned in every `corpus-13b-v2/worker-*/contract.json` of
`roccoangelella/small-llm-corpus-100b-v2-workers` (2,100 regions).

They look interchangeable and are not:

- **identical**: `version`, `truncation`, `padding`, `added_tokens`, `normalizer`,
  `pre_tokenizer`, `post_processor`, `decoder`; 8,000 entries covering IDs 0..7999;
  7,749 merges; specials 7992..7999 with EOD 7992.
- **different**: 698 vocabulary strings exclusive to each file; 6,987 of the 7,302
  shared strings carry a different ID; different merge tables.
- **measured consequence**: 0 of 300 English documents encode to the same token
  sequence under both.

### Why not just overwrite the existing file

Because run 001 is defined against `a4daaa63…`, and
`dataset/superbpe_retokenization.py` fails closed on that exact Git blob. Overwriting
would silently redefine the 001 corpus identity and make it unreproducible from this
repo.

### Where the tokenizer is actually used

Grepped every `*.py` in the repo (`chat.py`, `moe_production.py`, `MOE_model/`,
`trainer/`, `tests/` included):

- the only runtime load of an 8k artifact is
  `dataset/superbpe_retokenization.py::_load_target_tokenizer`, reached only via
  `install_superbpe_retokenization`, called only from `dataset/production/cli.py` when
  `--tokenizer-contract superbpe_8000` is given — i.e. the **dataset producer**, the
  GPT-2 → SuperBPE retokenization that run 002 does not use.
- the **trainer/consumer never opens it**. `MOE_model/config.py` hardcodes
  `semantic_vocab_size = 8_000`; `trainer/decode.py` only rejects shard tokens
  `>= semantic_vocab_size`; `trainer/shard_config.py` checks geometry, not tokenizer
  identity. The `tokenizer` block that `dataset/production/cli.py` writes into the
  dataset manifest is never read back by any consumer.
- `chat.py` and the eval suites use GPT-2 `tiktoken` or the R-SFT reasoning tokenizer.

So run 002 needs no tokenizer file to train. It still needs the artifact **present and
identified**, for decoding samples, auditing shards, and re-producing the corpus — and
to stop the file name from meaning two different things across two repos.

### Change

- **new** `tokenizer/superbpe_8000_v2.json` — byte-identical copy, verified by
  `cmp` and both hashes.
- `dataset/superbpe_retokenization.py` — the single hardcoded pin becomes a registry
  of `TokenizerIdentity(contract_id, relative_path, expected_git_blob_sha1)`:
  `superbpe_8000` (unchanged pin) and `superbpe_8000_v2`.
  `install_superbpe_retokenization(root, *, tokenizer_contract=...)` defaults to
  `superbpe_8000`, so every existing call site is byte-identical. The unqualified
  constants `TARGET_TOKENIZER_ID`, `TARGET_TOKENIZER_RELATIVE_PATH` and
  `EXPECTED_TOKENIZER_GIT_BLOB_SHA1` stay, now as aliases of run 001's identity.
  Each contract keeps its own fail-closed blob check.
- `dataset/production/cli.py` — `--tokenizer-contract` gains the `superbpe_8000_v2`
  choice; default stays `gpt2`. The chosen id is what already lands in the manifest as
  `output_tokenizer_id`, so the selection is recorded, not ambient (no env var).
- `dataset/moe_100b.py` — **not touched**; it still hardcodes and locks
  `--tokenizer-contract superbpe_8000` for run 001.
- `tests/test_superbpe_100b_pipeline.py` — new `TokenizerContractSelectionTests`.
- `llm_docs/decisions/0184-select-the-corpus-tokenizer-by-contract-id.md` + README entry.

### Tests

```
python -m pytest tests/test_superbpe_100b_pipeline.py \
  tests/test_production_cli_bootstrap_resume.py tests/test_modal_incremental_smoke.py
25 passed, 12 subtests passed
```

Full suite, run on this branch and on its base `87e412f` with the same interpreter:

```
python -m unittest discover
base   87e412f : Ran 760 tests — FAILED (failures=5, errors=29, skipped=2)
branch          : Ran 770 tests — FAILED (failures=5, errors=29, skipped=2)
```

The failing and erroring test sets are **identical** on both (diffed by name): zero
regressions, +10 new passing tests. The 29 errors are import failures for optional
kaggle/modal/SFT dependencies missing from the local interpreter; the 5 failures are
pre-existing. One of them is worth a decision: `tests/test_project_memory.py` requires
every ADR to start with YAML frontmatter, and
`0183-continue-moe-training-across-provider-accounts.md` does not. ADR 0184 here follows
`llm_docs/decisions/template.md` and passes; 0183 was deliberately left untouched rather
than reformatting an accepted decision record in this MR.

New assertions: the default contract is still run 001's artifact and pin; every
registered artifact's blob SHA-1 matches the file on disk; the v2 SHA-256 equals the
corpus pin; both artifacts share IDs 0..7999, specials, pre-tokenizer and decoder while
differing in vocabulary and merges; the same text encodes differently under the two;
installing `superbpe_8000_v2` reports the v2 artifact and hashes and restores cleanly;
an unknown contract id is rejected; the CLI exposes exactly three choices; and no Python
module outside `dataset/` and `tests/` mentions a SuperBPE artifact — the regression
guard for the producer/consumer split.

### Known gap, deliberately left open

Nothing verifies that a streamed dataset manifest's `tokenizer.tokenizer_sha256`
matches what the run expects. Today nothing stops run 002 from streaming v2-tokenized
shards into a model initialised for the 001 vocabulary, or vice versa — both report
`semantic_vocab_size = 8000`. This was already true for run 001; closing it changes
consumer behaviour mid-run and deserves its own decision. Flagged in ADR 0184.

### Reviewer checklist

- [ ] Confirm the v2 artifact is the file the corpus workers used
      (`sha256sum tokenizer/superbpe_8000_v2.json` → `068e20ef…d373bb`).
- [ ] Confirm run 001's file and pin are unchanged (`git diff` shows no change to
      `tokenizer/superbpe_8000.json`, and `SUPERBPE_8000.expected_git_blob_sha1` is
      still `a4daaa63…9fd4`).
- [ ] Agree that run 002 needs no trainer-side tokenizer change, or say what the
      trainer should verify instead.
- [ ] Decide whether the manifest-tokenizer check above becomes a follow-up.
- [ ] Accept or reject the new `llm_docs/mr/` directory — the `llm_docs` taxonomy in
      `AGENTS.md` has no home for MR drafts, so this adds one rather than bending an
      existing category.
- [ ] Decide whether ADR 0183 should get the YAML frontmatter its own repo test
      requires (out of scope here).

---

## Second commit: `tools/pack_regions_to_shards/`

The converter that turns the 100B v2 region corpus into the schema-v2 shards this trainer consumes
(`run/moe-100b-superbpe-b64-dataset-002/...` in bucket `roccoangelella/small-llm-corpus-100b-v2-dataset`).
It drives this repository's own producer classes and `verify()`; nothing about the format is
reimplemented. Evidence recorded in the project hub investigation of 2026-09-22:

- shards byte-identical between the reference `SequencePacker` path (`--packer slow`) and the
  vectorised `FastBlockPacker` (`--packer fast`), on real regions;
- `verify(full_scan=True)` passes; max token id 7992; separators counted;
- crash after a checkpoint + resume reproduces the uninterrupted output byte for byte;
- contract/frontier/manifest read back with `incremental_frontier.read_run_contract/read_frontier`.

`config.EOD_TOKEN_ID` is forced to 7992 at import — the producer classes insert it and the module
default is GPT-2's 50256, outside the 8000-entry vocabulary. The default `--tokenizer` is
`tokenizer/superbpe_8000_v2.json` (first commit), i.e. the artifact the regions were built with.

Run from the repo root; `SMALL_LLM_BUILDER_ROOT` points at the corpus builder checkout (its
`merge_workers.load_workers` reads the worker receipts). This copy is the versioned twin of the
script currently producing the 002 dataset from the builder directory.
