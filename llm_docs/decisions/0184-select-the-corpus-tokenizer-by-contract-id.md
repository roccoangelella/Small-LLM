---
status: proposed
date: 2026-09-22
supersedes: null
---

# 0184 — Select the corpus tokenizer by contract id

## Context and problem statement

The 100B v2 corpus (HF repo `roccoangelella/small-llm-corpus-100b-v2-workers`, 2,100
regions) was tokenized with a SuperBPE 8k artifact that is **not** the one frozen in
this repo by ADR 0175, even though both files are named `superbpe_8000.json`.

Verified identities (measured 2026-09-22 with `sha256sum` and `git hash-object`):

| artifact | SHA-256 | Git blob SHA-1 | size |
| --- | --- | --- | --- |
| this repo, `tokenizer/superbpe_8000.json` (run 001) | `4220ad83137407434877a128cb7f6b314161e8ed241514bfe9ba15045e899b06` | `a4daaa638a4d9db10270a65e8a6b55a9b94a9fd4` | 264,299 B |
| corpus v2 workers, `seed-v2/tokenizer/superbpe_8000.json` | `068e20ef8a16b1bff2a2a1d161e5a6f1cfd0935befa7c0ae734c9ec376d373bb` | `961597b0a1196a5dbfe1c0a18f2659f908b425ae` | 599,182 B |

The v2 identity is the one pinned as `tokenizer_sha256` in every
`corpus-13b-v2/worker-*/contract.json` of the corpus repository.

The two files are *geometrically* interchangeable and *semantically* incompatible.
Measured on the two payloads:

- identical: `version`, `truncation`, `padding`, `added_tokens`, `normalizer`,
  `pre_tokenizer`, `post_processor`, `decoder`; 8,000 vocabulary entries covering IDs
  0..7999 exactly; 7,749 merges; the eight specials at 7992..7999 with EOD 7992.
- different: 698 vocabulary strings exist only in one file and 698 only in the other;
  6,987 of the 7,302 shared strings carry a different ID; the merge tables differ.
- consequence: 0 of 300 English documents encode to the same token sequence under both
  artifacts.

Nothing in the repo would have caught the swap. `dataset/superbpe_retokenization.py`
pinned exactly one Git blob SHA-1, so pointing the producer at the v2 file would have
been rejected, and the only way to use it would have been to overwrite the file run 001
depends on.

Evidence about where the artifact is actually read (grep over every `*.py` in the repo,
including `chat.py`, `moe_production.py`, `MOE_model/`, `trainer/`, `tests/`):

- the *only* runtime load of `tokenizer/superbpe_8000.json` is
  `dataset/superbpe_retokenization.py::_load_target_tokenizer`, reached solely through
  `install_superbpe_retokenization`, which `dataset/production/cli.py` calls when
  `--tokenizer-contract superbpe_8000` is passed. That is the dataset **producer**, the
  GPT-2 → SuperBPE retokenization that run 002 does not use.
- the **trainer/consumer** never opens it. `MOE_model/config.py` hardcodes
  `semantic_vocab_size = 8_000`; `trainer/decode.py` only rejects shard tokens
  `>= semantic_vocab_size`; `trainer/shard_config.py` validates geometry, not tokenizer
  identity. The `tokenizer` block written into the dataset manifest by
  `dataset/production/cli.py` is never read back by any consumer.
- `chat.py` and the eval suites use the GPT-2 `tiktoken` encoding or the R-SFT
  reasoning tokenizer; neither touches the SuperBPE artifact.

So run `moe-100b-superbpe-b64-dataset-002`, which streams an already-produced v2
corpus, does not technically need any tokenizer file in this repo. It still needs the
artifact to be *present and identified*, because decoding a sample, auditing a shard,
or ever re-producing the corpus requires the exact file, and because "the file called
`superbpe_8000.json`" now means two different things across two repositories.

## Considered options

- **A. Overwrite `tokenizer/superbpe_8000.json` with the v2 file.** Breaks the ADR 0176
  fail-closed pin, changes the artifact run 001 is defined against, and makes the 001
  corpus unreproducible from this repo.
- **B. Add the v2 file under its own name and select the artifact by contract id.**
- **C. Add nothing; document in prose that the v2 corpus used a different tokenizer.**
  Correct today, because the trainer reads no tokenizer, but it leaves the v2 artifact
  absent from the repo, so nothing here can decode or re-produce that corpus, and the
  ambiguity of the file name stays unguarded.
- **D. Select the artifact through an environment variable.** The producer already
  carries an explicit `--tokenizer-contract` argument whose value is recorded in the
  manifest as `output_tokenizer_id`; an env var would be a second, unrecorded channel.

## Decision outcome

Chosen option: **B**, because it is the only option that keeps run 001 byte-identical
while making the v2 artifact usable and identifiable, and because the selector it needs
already exists and is already recorded in the manifest.

Add `tokenizer/superbpe_8000_v2.json`, byte-identical to
`seed-v2/tokenizer/superbpe_8000.json` of the corpus v2 workers
(SHA-256 `068e20ef…d373bb`, Git blob `961597b0…425ae`).

Turn the single hardcoded pin in `dataset/superbpe_retokenization.py` into a registry
of `TokenizerIdentity(contract_id, relative_path, expected_git_blob_sha1)`:

- `superbpe_8000` → `tokenizer/superbpe_8000.json`, blob `a4daaa63…9fd4` (unchanged).
- `superbpe_8000_v2` → `tokenizer/superbpe_8000_v2.json`, blob `961597b0…425ae`.

`install_superbpe_retokenization(root)` keeps `superbpe_8000` as its default, so every
existing call site — including `dataset/moe_100b.py`, which hardcodes
`--tokenizer-contract superbpe_8000` for run 001 and forbids overriding it — behaves
byte-identically. `dataset/production/cli.py` gains `superbpe_8000_v2` as a third choice
for `--tokenizer-contract`. Each artifact keeps its own fail-closed blob check; mixing
artifacts within one corpus remains impossible by construction, because the contract is
chosen once per producer process and recorded in the manifest.

The unqualified module constants `TARGET_TOKENIZER_ID`, `TARGET_TOKENIZER_RELATIVE_PATH`
and `EXPECTED_TOKENIZER_GIT_BLOB_SHA1` keep meaning run 001's artifact, as aliases of
`SUPERBPE_8000`.

No trainer, `MOE_model`, `chat.py` or evaluation code is touched.

## Consequences

### Positive

- Run 001 is untouched: same file, same pin, same default, same CLI invocation, same
  configuration and schema hashes.
- The v2 corpus's tokenizer is now present in the repo and identified by hash, so a
  shard produced by the v2 workers can be decoded and audited here.
- The producer fails closed per contract: each id is bound to one blob identity, so a
  drifted or swapped artifact is rejected rather than silently used.

### Negative or limiting

- The repo carries two 8k artifacts with the same vocabulary size. They are
  distinguishable only by hash and by contract id, never by geometry — treat any claim
  that they are interchangeable as false.
- Known gap, deliberately not closed here: the trainer does not verify that the streamed
  dataset manifest's `tokenizer.tokenizer_sha256` matches any expected value. Nothing
  enforces that run 002 streams v2-tokenized shards into a model initialised for that
  vocabulary. This was already true for run 001, and closing it would change consumer
  behaviour mid-run; it needs its own decision.
- A model pretrained on the v2 corpus is not vocabulary-compatible with a checkpoint
  pretrained on the 001 corpus, despite both reporting `semantic_vocab_size = 8000`.
- This authorizes the selection mechanism, not a corpus run, a deployment or a push.

## Validation

`tests/test_superbpe_100b_pipeline.py::TokenizerContractSelectionTests` asserts: the
default contract is still run 001's artifact and pin; every registered artifact's Git
blob SHA-1 matches the file on disk; the v2 file's SHA-256 equals the corpus pin; both
artifacts share IDs 0..7999, the eight specials, the pre-tokenizer and the decoder while
differing in vocabulary and merges; the same text encodes differently under the two;
installing `superbpe_8000_v2` reports the v2 artifact and hashes and restores cleanly;
an unknown contract id is rejected; the CLI exposes exactly the three choices; and no
Python module outside `dataset/` and `tests/` mentions a SuperBPE artifact — the
regression guard for the producer/consumer split this ADR relies on.

Run: `python -m pytest tests/test_superbpe_100b_pipeline.py
tests/test_production_cli_bootstrap_resume.py tests/test_modal_incremental_smoke.py`.

Live production with `--tokenizer-contract superbpe_8000_v2` has not been exercised.

## Links

- [ADR 0175 — Freeze the 8,000-token SuperBPE tokenizer](0175-freeze-the-8000-token-superbpe-tokenizer.md)
- [ADR 0176 — Retokenize 100B SuperBPE before stratification](0176-retokenize-100b-superbpe-before-stratification.md)
- [ADR 0178 — Publish the MoE 100B SuperBPE corpus to a public HF bucket](0178-publish-moe-100b-superbpe-corpus-to-public-hf-bucket.md)
- [ADR 0182 — Stream the MoE corpus while it is produced](0182-stream-the-moe-corpus-while-it-is-produced.md)
- [MR draft](../mr/2026-09-22-tokenizer-v2-corpus.md)
