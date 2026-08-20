# R-SFT expanded-corpus resume — 2026-08-20

## Resume gate

The preserved expansion state was intact before new traffic: 9,624 frozen over-context candidates, 1,122 accepted historical Variant-D batch files (4,488 rewrites), and complete manual curation.

GemRouter's process environment said `GEMROUTER_BACKEND_ORDER=gemini-api`, but its live health endpoint still advertised `backendOrder=["gemini-api","nvidia"]` and fallback enabled. Source inspection showed that an enabled NVIDIA provider was appended automatically. The local GemRouter environment was therefore changed to `GEMROUTER_NVIDIA_ENABLED=false` and the service restarted. The post-restart health gate reported exactly `backendOrder=["gemini-api"]`, `fallbackEnabled=false`, and Gemini available.

A minimal request through Small-LLM's actual `GeminiDistillationClient` returned `quota-ok`. GemRouter audit evidence recorded `provider=gemini-api`, `backend=gemini-api`, and no provider fallback, confirming that the daily Gemini capacity had refreshed enough to resume.

## Curation correction

A four-row canary repeatedly failed with a Gemini empty-completion 502. Deterministic split recovery isolated the failure to candidate `0f4c5ad0-384c-4334-84a3-25c2fa8036e0`, whose task requested a naked imprisoned woman. That row conflicted with the safety policy already evident in the original curation.

A targeted semantic review of related v1 keepers found 24 clear policy misses. The historical curation was not modified. Instead, expansion curation v2 was frozen at:

```text
artifacts/rsft-superior-instruction-r0-adaptation/manual-curation.expanded-v2.jsonl
```

SHA-256:

```text
fb4da2929b47ececbde839da199437144677e4c7e1ea52ef2e8f6d4525ae1cde
```

Counts are 8,473 keep, 829 code exclusions, 212 math exclusions, and 110 safety exclusions. Twelve of the 24 corrected rows had already been adapted in the historical checkpoint and twelve had still been pending.

## Keeper-only resume

The new keeper stream harvests 4,009 still-valid old accepted keep rewrites and freezes only the 4,464 missing v2 keepers, or 1,116 maximum-size-four batches. The generated `keep-resume/` state is local/ignored so provider attempts can accumulate without polluting Git.

Initial v2 live execution accepted batches 1, 2, 3, 4, 5, 6, and 8 before the synchronous tool windows ended. At the last recorded evidence boundary this represented 28 new v2 keeper rewrites, leaving 4,436 pending; this number is expected to advance autonomously and should be read with the `status` command rather than treated as a frozen corpus count.

The only early v2 rejected attempts were output-contract errors (three records instead of four, or an item with the wrong fields). No NVIDIA traffic or quota-exhaustion error was observed in the v2 canary window.

## Operational status command

```bash
.venv/bin/python post_training/R-SFT/dataset/resume_superior_keep_adaptation.py status \
  --work-dir artifacts/rsft-superior-instruction-r0-adaptation
```

Final corpus size remains unknown until all 8,473 keepers have accepted rewrites and normalized-prompt collisions are audited against the baseline and one another.
