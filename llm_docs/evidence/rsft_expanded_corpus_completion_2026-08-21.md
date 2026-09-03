# R-SFT expanded corpus completion — 2026-08-21

## Completion boundary

The keeper-only expansion lane reached full coverage: 8,473/8,473 curation-v2 keepers have accepted compressed supervision, 1,116/1,116 resume batches are complete, and `resume_pending_records=0`. The historical v1 curation/batches used by the trained 12,306-row model were not modified.

## Batch-305 safety recovery

The final unresolved batch was 305. A minimal authenticated GemRouter probe returned HTTP 200 and `quota-ok`, proving the endpoint itself was healthy. GemRouter remained hard Gemini-only with fallback disabled and NVIDIA disabled. The problematic candidate `ad53292c-7d5b-4fde-ae83-ba956724228d` appended an unsafe image-generation request involving minors to a benign meeting-reminder formatting task. Gemini repeatedly returned `gemini_api_empty_response` for that candidate even when isolated.

The row was recovered manually as an audited safe-refusal compression: the rewritten problem preserves the benign formatting constraints, explicitly marks the appended unsafe image request as something to refuse, and the answer preserves the source answer's refusal semantics. The recovered atomic serialization is 218 tokens. The exact recovery provenance is recorded in `artifacts/rsft-superior-instruction-r0-expanded/manual-safety-recovery.json`. No NVIDIA or alternate teacher provider was used.

## Frozen corpus

Finalization against the 7,683-row unchanged Superior baseline produced:

```text
path:      artifacts/rsft-superior-instruction-r0-expanded/reasoning.jsonl
rows:      16,716
sha256:    d13052b6fc33108ec65511b790a75f6473144855059b16b55167b046f787c405
bytes:     62,209,931
min/max atomic serialized tokens: 61 / 2,048
```

Composition is 7,683 unchanged Superior instruction rows, 8,403 unique simplified Superior instruction rows, and 630 Gemini logic anchors. Seventy otherwise accepted keeper rewrites were excluded because normalized-prompt deduplication found a collision with the baseline or another accepted rewrite. The adjacent generated manifest records all 70 excluded IDs and the full source/candidate/curation hashes.

## Verified native bundle

The deterministic atomic production builder completed successfully (`rc=0`) against the completed 100M/2B S0 bundle:

```text
bundle: <working-root>/rsft-r0-superior-instruction-expanded-16716
train blocks: 417
train packed records: 20,313
reasoning train targets: 12,077,733
S0 retention train targets: 1,343,090
total train targets: 13,420,823
validation: 4 blocks / 182 records / 125,419 targets
test: 4 blocks / 182 records / 117,184 targets
optimizer target tokens/block: 32,768
context length: 2,048
seed: 17
```

The train manifest identity is `476bda8bbc00129b2711f947e470422d28e86eda458d19c70ee544ad3f8c80f7`. Atomic marker IDs remain `<think>=50257`, `</think>=50258`, `<answer>=50259`.

This completes dataset creation and bundle preparation only. The currently accepted trained R-SFT model remains `100m-2b-rsft-r0-12306-001`; any expanded-corpus training must receive a new run identity.
