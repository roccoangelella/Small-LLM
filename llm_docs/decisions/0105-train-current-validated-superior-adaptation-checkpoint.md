---
status: accepted
date: 2026-08-19
supersedes: 0104
---

# ADR 0105 — Train the current validated Superior adaptation checkpoint

## Context

ADR 0103 selected the fidelity-first Variant-D compressor for Superior instruction examples that exceed the 2,048-token atomic R-SFT context. Manual semantic curation is now complete for all 9,624 over-context candidates: 8,497 are retained, 829 are code-primary exclusions, 212 are math-primary exclusions, and 86 are safety exclusions.

GemRouter adaptation is only partially complete because the Gemini quota was exhausted. There are 1,122 accepted four-row batch files covering 4,488 candidates. Of those accepted records, 4,021 have a final `keep` curation decision; 4,476 additional kept candidates still await compression.

A baseline-aware normalized-prompt audit found that 28 of the 4,021 accepted kept rewrites cannot safely coexist in the current training corpus. Twenty-two rewrites collapse onto a normalized prompt already present in the unchanged Superior baseline, and six additional rewrites collapse onto another accepted rewrite. Their reasoning/answer targets are not identical, so keeping both copies would create conflicting supervision for an identical model input.

## Decision

Freeze and train the currently available conflict-free checkpoint rather than waiting for the remaining adaptations.

The committed reasoning corpus is:

```text
artifacts/rsft-superior-instruction-r0-checkpoint-12306/reasoning.jsonl
```

It contains exactly:

- 7,683 unchanged, clean, context-fit Superior `instruction_following` rows;
- 3,993 accepted, manually-kept, Variant-D Superior rewrites with unique normalized prompts;
- 630 frozen Gemini logic anchors;
- 12,306 total reasoning examples.

The 28 prompt-colliding accepted rewrites are excluded from this checkpoint and recorded explicitly in the adjacent manifest. The remaining 4,476 manually-kept candidates are recorded as pending rather than treated as rejected.

Every emitted row must pass the exact atomic R-SFT serialization at context length 2,048. The reasoning/S0 mixture remains 90% reasoning and 10% completed S0 instruction retention by loss-bearing target tokens, with a 32,768-target optimizer block and one exact pass.

The Kaggle production run identity for this checkpoint is:

```text
100m-2b-rsft-r0-12306-001
```

This identity is intentionally distinct from both the earlier 8,313-row run configuration and any future complete 16k-scale corpus so checkpoints cannot cross-resume.

Canonical launch remains:

```bash
python kaggle/launch_r_sft.py train --model 100M --tokens 2B
```

The launcher SHA-pins the committed reasoning JSONL, resolves the frozen 100M/2B S0 bundle, builds and verifies the atomic production bundle, and then dispatches the qualified dual-T4 trainer.

A local build against the completed S0 bundle verifies 361 train blocks. The train stream contains 10,448,098 reasoning loss-bearing target tokens and 1,161,354 S0-retention target tokens, for 11,609,452 total target tokens and a realized retention share of approximately 10.004%. Validation and test each contain 138 reasoning records packed into four blocks.

## Repository hygiene

The adaptation workspace is generation state, not source code. The 170 MiB candidate cache, accepted provider batches, rejected attempts, OpenCode review state, review extracts, hand-review packs, and logs are ignored by Git. The small candidate manifest and final manual-curation JSONL remain committable audit artifacts. Local accepted batch files and the candidate cache are retained on the VPS so the remaining 4,476 kept candidates can be resumed later.

## Validation

Before launch, require all of the following:

- exact 12,306 reasoning rows and the pinned JSONL SHA-256;
- zero duplicate normalized prompts in the emitted checkpoint;
- every reasoning record at or below 2,048 atomic serialized tokens;
- exact 630 Gemini anchor count;
- exact 3,993 trainable adapted Superior count and 4,476 pending kept adaptations;
- native atomic production bundle verification against the completed S0 bundle;
- Kaggle launcher tests showing the 12,306-row checkpoint path and distinct run ID.
