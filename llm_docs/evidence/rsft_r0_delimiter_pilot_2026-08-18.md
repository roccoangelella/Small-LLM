---
status: evidence
observed: 2026-08-18
---

# 100M/2B R-SFT R0 delimiter pilot

Both 630-example delimiter arms completed on Kaggle from the same S0 parent checkpoint identity and the same R-SFT source-manifest identity.

## Shared parent

- parent run: `100m-2b-sft-s0-001`
- parent checkpoint: `step-00002485`
- parent consumed tokens: 80,039,261
- parent identity SHA-256: `93adb9bb3d1d884889c480282c25e8b130206f489569c0d3a255f94595623775`
- shared R-SFT source-manifest SHA-256: `2c70d8524179721f49de2c6abd5a3b722b96a0ec37dcedb456a7cb8da2a28c46`

## Textual arm

- run: `100m-2b-rsft-r0-textual-pilot-001`
- complete: yes
- steps: 30
- train loss-bearing targets: 58,099
- validation targets: 1,779
- validation loss: 2.0444399194143807
- validation perplexity: 7.7248307983242155
- bundle manifest SHA-256: `44d02a526b37ed72f79acaa35da2c9be2fc989f5a5b83f4aa722239591cc1076`

## Atomic arm

- run: `100m-2b-rsft-r0-atomic-pilot-001`
- complete: yes
- steps: 29
- train loss-bearing targets: 54,571
- validation targets: 1,653
- validation loss: 2.4455797088124576
- validation perplexity: 11.537235897919775
- bundle manifest SHA-256: `f0fee40b8b4fb51cdd5e1c82698ce621fc31130c8b5f2a6349da2fc4e7772d8a`

## Interpretation boundary

The textual arm had the lower teacher-forced validation loss in this small one-pass pilot. This evidence does not establish that textual delimiters are the preferred architecture. The atomic arm uniquely had to learn three newly promoted, highly frequent control-token rows from scratch during the short run, and the two arms contain different counts of loss-bearing delimiter tokens.

ADR 0099 records the project owner's subsequent architectural decision to use atomic special tokens for production despite the pilot validation-loss difference. The production choice is based on unambiguous reasoning/answer control semantics and parseability, not a claim that the atomic arm won the pilot metric.

Behavioral R-SFT qualification was intentionally deferred in both run summaries, so this evidence contains no behavioral pass-rate, EOS, runaway, or reasoning-correctness comparison.
