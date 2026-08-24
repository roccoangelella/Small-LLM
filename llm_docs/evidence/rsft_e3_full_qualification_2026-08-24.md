---
status: evidence
date: 2026-08-24
---

# Expanded 3-epoch R-SFT full qualification

## Artifact identity

The qualified checkpoint is the completed expanded-corpus R-SFT run:

```text
run ID:       100m-2b-rsft-r0-16716-e3-001
checkpoint:   step-00001251
train blocks: 417 × 3 exact passes
consumed loss-bearing targets: 40,262,469
```

The frozen R-SFT train manifest identity is `476bda8bbc00129b2711f947e470422d28e86eda458d19c70ee544ad3f8c80f7`. The qualification report has SHA-256 `71441782f474c73c70d3b385e8f4d764fbc0301cf0e5332642d8f4631484c104`; the novel-reasoning suite identity is `bc276fa2dbdf1e83f21822f0f8ceb9a4e89f6c39d4008b4bdf8863053f86fa17`.

The comparison uses the completed S0 parent `100m-2b-sft-s0-001` as the frozen baseline.

## Frozen S0 → R-SFT comparison

| metric | S0 | R-SFT e3 | R-SFT − S0 |
|---|---:|---:|---:|
| eval-core loss | 3.400914 | 3.564607 | +0.163692 |
| eval-core perplexity | 29.991506 | 35.325556 | +5.334050 |
| eval-core BPB | 1.061539 | 1.112633 | +0.051094 |
| eval-core top-1 | 0.395538 | 0.380169 | -0.015369 |
| eval-core top-5 | 0.614231 | 0.597042 | -0.017188 |
| eval-core top-10 | 0.688148 | 0.671240 | -0.016909 |
| instruction-behavior pass rate | 0.066667 | 0.000000 | -0.066667 |
| novel reasoning greedy accuracy | 0.457143 | 0.257143 | -0.200000 |
| novel reasoning sampled pass@1 | 0.392857 | 0.282143 | -0.110714 |
| S0 validation retention loss | 1.700563 | 1.788942 | +0.088378 |

The e3 checkpoint therefore regresses against S0 on every frozen headline comparison axis reported by the qualification: general eval-core quality, instruction behavior, novel reasoning, and S0 retention.

## Behavioral failure mode

The 30-case instruction-behavior suite is a structural failure, not merely a small accuracy regression:

- pass rate: `0/30 = 0.0`
- well-formed `<think>…</think><answer>…` rate: `0.066667`
- non-empty answer rate: `0.066667`
- single `<think>` start rate: `0.466667`
- EOS termination rate: `0.5`
- runaway rate: `0.5`

Many generations either never enter the full protocol or enter `<think>` and fail to close it before running away. The few well-formed outputs show that the special-token path is learnable, but protocol reliability is not established by this run.

## What the R-SFT run did learn

The qualification still provides positive evidence that R-SFT changed generation behavior. On the dedicated chat-conditioned prompt-wrapper test, the chat wrapper starts reasoning in `14/14` cases (`1.0`) and produces a fully well-formed reasoning/answer protocol in `9/14` cases (`0.642857`). This demonstrates acquisition of the intended reasoning mode under the trained chat context.

However, that behavior is strongly template-dependent:

| wrapper | reasoning-start rate | well-formed rate | answer accuracy, any format |
|---|---:|---:|---:|
| chat | 1.000000 | 0.642857 | 0.214286 |
| plain | 0.000000 | 0.000000 | 0.214286 |
| `Question: … Answer:` | 0.000000 | 0.000000 | 0.357143 |

The correct interpretation is therefore two-axis: protocol transfer and answer correctness are separate. The checkpoint learned a chat-conditioned reasoning-generation mode, but this did not translate into improved reasoning correctness.

Several novel-reasoning examples reinforce this distinction: outputs can be syntactically well formed while the reasoning text is contradictory or semantically wrong. The run therefore teaches a useful project lesson: **learning to emit reasoning-shaped text is not evidence of learning better reasoning**.

## Qualification interpretation

This checkpoint is a **failed model-improvement qualification**. Its low in-distribution R-SFT validation loss must not be used as evidence that the three-pass recipe improved the model overall. Repeated narrow-distribution optimization improved fit to the R-SFT training distribution while degrading the frozen general-language and novel-reasoning evaluations.

Preserve `step-00001251` as an experimental landmark because it demonstrates real acquisition of the R-SFT protocol. Do not treat it as a qualified reasoning improvement or as evidence that three exact passes over this corpus are beneficial.

The next R-SFT iteration should be selected on frozen retention plus novel-reasoning generalization, not R-SFT validation loss alone. Exact next-recipe choices such as fewer passes or broader S0/instruction replay remain a separate decision rather than an implication of this evidence record.