---
status: accepted
date: 2026-08-24
supersedes: 0116
---

# ADR 0117 — Reject expanded 3-epoch R-SFT as a qualified default after full evaluation

## Context and problem statement

ADR 0116 provisionally promoted `100m-2b-rsft-r0-16716-e3-001` as the current R-SFT R0 default after the expanded 16,716-row run completed three exact passes. That promotion was explicitly provisional pending qualification.

The completed post-R-SFT qualification now compares `step-00001251` against the frozen S0 parent on eval-core retention, instruction behavior, S0 validation retention, and the frozen novel-reasoning suite. The result is negative on every headline comparison axis:

- eval-core loss: `+0.163692` worse;
- eval-core perplexity: `+5.334050` worse;
- top-1 / top-5 / top-10 accuracy: all lower;
- instruction-behavior pass rate: `0.066667 → 0.0`;
- novel-reasoning greedy accuracy: `0.457143 → 0.257143`;
- novel-reasoning sampled pass@1: `0.392857 → 0.282143`;
- frozen S0 validation loss: `+0.088378` worse.

The run did acquire part of the intended reasoning protocol: under the trained chat wrapper, reasoning starts in all 14 wrapper-robustness cases and 9/14 generations are well formed. Under plain and `Question: … Answer:` wrappers, reasoning-start and well-formed rates both fall to zero. This is evidence of chat-conditioned protocol learning, not improved reasoning correctness.

Canonical evidence: [`../evidence/rsft_e3_full_qualification_2026-08-24.md`](../evidence/rsft_e3_full_qualification_2026-08-24.md).

## Decision outcome

Supersede ADR 0116's provisional model-quality promotion.

`100m-2b-rsft-r0-16716-e3-001` / `step-00001251` is now classified as a **failed R-SFT model-improvement qualification** and an **experimental landmark**, not as a qualified reasoning improvement.

Concretely:

- Preserve the completed checkpoint and its Hugging Face/W&B history.
- Do not use its low in-distribution R-SFT validation loss as evidence that the recipe improved the model overall.
- Treat protocol acquisition and reasoning correctness as separate axes in subsequent R-SFT evaluation.
- Future R-SFT selection must include frozen S0 retention and novel-reasoning generalization; training/validation loss alone is insufficient.
- The exact next recipe — for example fewer passes, broader S0/instruction replay, or other mixture changes — is not chosen by this ADR and requires separate evidence/decision.
- This project-memory change does not itself edit the runtime chat registry. If the registry still resolves bare `--r-sft` to the e3 run, that is an operational pointer to an experimental checkpoint, not a qualification endorsement, until separately changed.

## Consequences

- ADR 0116 is superseded by the completed qualification it anticipated.
- Future project summaries must describe the e3 run as having learned a chat-conditioned reasoning protocol while regressing on frozen reasoning/general capability.
- `step-00001251` remains useful for ablations, qualitative comparison, and studying the gap between reasoning-shaped generation and correct reasoning.
- Any future default/promotion decision must be based on a new qualified checkpoint rather than inheriting ADR 0116's provisional status.