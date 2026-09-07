# ADR 0155 — Investigate training-free looping on the frozen 100M/10B model

Date: 2026-09-07
Status: Accepted; loop design pending joint review

## Decision

Investigate whether the completed 100M/10B pretrained Small-LLM checkpoint can gain capability at evaluation time through **training-free recurrent/looped depth**.

This experiment must:

- keep the trained 100M/10B checkpoint weights frozen;
- require no retraining, continued pretraining, or architecture-specific fine-tuning;
- preserve the existing canonical evaluation-v2 benchmark cases and scoring contracts so the normal and looped executions are directly comparable;
- isolate implementation work on branch `looping_transformer` rather than changing the production model path on `main`;
- treat the normal, unmodified forward pass as the primary baseline;
- record added inference compute/runtime alongside any capability change.

## Not yet decided

No looping mechanism is accepted yet. In particular, the following remain open and must be designed before implementation:

- which physical layers or layer groups are recurrent;
- whether recurrence applies to a complete DecoderBlock, only the GDN-2/MHA mixer, or another subcomponent;
- recurrence count / effective depth;
- residual damping or interpolation rule for repeated applications;
- whether recurrence is uniform or layer-dependent;
- whether the first experiment uses one literature-derived configuration or a controlled ablation set;
- stopping/acceptance criteria for deciding whether training-free looping is beneficial.

## Rationale

Recent 2025–2026 recurrent-depth work provides evidence that repeated latent computation can improve capability without increasing stored parameter count, while also showing that naive repeated application of pretrained blocks can destabilize or degrade performance. Because the current 100M/10B checkpoint is already trained and retraining a different recurrent architecture is outside the intended scope, the relevant question for Small-LLM is specifically whether a carefully controlled **training-free** looping transform can improve evaluation results.

The experiment is especially relevant because Small-LLM is a 20-layer hybrid decoder with a repeating GDN-2/GDN-2/GDN-2/MHA schedule, so full-block recurrence and mixer-only recurrence are materially different hypotheses and must not be conflated.

## Implementation gate

Do not wire a looping implementation until the loop semantics have been jointly specified and the user has demonstrated understanding of the chosen mechanism through open-ended design questions, per project protocol.
