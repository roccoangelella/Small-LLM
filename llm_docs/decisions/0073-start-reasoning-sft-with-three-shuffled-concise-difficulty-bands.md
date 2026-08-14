---
status: accepted
date: 2026-08-14
---

# ADR 0073 — Start reasoning SFT with three shuffled concise difficulty bands

## Decision

If the current 100M/2B instruction SFT passes behavioral qualification and is promoted into reasoning-oriented post-training, the first reasoning-SFT stage will use **concise reasoning traces on simple-to-moderate verifiable prompts**, organized into **three explicit difficulty/toughness bands**.

The three bands are a curriculum/data-selection device, not a monotonic presentation schedule. Training examples from the retained bands will be **shuffled/mixed**, rather than fed to the model in a strictly easy-to-hard sequence.

The purpose of the first reasoning-SFT stage is to establish a reliable short reasoning policy before attempting deeper or longer reasoning trajectories. Long monologue-style chain-of-thought is out of scope for the initial reasoning-SFT stage.

## Rationale

Recent small-model reasoning work supports capacity-aligned and concise reasoning distillation, while controlled curriculum studies do not establish universal gains from strict easy-to-hard sample ordering. For the approximately-100M student, the safer first experiment is therefore to control the difficulty distribution explicitly while avoiding a deterministic increasing-difficulty stream.

## Not yet decided

This ADR does **not** yet freeze:

- the exact definitions or thresholds for the three difficulty bands;
- whether the training corpus is primarily premade, synthetically generated, or hybrid;
- the teacher model/provider used for any synthetic traces;
- the exact number of reasoning examples or target-token budget;
- the exact trace-length ceiling;
- the verifier/judge stack;
- the proportions assigned to the three difficulty bands;
- whether subsequent reasoning stages use two, three, or more capability bands.

Those choices require a separate evidence-backed design decision.
