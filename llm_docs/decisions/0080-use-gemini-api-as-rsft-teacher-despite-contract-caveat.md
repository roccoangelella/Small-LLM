---
status: accepted
date: 2026-08-14
---

# ADR 0080 — Use Gemini API as the R-SFT teacher despite the contract caveat

## Context and problem statement

The first R-SFT pilot needs enough varied teacher traces to exercise the frozen reasoning contract, but generated content cannot be treated as verified ground truth and the selected provider has a recorded contract caveat.

## Considered options

1. Defer teacher-backed generation entirely.
2. Bind the dataset pipeline directly to Gemini output.
3. Use Gemini behind a provider-neutral interface and accept examples only after local validation and verification.

## Decision outcome

The user decided to proceed with Gemini API as the teacher for generating the first reasoning-SFT dataset/pilot.

The generation pipeline should remain provider-neutral and should not trust teacher output as ground truth. Small-LLM code will define the reasoning skill, L1/L2/L3 band, structural constraints, and verifier-visible answer contract; Gemini will generate/naturalize prompts and concise reasoning traces under that contract. Generated examples must be schema-validated, independently verified wherever possible, deduplicated, and rejected when malformed, incorrect, overly verbose, or outside the requested difficulty band.

Hundreds of API calls are considered sufficient for the first R-SFT pilot because each call should generate a batch of multiple independent structured examples rather than one example per call.

## Contract caveat

Google's Gemini API Additional Terms effective 2026-03-23 state that users may not use the Services to develop models that compete with the Services (for example Gemini API or Google AI Studio). The terms do not define a model-size, student-project, or commercial-scale exception. The user nevertheless elected to proceed with Gemini for this project. This ADR records the project decision and the caveat; it does not characterize the use as legally or contractually risk-free.

## Data-safety boundary

Do not submit secrets, credentials, personal data, confidential material, or proprietary project information as generation content. The dataset prompts should consist only of synthetic/public reasoning specifications and generated task material.

## Implementation direction

1. Keep the teacher wrapper behind a provider-neutral interface.
2. Batch multiple independent examples per API request.
3. Prefer deterministic/procedural ground truth for verifiable task families.
4. Require structured output containing skill, difficulty, prompt, reasoning trace, and final answer.
5. Run local exact verifiers before accepting an example.
6. Track raw generation provenance, model/version, prompt template, generation parameters, verifier result, and rejection reason.
7. Freeze the accepted dataset before the R-SFT serialization ablation and training pilot.

## Consequences

- Gemini output is untrusted input and must pass schema, correctness, provenance, and deduplication gates.
- The transport stays provider-neutral so a later teacher change does not redefine the dataset contract.
- The recorded contractual caveat remains an explicit project risk rather than an implicit assumption.
