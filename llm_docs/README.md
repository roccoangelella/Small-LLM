# Small LLM Technical Documentation

This directory contains the detailed, topic-based specification for the Small LLM project. `LLM_PROJECT_MEMORY.md` remains the compact decision log and current-status summary; the files here contain the fuller technical contracts and rationale needed for implementation.

## Documents

- [`model_architecture.md`](model_architecture.md): decoder block structure, mixers, normalization, position handling, FFNs, embeddings, and output path.
- [`model_geometry.md`](model_geometry.md): scalable model family, frozen smoke and approximately 100M configurations, parameter accounting, and hardware-friendly dimensions.
- [`dataset_and_tokenization.md`](dataset_and_tokenization.md): tokenizer, source corpus, packing, and the interface expected by the model.
- [`training_and_evaluation.md`](training_and_evaluation.md): currently decided training constraints plus unresolved trainer and evaluation choices.
- [`decisions_and_ablations.md`](decisions_and_ablations.md): frozen defaults, controlled ablations, and architecture questions that remain open.

## Maintenance rule

When a project decision changes, update both:

1. `LLM_PROJECT_MEMORY.md`, as the concise source of current truth;
2. the relevant topic file in this directory, with implementation-level detail.

Do not silently overwrite historical reasoning. When replacing a decision, state what changed, why it changed, and which benchmark or operational result justified it.
