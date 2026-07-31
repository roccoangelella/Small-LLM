# Model Implementation Coding-Agent Task

Implement the Small LLM model package exactly as specified in `llm_docs/model_architecture.md`, `llm_docs/model_geometry.md`, `llm_docs/decisions_and_ablations.md`, and `journals/tomorrow's todo.md`.

Use three implementation subagents with clearly separated ownership:

1. shared Transformer components, attention, SWA-512, embeddings, and configuration;
2. readable PyTorch GDN-2, state/cache interfaces, and optimized-backend adapter;
3. model assembly, parameter accounting, initialization experiment, and tests.

Integrate their work into one coherent implementation. Then use two fresh review subagents independently:

1. correctness and architecture-contract review;
2. tests, numerical stability, maintainability, and regression review.

Resolve all review findings, run the complete test suite, and commit the finished implementation to `main`. Do not redesign frozen architecture decisions; document any unavoidable deviation before proceeding.