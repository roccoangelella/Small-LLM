# Technical reference

These documents define the **current** system in detail. They answer “what is the contract?” rather than “what command should I run?” or “how did we discover this?”.

- [`project_overview.md`](project_overview.md)
- [`model_architecture.md`](model_architecture.md)
- [`model_geometry.md`](model_geometry.md)
- [`gdn2_chunkwise_training.md`](gdn2_chunkwise_training.md) — readable recurrence/chunkwise correctness reference.
- [`gdn2_fla_backend.md`](gdn2_fla_backend.md) — selected CUDA GDN-2 production backend.
- [`dataset_and_tokenization.md`](dataset_and_tokenization.md)
- [`100m_10b_incremental_dataset.md`](100m_10b_incremental_dataset.md) — current ADR-0058 incremental producer/consumer contract.
- [`training_system.md`](training_system.md)
- [`optimizer_strategy.md`](optimizer_strategy.md)
- [`fp16_overflow_recovery.md`](fp16_overflow_recovery.md)
- [`training_and_evaluation.md`](training_and_evaluation.md)
- [`eval_core_v1_design.md`](eval_core_v1_design.md)
- [`post_training_sft.md`](post_training_sft.md)

Investigation chronology belongs in research/archive and measurements belong in evidence. When a reference conflicts with `../current/status.md` or an accepted ADR, current status and the accepted ADR take precedence until the reference is corrected.
