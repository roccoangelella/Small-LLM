# ADR 0148 — Skip separate fast smoke when evaluator startup dominates

Date: 2026-09-04

## Decision

For Kaggle evaluation-v2 runs where CUDA/kernel compilation and model initialization dominate startup cost, do not spend a separate session waiting for the `fast` suite solely as a timing smoke test. After the batching equivalence tests pass, proceed directly to the canonical `full` qualification run.

This is an operational scheduling choice only. It does not change the evaluation protocol, benchmark sample counts, scoring, decoding parameters, batching limits, or output schema defined by ADR 0140/0141/0147.

The current 100M/2B pretrained qualification should therefore run `suite=full` with `eval_core_v1 --batch-size 4`; L20 continues to use its own ADR-0147 internal length-bucketed batching limits.
