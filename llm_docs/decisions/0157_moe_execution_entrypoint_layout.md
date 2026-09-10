# ADR 0157: MoE execution entrypoint layout

- **Date:** 2026-09-06
- **Status:** Accepted

## Decision

The 8-expert Top-1 MoE experiment will expose execution entrypoints according to the environment in which each workload is actually run:

- `modal/moe_launch.py` for MoE pretraining launched on Modal.
- `beam/moe_launch.py` for MoE pretraining launched on Beam.
- `kaggle/moe_eval.py` for canonical MoE evaluation suites run on Kaggle.
- No MoE launch or evaluation entrypoints will live at repository root.

## Rationale

MoE pretraining is intended to run only on the supported high-performance training backends (Modal H100 or Beam RTX 4090), while evaluation suites are run on Kaggle. Keeping the entrypoints beside their execution environment makes the supported workflow explicit, reduces ambiguity about where a command is meant to run, and keeps the repository root free of environment-specific wrappers.

This is an entrypoint-layout decision only. It does not change the MoE architecture, optimizer assignment, routing semantics, training objective, checkpoint contract, or evaluation methodology.
