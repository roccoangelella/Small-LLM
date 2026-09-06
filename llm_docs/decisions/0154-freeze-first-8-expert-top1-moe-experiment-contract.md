# ADR 0154 — Freeze first 8-expert Top-1 MoE experiment contract

Date: 2026-09-06
Status: Accepted, implementation pending router/load-balancing design review

## Decision

The first Small-LLM MoE experiment will be implemented on a dedicated branch and will focus only on an 8-expert, Top-1 sparse FFN architecture.

Frozen choices:

- `num_experts = 8`.
- `top_k = 1` for the first implementation and run. Top-2 is explicitly deferred to a later ablation.
- Every expert is a full architectural copy of the current dense SwiGLU FFN, preserving `d_model=512` and `d_ff=1408`; experts are independently initialized rather than cloned from pretrained weights.
- The MoE replaces the dense FFN pathway while leaving the current mixer architecture unchanged.
- Training starts from scratch. The completed dense 100M/10B checkpoint is a comparison baseline, not an initialization source.
- The router parameters use AdamW rather than Muon and initially use the same base learning rate as the AdamW branch of the training recipe. A separate router LR is not introduced in the first experiment.
- Sparse execution must dispatch tokens only to selected experts; computing all experts and masking unused outputs is forbidden.
- The implementation is isolated under a new root-level `MOE_model/` package/directory rather than modifying the dense model package in place.
- MoE training receives a dedicated root-level `moe_launch.py` entrypoint.
- The implementation will live on a new dedicated Git branch rather than being wired directly on `main`.
- MoE checkpoints must serialize every architecture, routing, balancing, optimizer, scheduler, dataset-position, RNG, precision, and run-identity field needed to fail closed on incompatible resume. Exact fields will be finalized with the router/load-balancing contract.
- W&B telemetry must include MoE-specific routing/load metrics in addition to the existing training metrics.
- Before a 10B-token production run, the implementation must pass routing, gradient-flow, optimizer-routing, accounting, checkpoint/resume, no-token-loss, stability, throughput, and memory preflight tests.

## Open design items before implementation

The following are deliberately not frozen yet and must be understood and selected before code is wired:

1. exact router scoring/gating definition for Top-1, including how the router receives task-loss gradients;
2. load-balancing mechanism;
3. dropless versus capacity-limited dispatch;
4. router stabilization terms such as z-loss and their coefficients;
5. router initialization details and any exploration/noise policy;
6. exact MoE checkpoint schema and telemetry derived from the selected routing/balancing mechanism.

The implementation branch may be created before these items are frozen, but MoE routing/training code must not be wired until the design review is complete.
