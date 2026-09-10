---
status: current
last_reviewed: 2026-09-10
---

# ADR 0169 — MoE initial expert scaling sequence

## Context

The MoE design exploration moved beyond the original `moe-8e-top1` qualification geometry. The next experiments are intended to preserve a roughly comparable aggregate expert-width budget while testing whether more, smaller experts and a correspondingly larger Top-K improve specialization/routing without making active compute grow uncontrollably.

## Decision

Use the following MoE experimental sequence:

1. Initial experiment: **64 experts, Top-2, expert hidden width 352**.
2. Next scaling experiment: **128 experts, Top-4, expert hidden width 176**.
3. Following scaling experiment: **256 experts, Top-8, expert hidden width 88**.

These are ordered experiments, not three production configurations to launch simultaneously. The 64E/Top-2/352 configuration is the starting point; the later two are the predeclared follow-up comparisons.

## Consequences

- The existing `moe-8e-top1` branch/configuration is historical implementation state and does not represent the newly accepted starting geometry.
- Any wiring of the new geometry must preserve the accepted routing design decisions from the MoE planning work and be separately reviewed before implementation.
- Results should be compared with attention to active parameters/FLOPs, routing specialization, expert utilization, and training throughput rather than total parameter count alone.
