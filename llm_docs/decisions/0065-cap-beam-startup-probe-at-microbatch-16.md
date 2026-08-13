---
status: accepted
date: 2026-08-13
---

# ADR 0065: Probe Beam microbatches 8, 12, and 16 only

## Context

The completed 100M/2B H100 run froze microbatch 16. Beam's default serverless training GPU is the RTX 5090, which has materially less VRAM than the H100 used for that completed run. The Beam adapter currently probes a wider set including values above 16. Those candidates are not useful enough to justify spending startup GPU time on them before the real 100M training trajectory.

The startup probe already measures actual tokens per second, finite loss and gradient norm, and peak reserved CUDA memory, and selects the fastest safe measured candidate. The optimizer/data block remains 64 sequences; changing execution microbatch does not change the scientific batch contract.

## Decision

For fresh Beam single-GPU pretraining runs, the automatic startup qualification candidates are exactly microbatch 8, 12, and 16.

The probe must retain the existing measured-throughput and memory-safety behavior: short real forward/backward training probes, warmup exclusion, median tokens-per-second comparison, finite loss/gradient checks, peak reserved-memory measurement, and rejection above the existing reserved-memory safety fraction. The fastest safe candidate among 8/12/16 is frozen for the run and preserved on exact resume.

Do not probe Beam microbatches above 16 by default. Do not change the 64-sequence optimizer block or any scientific training schedule because of this execution-only choice.

## Consequences

The first real RTX 5090 training allocation can double as the live compatibility test: it pays only for the 8/12/16 startup probe and then continues directly into the authorized canonical trajectory with the selected microbatch. No separate paid GPU smoke is required solely to benchmark larger execution microbatches.
