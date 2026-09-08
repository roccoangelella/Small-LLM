# ADR 0153 — Add a controlled top-k sampling benchmark

Date: 2026-09-08
Status: accepted in principle; exact positive `top_k` value pending implementation confirmation

## Decision

Add an additional sampled Base Prompt benchmark for both pretrained and SFT evaluation paths, keeping the current sampled temperature and nucleus settings (`temperature=1.0`, `top_p=1.0`) while applying a positive `top_k` cutoff.

This new view is intended to measure whether restricting sampling to the highest-probability token set improves robustness relative to the existing canonical sampled contract (`temperature=1.0`, `top_p=1.0`, `top_k=0`).

Important semantic clarification: in the current evaluator, `top_k=0` means top-k filtering is disabled (full-distribution sampling). Therefore a positive value such as `top_k=50` is a *narrower*, not broader, sampling distribution.

The existing canonical sampled benchmark remains unchanged for comparability with prior runs. The new top-k benchmark is additive and should be judgeable by the same GemRouter Base Prompt semantic-judge pipeline.

## Rationale

The current full-distribution sampled results are substantially below greedy results. A controlled positive-top-k view can distinguish general temperature sensitivity from degradation caused by allowing low-probability tail tokens into the candidate set.

## Pending implementation detail

Before wiring, the exact positive `top_k` value must be confirmed after the user demonstrates understanding of the difference between `top_k=0` and positive top-k filtering, per project workflow.
