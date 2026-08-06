---
status: current
last_reviewed: 2026-08-06
---

# Current roadmap

## While the 100M-token run is active

1. Build the immutable `eval_core_v1` corpus from the existing validation hash partition.
2. Verify fast is a document-level subset of full and freeze the realized manifest hashes.
3. Benchmark fast and full evaluation time, throughput, peak VRAM, and useful batch size on a T4.
4. Run the unified full suite on the accepted 10M checkpoint to create the historical scorecard anchor.
5. Keep the training architecture and data recipe unchanged unless the active run exposes a defect.

## When the run completes

1. Verify the final checkpoint, consumed-token count, dataset identity, and absence of non-finite training events.
2. Run both fast and full `eval_core_v1` suites, including the existing printed prompt answers.
3. Compare the 10M and 100M loss, perplexity, BPB, top-k accuracy, calibration, cluster slices, throughput, and generation samples.
4. Fit the local data-scaling slope and review whether the 100M result lands inside the working loss band.
5. Decide whether to train the same 20M model longer or authorize the first larger model.

## Explicitly deferred

- matched all-attention and alternative-mixer training at the 20M stage;
- new attention mechanisms merely for novelty;
- longer context;
- model enlargement before the 100M result and evaluation evidence are reviewed;
- instruction tuning before the base-model learning curve is understood.

## Open decisions

- exact next model size after the 100M evaluation;
- whether GDN-2-specific parameterization or optimizer scaling experiments are needed before enlargement;
- which external zero-shot tasks enter the first stable public scorecard after intrinsic evaluation is accepted.
