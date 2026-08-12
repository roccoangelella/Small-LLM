---
status: accepted
date: 2026-08-12
supersedes: null
---

# 0050 — Scale the 100M model to a fresh 10B-token run with a 5B capability gate

## Context and problem statement

The approximately-100M / 2B Modal trajectory is nearing completion and has shown substantially lower validation loss and perplexity than the smaller-model trajectories. Those intrinsic metrics are encouraging but are not by themselves sufficient evidence of improved usable capability; the completed 100M / 2B model must still pass the project's frozen behavioral and intrinsic post-pretraining qualification.

The project's scaling principle is to keep a model geometry fixed and increase training data while downstream performance continues to improve, rather than enlarging the model again immediately. The next experiment should therefore test whether the fixed approximately-100M geometry continues to gain capability with substantially more data without committing all 10B-token compute before an intermediate performance check.

## Considered options

- Continue the existing 100M / 2B checkpoint directly to 10B tokens.
- Train a separate fresh 100M / 5B run, then decide whether to train another fresh 100M / 10B run.
- Start a fresh 100M / 10B run and perform the full qualification suite at approximately 5B consumed tokens as an intermediate continuation gate.
- Increase model size immediately after the 100M / 2B experiment.

## Decision outcome

Chosen option: **after the completed 100M / 2B behavioral qualification confirms a material capability improvement, launch a fresh approximately-100M / 10B-token pretraining trajectory and use approximately 5B consumed tokens as an intermediate full-evaluation gate.**

The 10B trajectory must:

- start from fresh initialization rather than resume the 100M / 2B terminal checkpoint;
- retain the same approximately-100M model geometry, initialization seed/policy, tokenizer/context, optimizer family, GDN-2 execution contract, precision policy, and data-policy assumptions unless a later ADR explicitly changes one;
- use a new deterministic finite approximately-10B-token training corpus and a one-pass token-based WSD schedule derived for that corpus;
- preserve normal durable checkpointing/W&B semantics so an approximately-5B consumed-token checkpoint can be qualified without disrupting training;
- run the project's full post-pretraining evaluation suite at approximately 5B consumed tokens;
- continue toward 10B when the 5B evaluation shows meaningful downstream capability growth relative to 100M / 2B;
- stop early when downstream capability has materially plateaued, even if validation loss is still improving.

The approximately-5B checkpoint is explicitly an **intermediate checkpoint of the 10B WSD trajectory**. It is not treated as equivalent to the terminal checkpoint of an independently scheduled 5B run because it has not necessarily undergone the cooldown that a standalone 5B WSD run would use.

No separate fresh 5B training run is authorized by this ADR. A standalone 5B endpoint can be authorized later only if a rigorously terminal 5B comparison becomes scientifically necessary.

The trigger for the fresh 10B run is behavioral/capability evidence from the completed 100M / 2B qualification, not validation loss or perplexity alone.

## Consequences

### Positive

- Tests the project's fixed-model data-scaling hypothesis cleanly with a fresh independent 10B trajectory.
- Avoids conflating continued pretraining from the 2B terminal state with a true 2B-versus-10B data-budget comparison.
- Provides a practical ~5B escape hatch before spending the full 10B compute budget.
- Keeps model size fixed long enough to measure where data scaling begins to saturate in actual behavior.
- Avoids paying for a redundant standalone 5B run unless that endpoint later becomes necessary.

### Negative or limiting

- The 5B intermediate checkpoint is not a terminal 5B WSD model and must not be represented as one.
- Fresh 10B training repeats the first 2B tokens' worth of optimization work instead of reusing the existing 2B checkpoint.
- The experiment remains compute-intensive and should be terminated if the intermediate behavioral gate shows saturation.
- Exact 10B dataset construction, block geometry, Modal execution microbatch, and resulting update count remain implementation decisions to qualify before launch.

## Validation

Before launch:

1. Complete and verify the 100M / 2B final checkpoint.
2. Run the frozen full post-pretraining qualification and confirm a material behavioral/capability improvement sufficient to justify further fixed-100M data scaling.
3. Build and fully verify the deterministic approximately-10B dataset and its token-derived WSD plan.

During the run:

1. At approximately 5B consumed target tokens, preserve a durable checkpoint and run the same full qualification suite used for the 2B endpoint.
2. Compare downstream behavior/capability against the 100M / 2B endpoint; use loss/perplexity as supporting diagnostics rather than the sole continuation criterion.
3. Continue to 10B only if capability remains meaningfully improved; otherwise terminate the trajectory and treat the observed plateau as the fixed-100M data-scaling boundary.
4. If continued, fully qualify the terminal 10B checkpoint and then decide whether the next scaling axis should be model parameters, architecture, data quality/mixture, or post-training.

## Links

- [`0041-use-block64-modal-corpus-and-probe-microbatch-16-32-48-64.md`](0041-use-block64-modal-corpus-and-probe-microbatch-16-32-48-64.md)
- [`../current/roadmap.md`](../current/roadmap.md)
- [`../current/status.md`](../current/status.md)
- [`../runbooks/modal_training_launcher.md`](../runbooks/modal_training_launcher.md)
