---
status: accepted
date: 2026-08-10
supersedes: null
---

# 0033 — Use comprehensive post-SFT qualification and pretraining-equivalent T4 cadence

## Context and problem statement

ADR 0032 authorized qualifying SFT on the completed 500M-parent checkpoint and switching to the 2B-parent checkpoint as soon as the latter is ready. The remaining questions were how to select/evaluate SFT checkpoints and whether SFT should reuse the already-qualified T4 operational geometry from pretraining.

A single retention threshold or SFT validation loss would be incomplete: post-training should be evaluated as a model that must both acquire instruction/chat behavior and retain measurable base-language capability. At the same time, the model geometry, context ceiling, precision path, and accelerator are unchanged, so there is no reason to introduce a second hardware-training geometry without evidence.

## Considered options

- Reuse only the pretraining validation/evaluation suite and select by held-out base loss.
- Evaluate only SFT held-out loss and instruction prompts.
- Build one post-SFT qualification suite that reports base/pretraining capability and instruction/chat capability side by side, without collapsing them into one arbitrary scalar.
- Re-probe SFT microbatch/checkpoint geometry from scratch.
- Reuse the qualified pretraining microbatch and 250-update operational cadence for the first SFT runs.

## Decision outcome

Chosen evaluation option: **build a comprehensive post-SFT qualification suite that evaluates the parent base checkpoint and each SFT checkpoint on the same report surface.**

The report must contain at least:

- unchanged `eval_core_v1` intrinsic/base metrics (loss, perplexity, BPB, top-k accuracy, calibration, cluster/position slices and throughput);
- the frozen base qualitative continuation/Q&A suite so regressions in pretrained behavior remain visible;
- held-out masked SFT validation loss/perplexity;
- deterministic instruction-following cases covering direct QA, transformations, extraction/formatting, explicit constraints, system-message adherence, multi-turn dialogue, concise elementary reasoning, uncertainty/correction and ordinary safe refusal behavior;
- generation diagnostics including EOS termination, maximum-budget/runaway rate, empty responses, role-label leakage, response length and repetition/degeneration indicators;
- per-category instruction success and an overall mechanically verifiable pass rate;
- parent-versus-SFT deltas for every comparable metric.

The comprehensive report is a scorecard, not a single master score. Final checkpoint selection remains evidence-based across capability gain and retention rather than being reduced to one predeclared weighted scalar.

Chosen T4 operational option: **reuse the pretraining-qualified defaults for the first SFT qualification/production comparison:**

```text
training microbatch:             4
local checkpoint cadence:        250 optimizer updates
validation cadence:              250 optimizer updates
remote publication cadence:      250 optimizer updates
precision/backend:               existing qualified CUDA FP16 + mixed FLA path
```

These values remain configurable launcher/profile fields so a measured SFT-specific failure can supersede them without changing trainer mathematics.

The canonical Kaggle SFT surface is one profile-driven entry point:

```text
kaggle/launch_sft.py
```

It should follow the same operational philosophy as the canonical pretraining launcher: explicit profiles/arguments, dry-run inspection, fail-closed dataset/checkpoint identities, automatic verified resume where possible, and no new per-run wrapper scripts.

## Consequences

### Positive

- SFT checkpoint evaluation becomes strictly more informative than either base-loss-only or SFT-loss-only selection.
- Parent and tuned checkpoints are directly comparable on one immutable report schema.
- The 500M qualification can establish the real behavior/retention curve before the 2B-parent SFT run.
- Reusing microbatch 4 and 250-update durability avoids an unnecessary hardware-geometry experiment while keeping the values configurable.
- One `launch_sft.py` prevents the wrapper proliferation that was removed from pretraining by ADR 0030.

### Negative or limiting

- The complete post-SFT suite is more expensive than one validation loss, so a fast/full mode is desirable.
- A comprehensive scorecard deliberately does not answer checkpoint selection with one scalar; the first 500M trajectory is still needed to establish what tradeoffs are acceptable.
- External public benchmarks are not automatically frozen by this ADR; the deterministic in-repo suite must work without network access, while later public-scorecard benchmarks can be added separately.

## Validation

1. Unit-test every deterministic verifier and chat serialization path.
2. Prove the behavior suite is deterministic under greedy generation and produces stable JSON identities.
3. Prove the same evaluator can score both the immutable parent and an SFT checkpoint and emit deltas.
4. Run the existing `eval_core_v1` tests unchanged to ensure the SFT suite reuses rather than forks base evaluation mathematics.
5. Run a T4 SFT smoke/qualification at microbatch 4 and confirm finite FP16/mixed-FLA execution and acceptable memory.
6. Interrupt after at least one completed optimizer block, restore the exact SFT checkpoint/data cursor, and prove next-block equivalence.
7. Verify local validation/checkpoint/remote-publication events occur at 250-update boundaries and at the final partial endpoint.

## Links

- [`0032-scale-sft-budget-with-pretraining-and-qualify-on-500m-first.md`](0032-scale-sft-budget-with-pretraining-and-qualify-on-500m-first.md)
- [`0030-consolidate-kaggle-profile-wrappers-behind-one-runtime.md`](0030-consolidate-kaggle-profile-wrappers-behind-one-runtime.md)
- [`../reference/post_training_sft.md`](../reference/post_training_sft.md)
- [`../current/roadmap.md`](../current/roadmap.md)
