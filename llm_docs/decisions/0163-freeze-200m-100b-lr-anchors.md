---
status: accepted
date: 2026-09-08
supersedes: null
owners: [Small-LLM]
---

# ADR 0163: freeze the 200M/100B numerical LR anchors

## Context and problem statement

ADR 0160 selects approximately 200M parameters / 100B target tokens as the next major pretraining trajectory. ADR 0161 requires a modest widening of every major LR anchor relative to the successful 100M/10B deep-decay run, and ADR 0162 freezes the phase positions by scaling the successful 10B token/update anchors by 10x.

The remaining LR decision is the exact numerical anchor values.

## Decision outcome

Chosen option: **use the previously proposed modestly widened LR anchors for the 200M/100B run:**

```text
peak LR:                     3.5e-4
aggressive-settle end LR:    8.0e-5
late long-decay anchor LR:   8.0e-6
final LR:                    4.0e-6
```

Relative to peak LR, the corresponding scheduler ratios are:

```text
settle_lr_ratio:             0.2285714285714286   (8/35)
cooldown_start_lr_ratio:     0.02285714285714286  (4/175)
minimum_lr_ratio:            0.01142857142857143  (2/175)
```

Combined with ADR 0162's fixed token anchors, the long calibrated power-law phase retains the same exponent as the successful 100M/10B deep-decay trajectory because the settle-to-cooldown LR ratio remains exactly 10x and the token-anchor ratio is unchanged by the 10x horizon scaling:

```text
base_power ~= 1.6270515945225403
```

The terminal cooldown remains linear from `8e-6` to `4e-6` over the final proportional approximately-4.0003B target-token span.

## Consequences

### Positive

- The schedule stays close to the empirically successful 100M/10B recipe while expanding the LR operating range modestly at every anchor.
- The central deep-decay shape is preserved exactly; only the absolute LR scale and endpoint ratios change.
- The exact scheduler ratios and power exponent are deterministic and checkpoint-serializable.

### Negative or limiting

- The higher `3.5e-4` peak and lower late-stage anchors are extrapolations to the 200M geometry and require stability validation.
- The proportional schedule retains a long peak-LR interval, so early validation must detect whether the 200M model becomes over-energetic before the aggressive settle begins.

## Validation

Before production launch:

1. verify the trainer reproduces all exact token/LR landmarks from ADRs 0162 and 0163;
2. exercise the 200M geometry at the `3.5e-4` peak in short GPU probes;
3. verify the start of the aggressive settle and scheduler continuity;
4. verify exact checkpoint/resume preserves scheduler state and LR phase;
5. confirm finite gradients, bounded overflow/clipping behavior, and improving loss.

## Links

- `llm_docs/decisions/0160-select-200m-100b-as-next-pretraining-run.md`
- `llm_docs/decisions/0161-broaden-200m-100b-lr-anchor-range-slightly.md`
- `llm_docs/decisions/0162-scale-100m-10b-lr-phase-anchors-proportionally-to-200m-100b.md`
- `llm_docs/decisions/0095-decay-1e-4-to-1e-5-then-5e-6.md`
