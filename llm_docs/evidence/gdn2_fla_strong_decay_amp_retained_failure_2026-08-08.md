# GDN-2 FLA strong-decay AMP retained-intermediate failure — 2026-08-08

## Result

The focused retained-intermediate qualification probe was run after the AMP-realistic full-layer test had shown that normal-decay forward/backward passes but forced strong decay (`log_decay≈-6`) produces non-finite gradients.

Probe verdict reported by the user:

```text
Small-LLM FLA GDN-2 strong-decay AMP retained-intermediate probe

VERDICT: FAIL — retained-intermediate FLA is still not qualified for resumed training.
```

## What this rules out

The integrated backend had previously used FLA's default `disable_recompute=False`, so one plausible explanation was that the backward-only recomputation of WY/state intermediates was introducing the non-finite values.

The focused probe instead exercised the retained-intermediate path (`disable_recompute=True`). It still failed. Therefore the strong-decay AMP backward problem is not explained solely by FLA's recomputation branch.

## Current interpretation

- FLA v0.5.1 remains strongly forward-qualified on Tesla T4, including `log_decay=-6` and `-10`.
- Normal-decay AMP full-layer backward parity passes with FP32 parameters + FP16 autocast.
- Strong-decay AMP full-layer backward is not qualified: NaN/non-finite gradients appear under the forced `log_decay≈-6` stress case.
- Retaining forward intermediates does not remove that failure.
- The valid 500M training checkpoint remains verified step 4000; no FLA resume update has been accepted.

The next diagnostic should determine the failure boundary instead of treating constant `-6` as representative of all learned decay:

1. sweep forced log-decay values around the adaptive-backend danger region (for example `-0.5, -0.75, -1, -1.5, -2, -4, -6`) under the exact trainer AMP contract and check finite gradient parity;
2. inspect the real step-4000 checkpoint's per-token log-decay distribution and 64-token cumulative decay spans on representative training data;
3. compare the real checkpoint distribution with the FLA backward failure boundary;
4. separately benchmark/qualify FLA's fused-recurrent GDN-2 path as a possible exact-recurrence fallback before considering any learned-decay bound.

## Upstream context

As of 2026-08-08, FLA v0.5.1 remains the latest release. An upstream post-v0.5.1 PR (`#1007`, closed without merge) explored an opt-in bounded/safe GDN-2 gate path after a training failure involving extreme learned gate state. That evidence makes strong-decay numerical robustness an upstream-recognized issue, but it does not by itself justify changing Small-LLM decay semantics.
